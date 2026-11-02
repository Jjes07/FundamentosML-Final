"""Pipeline de reentrenamiento (sección 16 del notebook).

Con datos nuevos etiquetados construye las ventanas con el mismo src/ecg_utils.py, entrena un candidato con los
hiperparámetros vigentes, lo compara con el modelo actual sobre un conjunto de validación fijo y solo lo promueve
si no empeora. Cada versión promovida queda guardada con su fecha y sus métricas.

Uso (desde la raíz del proyecto):
    python reentrenar.py --nuevos carpeta_nuevos          # metadata.csv, Annotation_Files/ y ecg_cache/, como en data/
    python reentrenar.py --simular-nuevos 60 --sin-guardar  # ensayo: 60 pacientes actuales hacen de datos nuevos
    python reentrenar.py --simular-nuevos 60 --sin-guardar --dañar-etiquetas  # ensayo: el candidato debe rechazarse

La regla de promoción y el conjunto de validación están explicados en la sección 16 del notebook.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
import ecg_utils as eu  # noqa: E402

DATA = RAIZ / "data"
MODELOS = RAIZ / "models"
PLIEGUE_VALIDACION = 0        # pacientes reservados: nunca entrenan al candidato, para comparar siempre contra lo mismo
PELIGROSAS = ["TV", "Bloqueo AV", "AFib"]
TOLERANCIA = 0.01             # caída tolerada por variación de muestreo en el F1 macro y en el recall de las peligrosas
MAX_POR_CLASE = 5000
MIN_LATIDOS, PUREZA_MIN = 3, 0.80
EVENTOS_CORTOS = {"TV": 3, "SVTA": 3}
SEMILLA = 42


def limpiar_anotaciones(carpeta: Path, ids) -> pd.DataFrame:
    """Leer y limpiar las anotaciones igual que la sección 5: sin filas marcadoras, sin duplicados y con ritmo unificado."""
    partes = []
    for cid in ids:
        ruta = carpeta / f"Annotation_file_{cid}.csv"
        if ruta.exists():
            d = pd.read_csv(ruta).drop(columns=["caseid"], errors="ignore")
            d.insert(0, "case_id", cid)
            partes.append(d)
    ann = pd.concat(partes, ignore_index=True)
    marcador = ann["beat_type"].isin(["Start", "End"]) | (ann["beat_type"].isna() & ann["rhythm_label"].ne("Noise"))
    ann = ann[~marcador].dropna(subset=["time_second"]).sort_values(["case_id", "time_second"])
    ann = ann[~ann.duplicated(subset=["case_id", "time_second", "rhythm_label"])].reset_index(drop=True)
    ann["ritmo"] = ann["rhythm_label"].map(eu.normalizar_ritmo).replace("Sin etiqueta", np.nan)
    ann["ritmo"] = ann.groupby("case_id")["ritmo"].transform(lambda s: s.ffill().bfill())
    return ann


def ventanas_de_un_caso(cid, sid, g, ecg_dir: Path, ventana_s=10) -> list[dict]:
    """Etiquetar y describir las ventanas de un caso con la regla de la sección 8.1."""
    ecg, off = eu.cargar_ecg_cache(cid, ecg_dir)
    if ecg is None:
        return []
    t, rit = g.time_second.values, g.ritmo.values
    cambio = np.r_[True, rit[1:] != rit[:-1]]
    seg_ini, seg_rit = t[cambio], rit[cambio]
    seg_fin = np.r_[seg_ini[1:], t[-1]]
    filas = []
    for ini in np.arange(np.floor(t.min()), t.max() - ventana_s + 1e-9, ventana_s):
        fin = ini + ventana_s
        m = (t >= ini) & (t < fin)
        if not m.any():
            continue
        cobertura = pd.Series(np.clip(np.minimum(seg_fin, fin) - np.maximum(seg_ini, ini), 0, None)).groupby(seg_rit).sum() / ventana_s
        lat_m = m & (rit != "Ruido")
        n_por_ritmo = pd.Series(rit[lat_m]).value_counts()
        etiqueta = next((r for r, n in EVENTOS_CORTOS.items() if n_por_ritmo.get(r, 0) >= n), None)
        if etiqueta is None:
            etiqueta = cobertura.idxmax()
            if cobertura.max() < PUREZA_MIN or (etiqueta != "Ruido" and lat_m.sum() < MIN_LATIDOS):
                continue
        a, b = int(round((ini - off) * eu.FS)), int(round((fin - off) * eu.FS))
        if a < 0 or b > len(ecg) or etiqueta == "Inclasificable":
            continue
        x = ecg[a:b]
        f = eu.features_ventana(eu.detectar_picos_r(x, eu.FS), x, eu.FS)
        f.update(case_id=cid, subjectid=sid, t_ini=ini, etiqueta=etiqueta)
        filas.append(f)
    return filas


def construir_ventanas(carpeta: Path, excluir=()) -> pd.DataFrame:
    """Construir todas las ventanas de una carpeta con la misma estructura que data/."""
    meta = pd.read_csv(carpeta / "metadata.csv")
    meta = meta[~meta.case_id.isin(excluir)]
    ann = limpiar_anotaciones(carpeta / "Annotation_Files", meta.case_id).merge(meta[["case_id", "subjectid"]], on="case_id")
    partes = Parallel(n_jobs=-1)(delayed(ventanas_de_un_caso)(cid, int(g.subjectid.iloc[0]), g, carpeta / "ecg_cache")
                                 for cid, g in ann.groupby("case_id"))
    return pd.DataFrame([f for p in partes for f in p])


def tope(df: pd.DataFrame) -> pd.DataFrame:
    """Recortar cada clase a MAX_POR_CLASE ventanas: solo se aplica a lo que entrena."""
    return pd.concat([g.sample(min(len(g), MAX_POR_CLASE), random_state=SEMILLA) for _, g in df.groupby("etiqueta")])


def metricas(modelo, df: pd.DataFrame, cols) -> dict:
    """F1 macro y recall de las clases peligrosas sobre un conjunto de ventanas."""
    p = modelo.predict(df[cols])
    rec = recall_score(df.etiqueta, p, labels=PELIGROSAS, average=None, zero_division=0)
    return {"f1_macro": float(f1_score(df.etiqueta, p, average="macro", zero_division=0)),
            **{f"recall_{c}": float(r) for c, r in zip(PELIGROSAS, rec)}}


def decidir(actual: dict, candidato: dict, tolerancia: float = TOLERANCIA) -> tuple[bool, list[str]]:
    """Promover solo si el candidato no empeora el F1 macro ni el recall de ninguna clase peligrosa."""
    motivos = [f"{k}: {actual[k]:.3f} -> {candidato[k]:.3f}" for k in actual if candidato[k] < actual[k] - tolerancia]
    return not motivos, motivos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuevos", type=Path, help="carpeta con metadata.csv, Annotation_Files/ y ecg_cache/")
    ap.add_argument("--simular-nuevos", type=int, default=0, help="ensayo: N pacientes actuales hacen de datos nuevos")
    ap.add_argument("--dañar-etiquetas", action="store_true", help="ensayo: baraja las etiquetas de los datos nuevos")
    ap.add_argument("--sin-guardar", action="store_true", help="comparar sin escribir nada en models/")
    a = ap.parse_args()
    if bool(a.nuevos) == bool(a.simular_nuevos):
        ap.error("indica --nuevos o --simular-nuevos")
    t0 = time.time()

    # 1) Ventanas y características, con el mismo código de ecg_utils.py que usa la app
    actual = joblib.load(MODELOS / "modelo_evaluacion.joblib")
    cols = actual["meta"]["columnas"]
    excluidos = actual["meta"].get("casos_excluidos", [])
    print("1/5 Construyendo ventanas de los datos actuales...", flush=True)
    base = construir_ventanas(DATA, excluir=excluidos)
    pliegues = pd.read_csv(DATA / "pliegues_pacientes.csv").set_index("subjectid")["pliegue"]
    base["pliegue"] = base.subjectid.map(pliegues).fillna(1).astype(int)
    validacion = base[base.pliegue == PLIEGUE_VALIDACION]
    entrenamiento = base[base.pliegue != PLIEGUE_VALIDACION]

    if a.nuevos:
        print("    Construyendo ventanas de los datos nuevos...", flush=True)
        nuevos = construir_ventanas(a.nuevos)
        # Un paciente reservado para validar no puede entrenar al candidato aunque traiga una cirugía nueva
        nuevos = nuevos[~nuevos.subjectid.isin(validacion.subjectid)]
        ref = actual["modelo"]
    else:
        # Ensayo: los pacientes elegidos salen del entrenamiento y el modelo de referencia se entrena sin ellos
        rng = np.random.default_rng(SEMILLA)
        elegidos = rng.choice(entrenamiento.subjectid.unique(), a.simular_nuevos, replace=False)
        nuevos = entrenamiento[entrenamiento.subjectid.isin(elegidos)].copy()
        entrenamiento = entrenamiento[~entrenamiento.subjectid.isin(elegidos)]
        d = tope(entrenamiento)
        ref = clone(actual["modelo"]).fit(d[cols], d.etiqueta)
    if a.dañar_etiquetas:
        nuevos["etiqueta"] = np.random.default_rng(SEMILLA).permutation(nuevos.etiqueta.values)
    if nuevos.empty:
        sys.exit("No hay ventanas nuevas que aporten pacientes distintos a los de validación.")
    print(f"    Entrenamiento: {entrenamiento.subjectid.nunique()} pacientes | nuevos: {nuevos.subjectid.nunique()} pacientes, "
          f"{len(nuevos)} ventanas | validación fija: {validacion.subjectid.nunique()} pacientes, {len(validacion)} ventanas")

    # 2) Desempeño del modelo actual en los datos nuevos, que nunca vio (señal de degradación). Por paciente: no hay fuga.
    print("2/5 Modelo actual sobre los datos nuevos:", {k: round(v, 3) for k, v in metricas(ref, nuevos, cols).items()}, flush=True)

    # 3) Candidato: mismos hiperparámetros, entrenado con lo anterior más los pacientes nuevos
    print("3/5 Entrenando el candidato...", flush=True)
    d = tope(pd.concat([entrenamiento, nuevos]))
    candidato = clone(actual["modelo"]).fit(d[cols], d.etiqueta)

    # 4) Comparación sobre la validación fija y decisión
    m_actual, m_cand = metricas(ref, validacion, cols), metricas(candidato, validacion, cols)
    promover, motivos = decidir(m_actual, m_cand)
    print("4/5 Validación fija:")
    print(pd.DataFrame({"actual": m_actual, "candidato": m_cand}).round(3).to_string())
    print("    Decisión:", "PROMOVER" if promover else "RECHAZAR. Empeora " + "; ".join(motivos), flush=True)

    # 5) Promoción: guardar la versión anterior y la nueva, con fecha y métricas
    if not promover or a.sin_guardar:
        print(f"5/5 No se escribe nada en models/ ({'ensayo' if a.sin_guardar else 'candidato rechazado'}). {time.time() - t0:.0f}s")
        return
    version = time.strftime("%Y.%m.%d-%H%M")
    versiones = MODELOS / "versiones"; versiones.mkdir(exist_ok=True)
    for nombre in ("modelo_arritmias.joblib", "modelo_evaluacion.joblib"):
        if (MODELOS / nombre).exists():
            shutil.copy2(MODELOS / nombre, versiones / f"{Path(nombre).stem}_antes_de_{version}.joblib")
    todo = tope(pd.concat([base, nuevos]))
    meta = {**actual["meta"], "version": version, "f1_macro_test": m_cand["f1_macro"], "n_ventanas": int(len(todo)),
            "n_pacientes": int(todo.subjectid.nunique())}
    joblib.dump({"modelo": clone(actual["modelo"]).fit(todo[cols], todo.etiqueta), "meta": meta}, MODELOS / "modelo_arritmias.joblib", compress=3)
    joblib.dump({"modelo": candidato, "meta": meta}, MODELOS / "modelo_evaluacion.joblib", compress=3)
    (MODELOS / "meta_modelo.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    ruta_h = MODELOS / "historial_versiones.json"
    historial = json.loads(ruta_h.read_text(encoding="utf-8")) if ruta_h.exists() else []
    historial.append({"version": version, "metricas_validacion": m_cand, "metricas_anterior": m_actual, "pacientes_nuevos": int(nuevos.subjectid.nunique())})
    ruta_h.write_text(json.dumps(historial, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"5/5 Versión {version} promovida y guardada en models/. {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
