"""Genera los ejemplos de la demo: tres ECG de 1 minuto de pacientes del test y el monitor animado.

Salidas: app/ejemplos/ejemplo_*.csv (para la app de Streamlit) y demo/monitor.html (página autónoma).
Los tramos se eligen solo con las etiquetas de los médicos, sin mirar si el modelo acierta. Las predicciones son del modelo
de evaluación (models/modelo_evaluacion.joblib), que nunca vio a estos pacientes: la demo muestra aciertos y errores reales.

Uso (desde la raíz del proyecto, después de ejecutar el notebook): python generar_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
import ecg_utils as eu  # noqa: E402
from reentrenar import limpiar_anotaciones  # noqa: E402

DATA = RAIZ / "data"
DURACION_S, PASO_S, VENTANA_S, FS_DEMO = 60, 2, 10, 125  # el monitor predice cada 2 s con los 10 s anteriores

NOMBRES = {"Normal": "Ritmo normal", "AFib": "Fibrilación auricular", "SVTA": "Taquicardia supraventricular",
           "TV": "Taquicardia ventricular", "Ectopia ventricular": "Latidos ventriculares prematuros",
           "Ectopia auricular": "Latidos auriculares prematuros", "Disfuncion nodo sinusal": "Disfunción del nodo sinusal",
           "WAP/MAT": "Ritmo auricular multifocal", "Bloqueo AV": "Bloqueo auriculoventricular",
           "Ruido": "Señal con ruido", "Inclasificable": "Sin clasificar"}
GRUPO = {"Normal": "normal", "Ruido": "ruido", "Inclasificable": "ruido", "TV": "alerta", "Bloqueo AV": "alerta", "AFib": "alerta", "SVTA": "alerta"}

# Qué debe mostrar cada ejemplo, según las etiquetas de los médicos: ritmos obligatorios y fracción mínima de cada uno
ESCENAS = [
    {"id": "ectopias", "titulo": "Normal con ectopias", "requiere": {"Normal": 0.2, "Ectopia ventricular": 0.2},
     "nota": "Paciente con ritmo normal y latidos ventriculares prematuros."},
    {"id": "afib", "titulo": "Fibrilación auricular", "requiere": {"AFib": 0.9},
     "nota": "Paciente en fibrilación auricular sostenida."},
    {"id": "tv", "titulo": "Salva de TV", "requiere": {"TV": 0.02, "Normal": 0.2}, "en_medio": "TV",
     "nota": "Paciente con una salva breve de taquicardia ventricular."},
]


def cobertura(t, rit, ini, fin):
    """Fracción de [ini, fin) cubierta por cada ritmo (cada tramo va hasta el primer latido del ritmo siguiente)."""
    cambio = np.r_[True, rit[1:] != rit[:-1]]
    a, r = t[cambio], rit[cambio]
    b = np.r_[a[1:], t[-1]]
    return pd.Series(np.clip(np.minimum(b, fin) - np.maximum(a, ini), 0, None)).groupby(r).sum() / (fin - ini)


def elegir_tramos(ann, casos_test):
    """Para cada escena, tomar el primer tramo de 1 minuto de un caso del test que cumpla lo pedido (cada caso, una vez)."""
    usados, elegidos = set(), []
    for esc in ESCENAS:
        hallado = None
        for cid in casos_test:
            if cid in usados:
                continue
            g = ann[ann.case_id == cid]
            t, rit = g.time_second.values, g.ritmo.values
            ecg, off = eu.cargar_ecg_cache(cid, DATA / "ecg_cache")
            for ini in np.arange(np.ceil(max(t.min(), off)), t.max() - DURACION_S, 5.0):
                cob = cobertura(t, rit, ini, ini + DURACION_S)
                # en_medio: el episodio debe caer entre los segundos 20 y 45, para que se vea llegar y terminar
                centrado = "en_medio" not in esc or cobertura(t, rit, ini + 20, ini + 45).get(esc["en_medio"], 0) > 0
                if centrado and all(cob.get(r, 0) >= f for r, f in esc["requiere"].items()) and int((ini + DURACION_S - off) * eu.FS) <= len(ecg):
                    hallado = (cid, float(ini)); break
            if hallado:
                break
        if not hallado:
            raise SystemExit(f"Ningún caso del test cumple la escena {esc['id']}")
        usados.add(hallado[0]); elegidos.append({**esc, "case_id": int(hallado[0]), "ini": hallado[1]})
    return elegidos


def predecir(modelo, meta, x):
    """Predecir cada PASO_S segundos con los VENTANA_S segundos anteriores, como lo haría un monitor en vivo."""
    filas, fs = [], eu.FS
    for fin in range(VENTANA_S, DURACION_S + 1, PASO_S):
        seg = x[(fin - VENTANA_S) * fs:fin * fs]
        f = eu.features_ventana(eu.detectar_picos_r(seg, fs), seg, fs)
        p = modelo.predict_proba(pd.DataFrame([f])[meta["columnas"]])[0]
        k = int(np.argmax(p)); c = str(modelo.classes_[k])
        filas.append({"t": fin, "clase": c, "nombre": NOMBRES[c], "grupo": GRUPO.get(c, "revisar"), "conf": round(float(p[k]), 3),
                      "fc": None if np.isnan(f["fc_media"]) else round(float(f["fc_media"]), 1)})
    return filas


def main():
    d = joblib.load(RAIZ / "models" / "modelo_evaluacion.joblib")
    modelo, meta = d["modelo"], d["meta"]
    md = pd.read_csv(DATA / "metadata.csv")
    pl = pd.read_csv(DATA / "pliegues_pacientes.csv").set_index("subjectid")["pliegue"]
    md = md[~md.case_id.isin(meta.get("casos_excluidos", []))]
    casos_test = md[md.subjectid.map(pl) == 0].case_id.astype(int).tolist()
    ann = limpiar_anotaciones(DATA / "Annotation_Files", casos_test)

    ejemplos = RAIZ / "app" / "ejemplos"; ejemplos.mkdir(parents=True, exist_ok=True)
    casos, lista_ejemplos = [], []
    for e in elegir_tramos(ann, casos_test):
        ecg, off = eu.cargar_ecg_cache(e["case_id"], DATA / "ecg_cache")
        a = int(round((e["ini"] - off) * eu.FS))
        x = ecg[a:a + DURACION_S * eu.FS].astype(float)
        np.savetxt(ejemplos / f"ejemplo_{e['id']}.csv", x, fmt="%.4f")
        lista_ejemplos.append({"archivo": f"ejemplo_{e['id']}.csv", "titulo": e["titulo"], "nota": e["nota"], "case_id": e["case_id"]})

        # Etiqueta de los médicos, en tramos relativos al inicio del minuto
        g = ann[(ann.case_id == e["case_id"])]
        t, rit = g.time_second.values, g.ritmo.values
        cambio = np.r_[True, rit[1:] != rit[:-1]]
        i0, r0 = t[cambio], rit[cambio]
        f0 = np.r_[i0[1:], t[-1]]
        reales = [{"ini": max(float(i) - e["ini"], 0), "fin": min(float(f) - e["ini"], DURACION_S), "nombre": NOMBRES.get(r, r), "grupo": GRUPO.get(r, "revisar")}
                  for i, f, r in zip(i0, f0, r0) if f > e["ini"] and i < e["ini"] + DURACION_S]

        # Señal para dibujar: filtrada y reducida a 125 Hz, con una escala fija robusta a artefactos
        xf = eu.filtrar_ecg(x, eu.FS)[:: eu.FS // FS_DEMO]
        pred = predecir(modelo, meta, x)
        casos.append({"id": e["id"], "titulo": e["titulo"], "nota": f"Caso {e['case_id']}. {e['nota']}", "fs": FS_DEMO, "duracion": DURACION_S,
                      "escala": round(float(np.percentile(np.abs(xf), 99.8)) * 1.05, 3), "ecg": [round(float(v), 3) for v in xf],
                      "predicciones": pred, "reales": reales})
        print(f"{e['id']}: caso {e['case_id']}, segundo {e['ini']:.0f}. Reales: {sorted({r['nombre'] for r in reales})}. "
              f"Predichos: {pd.Series([p['clase'] for p in pred]).value_counts().to_dict()}")

    (ejemplos / "ejemplos.json").write_text(json.dumps(lista_ejemplos, indent=2, ensure_ascii=False), encoding="utf-8")
    datos = {"casos": casos, "nota": "Pacientes del test: el modelo mostrado nunca los vio. Los tramos se eligieron solo por sus etiquetas, sin mirar si el modelo acierta, "
                                     "así que también se ven sus errores. La predicción se calcula cada 2 s con los 10 s anteriores."}
    plantilla = (RAIZ / "demo" / "plantilla_monitor.html").read_text(encoding="utf-8")
    (RAIZ / "demo" / "monitor.html").write_text(plantilla.replace("/*DATOS*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print("Escrito demo/monitor.html y", len(lista_ejemplos), "ejemplos en app/ejemplos/")


if __name__ == "__main__":
    main()
