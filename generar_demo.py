"""Genera los ejemplos de la demo: cinco ECG de 1 minuto de pacientes del test y el monitor animado.

Salidas: app/ejemplos/ejemplo_*.csv (para la app de Streamlit) y demo/monitor.html (página autónoma).
Las predicciones son del modelo de evaluación (models/modelo_evaluacion.joblib), que nunca vio a estos pacientes.

Selección de los tramos (declarada también en la demo):
- Cuatro escenas ilustrativas se eligen MIRANDO las predicciones: entre los minutos con suficiente arritmia, el que el
  modelo clasifica mejor. Muestran cómo se ve el sistema cuando acierta; no representan su desempeño promedio.
- Una escena de una clase difícil se elige solo por su etiqueta, sin mirar al modelo, para mostrar sus límites.

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
from reentrenar import EVENTOS_CORTOS, limpiar_anotaciones  # noqa: E402

DATA = RAIZ / "data"
# El monitor predice cada 2 s con los 10 s anteriores. PREVIO_S: segundos de señal antes del minuto mostrado, para que
# desde el segundo 0 ya haya una ventana completa (como un monitor que lleva rato encendido)
DURACION_S, PASO_S, VENTANA_S, PREVIO_S, FS_DEMO = 60, 2, 10, 10, 125
CANDIDATOS_POR_CASO = 8   # minutos que se evalúan con el modelo en cada caso (los de mayor cobertura de la arritmia)

NOMBRES = {"Normal": "Ritmo normal", "AFib": "Fibrilación auricular", "SVTA": "Taquicardia supraventricular",
           "TV": "Taquicardia ventricular", "Ectopia ventricular": "Latidos ventriculares prematuros",
           "Ectopia auricular": "Latidos auriculares prematuros", "Disfuncion nodo sinusal": "Disfunción del nodo sinusal",
           "WAP/MAT": "Ritmo auricular multifocal", "Bloqueo AV": "Bloqueo auriculoventricular",
           "Ruido": "Señal con ruido", "Inclasificable": "Sin clasificar"}

# min_cob: fracción mínima del minuto con la arritmia, según los médicos (se relaja si ningún caso la cumple).
# elegir: "modelo" = el minuto mejor clasificado; "etiqueta" = el de mayor cobertura, sin mirar al modelo.
ESCENAS = [
    {"id": "afib", "titulo": "Fibrilación auricular", "clase": "AFib", "min_cob": 0.9, "elegir": "modelo",
     "nota": "Paciente en fibrilación auricular sostenida."},
    {"id": "ectopias", "titulo": "Latidos ventriculares prematuros", "clase": "Ectopia ventricular", "min_cob": 0.6, "elegir": "modelo",
     "nota": "Paciente con latidos ventriculares prematuros frecuentes."},
    {"id": "tv", "titulo": "Taquicardia ventricular", "clase": "TV", "min_cob": 0.15, "elegir": "modelo",
     "nota": "Paciente con salvas de taquicardia ventricular."},
    {"id": "svta", "titulo": "Taquicardia supraventricular", "clase": "SVTA", "min_cob": 0.15, "elegir": "modelo",
     "nota": "Paciente con episodios de taquicardia supraventricular."},
    {"id": "bloqueo", "titulo": "Bloqueo AV (clase difícil)", "clase": "Bloqueo AV", "min_cob": 0.6, "elegir": "etiqueta",
     "nota": "Paciente con bloqueo auriculoventricular. Es una de las clases que el modelo casi no reconoce: este tramo se eligió sin mirar sus predicciones."},
]


def tramos_ritmo(t, rit):
    """Tramos continuos de un mismo ritmo: cada uno va hasta el primer latido del ritmo siguiente."""
    cambio = np.r_[True, rit[1:] != rit[:-1]]
    a, r = t[cambio], rit[cambio]
    return a, np.r_[a[1:], t[-1]], r


def cobertura(tramos, ini, fin):
    """Fracción de [ini, fin) cubierta por cada ritmo."""
    a, b, r = tramos
    return pd.Series(np.clip(np.minimum(b, fin) - np.maximum(a, ini), 0, None)).groupby(r).sum() / (fin - ini)


def etiqueta_ventana(t, rit, tramos, fin):
    """Etiqueta de los médicos para la ventana [fin-10, fin), con la prioridad de la sección 8.1 para eventos cortos."""
    m = (t >= fin - VENTANA_S) & (t < fin) & (rit != "Ruido")
    n = pd.Series(rit[m]).value_counts()
    corto = next((r for r, k in EVENTOS_CORTOS.items() if n.get(r, 0) >= k), None)
    return corto or cobertura(tramos, fin - VENTANA_S, fin).idxmax()


class Predictor:
    """Predice ventanas de 10 s de un caso con el modelo, guardando lo ya calculado (los candidatos se solapan)."""

    def __init__(self, modelo, meta):
        self.modelo, self.meta, self.cache = modelo, meta, {}

    def __call__(self, cid, ecg, off, fines):
        nuevos = [f for f in fines if (cid, f) not in self.cache]
        if nuevos:
            filas = []
            for f in nuevos:
                a = int(round((f - VENTANA_S - off) * eu.FS))
                seg = ecg[a:a + VENTANA_S * eu.FS].astype(float)
                filas.append(eu.features_ventana(eu.detectar_picos_r(seg, eu.FS), seg, eu.FS))
            X = pd.DataFrame(filas)
            proba = self.modelo.predict_proba(X[self.meta["columnas"]])
            for f, fila, p in zip(nuevos, filas, proba):
                k = int(np.argmax(p))
                self.cache[(cid, f)] = (str(self.modelo.classes_[k]), float(p[k]), fila["fc_media"])
        return [self.cache[(cid, f)] for f in fines]


def elegir_tramos(ann, casos_test, predictor):
    """Para cada escena, el mejor minuto de un caso del test según su criterio (cada caso se usa una sola vez)."""
    usados, elegidos = set(), []
    for esc in ESCENAS:
        min_cob, hallado = esc["min_cob"], None
        while hallado is None and min_cob >= 0.02:
            candidatos = []
            for cid in casos_test:
                g = ann[ann.case_id == cid]
                if cid in usados or not (g.ritmo == esc["clase"]).any():
                    continue
                ecg, off = eu.cargar_ecg_cache(cid, DATA / "ecg_cache")
                if ecg is None:
                    continue
                t, rit = g.time_second.values, g.ritmo.values
                tramos = tramos_ritmo(t, rit)
                # inicios en segundos pares, con PREVIO_S de señal y anotación antes, y el minuto completo dentro del ECG
                ini0 = 2 * np.ceil((max(t.min(), off) + PREVIO_S) / 2)
                fin_max = min(t.max(), off + len(ecg) / eu.FS)
                inis = np.arange(ini0, fin_max - DURACION_S, 4.0)
                cobs = [cobertura(tramos, i, i + DURACION_S).get(esc["clase"], 0) for i in inis]
                orden = [j for j in np.argsort(cobs)[::-1] if cobs[j] >= min_cob][:CANDIDATOS_POR_CASO]
                candidatos += [(cid, float(inis[j]), float(cobs[j]), ecg, off, t, rit, tramos) for j in orden]
            if not candidatos:
                min_cob /= 2
                continue
            if esc["elegir"] == "etiqueta":
                hallado = max(candidatos, key=lambda c: c[2])
                puntaje = None
            else:
                mejor = None
                for c in candidatos:
                    cid, ini, cob, ecg, off, t, rit, tramos = c
                    fines = [ini + s for s in range(0, DURACION_S + 1, PASO_S)]
                    pred = [p[0] for p in predictor(cid, ecg, off, fines)]
                    real = [etiqueta_ventana(t, rit, tramos, f) for f in fines]
                    # puntaje: aciertos en la arritmia buscada menos errores de cualquier tipo, por predicción
                    ac = sum(p == r == esc["clase"] for p, r in zip(pred, real)) / len(fines)
                    err = sum(p != r for p, r in zip(pred, real)) / len(fines)
                    clave = (ac - err, cob)
                    if mejor is None or clave > mejor[0]:
                        mejor = (clave, c)
                puntaje, hallado = mejor[0][0], mejor[1]
        if hallado is None:
            raise SystemExit(f"Ningún caso del test tiene {esc['clase']}")
        usados.add(hallado[0])
        elegidos.append({**esc, "case_id": int(hallado[0]), "ini": hallado[1], "cob": hallado[2], "min_cob_usada": min_cob, "puntaje": puntaje})
    return elegidos


def main():
    d = joblib.load(RAIZ / "models" / "modelo_evaluacion.joblib")
    modelo, meta = d["modelo"], d["meta"]
    predictor = Predictor(modelo, meta)
    md = pd.read_csv(DATA / "metadata.csv")
    pl = pd.read_csv(DATA / "pliegues_pacientes.csv").set_index("subjectid")["pliegue"]
    md = md[~md.case_id.isin(meta.get("casos_excluidos", []))]
    casos_test = md[md.subjectid.map(pl) == 0].case_id.astype(int).tolist()
    ann = limpiar_anotaciones(DATA / "Annotation_Files", casos_test)

    ejemplos = RAIZ / "app" / "ejemplos"; ejemplos.mkdir(parents=True, exist_ok=True)
    for viejo in ejemplos.glob("ejemplo_*.csv"):
        viejo.unlink()   # quitar ejemplos de selecciones anteriores
    casos, lista_ejemplos = [], []
    for e in elegir_tramos(ann, casos_test, predictor):
        cid, ini = e["case_id"], e["ini"]
        ecg, off = eu.cargar_ecg_cache(cid, DATA / "ecg_cache")
        a = int(round((ini - PREVIO_S - off) * eu.FS))
        x = ecg[a:a + (PREVIO_S + DURACION_S) * eu.FS].astype(float)       # señal desde PREVIO_S antes del minuto
        np.savetxt(ejemplos / f"ejemplo_{e['id']}.csv", x[PREVIO_S * eu.FS:], fmt="%.4f")   # la app recibe solo el minuto
        lista_ejemplos.append({"archivo": f"ejemplo_{e['id']}.csv", "titulo": e["titulo"], "nota": e["nota"], "case_id": cid})

        # Etiqueta de los médicos, en tramos relativos al inicio del minuto
        g = ann[ann.case_id == cid]
        t, rit = g.time_second.values, g.ritmo.values
        a0, b0, r0 = tramos_ritmo(t, rit)
        reales = [{"ini": max(float(i) - ini, 0), "fin": min(float(f) - ini, DURACION_S), "clase": r, "nombre": NOMBRES.get(r, r)}
                  for i, f, r in zip(a0, b0, r0) if f > ini and i < ini + DURACION_S]

        # Predicción cada 2 s (desde el segundo 0, gracias a los PREVIO_S segundos anteriores)
        fines = [ini + s for s in range(0, DURACION_S + 1, PASO_S)]
        pred = [{"t": int(f - ini), "clase": c, "nombre": NOMBRES[c], "conf": round(p, 3), "fc": None if np.isnan(fc) else round(float(fc), 1)}
                for f, (c, p, fc) in zip(fines, predictor(cid, ecg, off, fines))]

        # Picos R detectados en toda la señal, para el pitido del monitor (segundos relativos al inicio del minuto)
        picos = [round(float(p) - PREVIO_S, 3) for p in eu.detectar_picos_r(x, eu.FS)]

        # Señal para dibujar: filtrada y reducida a 125 Hz, con una escala fija robusta a artefactos
        xf = eu.filtrar_ecg(x, eu.FS)[:: eu.FS // FS_DEMO]
        casos.append({"id": e["id"], "titulo": e["titulo"], "nota": f"Caso {cid}. {e['nota']}", "fs": FS_DEMO, "duracion": DURACION_S,
                      "previo": PREVIO_S, "escala": round(float(np.percentile(np.abs(xf), 99.8)) * 1.05, 3),
                      "ecg": [round(float(v), 3) for v in xf], "picos": picos, "predicciones": pred, "reales": reales})
        criterio = "por etiqueta" if e["puntaje"] is None else f"por modelo, puntaje {e['puntaje']:.2f}"
        print(f"{e['id']}: caso {cid}, segundo {ini:.0f}, {e['clase']} cubre {e['cob']:.0%} del minuto "
              f"(mínimo usado {e['min_cob_usada']:.0%}; {criterio}). Reales: {sorted({r['nombre'] for r in reales})}. "
              f"Predichos: {pd.Series([p['clase'] for p in pred]).value_counts().to_dict()}")

    (ejemplos / "ejemplos.json").write_text(json.dumps(lista_ejemplos, indent=2, ensure_ascii=False), encoding="utf-8")
    datos = {"casos": casos,
             "nota": "Pacientes del test: el modelo mostrado nunca los vio. Los cuatro primeros tramos se eligieron entre los que el modelo "
                     "clasifica mejor, para ilustrar; no representan su desempeño promedio (ver el notebook). El de Bloqueo AV se eligió "
                     "solo por su etiqueta. Cada predicción usa los 10 s anteriores, por eso cambia unos segundos después que la etiqueta."}
    plantilla = (RAIZ / "demo" / "plantilla_monitor.html").read_text(encoding="utf-8")
    (RAIZ / "demo" / "monitor.html").write_text(plantilla.replace("/*DATOS*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print("Escrito demo/monitor.html y", len(lista_ejemplos), "ejemplos en app/ejemplos/")


if __name__ == "__main__":
    main()
