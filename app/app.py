import sys, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import ecg_utils as eu

st.set_page_config(page_title="Apoyo a la detección de arritmias", layout="wide")
st.title("Apoyo a la detección de arritmias en cirugía")
st.caption("Herramienta de apoyo. No reemplaza el criterio clínico ni diagnostica.")

@st.cache_resource
def cargar():
    d = joblib.load(RAIZ / "models" / "modelo_arritmias.joblib")
    return d["modelo"], d["meta"]
modelo, meta = cargar()

NOMBRES = {"Normal": "Ritmo normal", "AFib": "Fibrilación auricular", "SVTA": "Taquicardia supraventricular",
           "TV": "Taquicardia ventricular", "Ectopia ventricular": "Latidos ventriculares prematuros",
           "Ectopia auricular": "Latidos auriculares prematuros", "Disfuncion nodo sinusal": "Disfunción del nodo sinusal",
           "WAP/MAT": "Ritmo auricular multifocal", "Ruido": "Señal con ruido (no interpretable)"}
ALERTA = {"AFib", "SVTA", "TV"}

col1, col2 = st.columns([2, 1])
with col2:
    fs = st.number_input("Frecuencia de muestreo (Hz)", 100, 1000, int(meta["fs"]))
    umbral = st.slider("Confianza mínima para no marcar revisión", 0.3, 0.95, 0.6)
    archivo = st.file_uploader("ECG en CSV (una columna con la señal)", type=["csv", "txt"])
if archivo is None:
    st.info("Sube un archivo para empezar.")
    st.stop()

x = pd.read_csv(archivo, header=None).iloc[:, -1].astype(float).values
if fs != meta["fs"]:
    from scipy.signal import resample
    x = resample(x, int(len(x) * meta["fs"] / fs)); fs = meta["fs"]
vent = int(meta["ventana_s"] * fs)
filas = []
for i in range(0, len(x) - vent + 1, vent):
    seg = x[i:i + vent]
    picos = eu.detectar_picos_r(seg, fs)
    f = eu.features_ventana(picos, seg if meta["usa_senal"] else None, fs)
    p = modelo.predict_proba(pd.DataFrame([f])[meta["columnas"]])[0]
    k = int(np.argmax(p))
    filas.append({"inicio_s": i / fs, "clase": modelo.classes_[k], "confianza": p[k], "fc": f["fc_media"]})
res = pd.DataFrame(filas)
if res.empty:
    st.error("La señal es más corta que una ventana."); st.stop()
res["Ritmo"] = res.clase.map(NOMBRES).fillna(res.clase)
res["Revisar"] = (res.confianza < umbral) | res.clase.isin(ALERTA)

with col1:
    t = np.arange(len(x)) / fs
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, eu.filtrar_ecg(x, fs), lw=0.5, color="k")
    for _, r in res.iterrows():
        c = "#d9534f" if r.clase in ALERTA else ("#999999" if r.clase == "Ruido" else "#5cb85c")
        ax.axvspan(r.inicio_s, r.inicio_s + meta["ventana_s"], color=c, alpha=0.18)
    ax.set_xlabel("segundos"); ax.set_yticks([])
    st.pyplot(fig)
st.subheader(f"Resumen por tramo de {meta['ventana_s']} segundos")
st.dataframe(res[["inicio_s", "Ritmo", "confianza", "fc", "Revisar"]].round(2), use_container_width=True)
n = int(res.Revisar.sum())
(st.warning if n else st.success)(f"{n} de {len(res)} tramos conviene revisarlos." if n else "Ningún tramo requiere revisión según el modelo.")
st.caption(f"Modelo {meta['modelo']} v{meta['version']}. Entrenado con VitalDB Arrhythmia (un hospital, Corea).")
