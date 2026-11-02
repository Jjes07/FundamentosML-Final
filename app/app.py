import sys, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import ecg_utils as eu

st.set_page_config(page_title="Apoyo a la detección de arritmias", layout="wide")
st.title("Apoyo a la detección de arritmias en cirugía")
st.caption("Herramienta de apoyo. No reemplaza el criterio clínico ni diagnostica.")

@st.cache_resource
def cargar(nombre):
    d = joblib.load(RAIZ / "models" / nombre)
    return d["modelo"], d["meta"]

NOMBRES = {"Normal": "Ritmo normal", "AFib": "Fibrilación auricular", "SVTA": "Taquicardia supraventricular",
           "TV": "Taquicardia ventricular", "Ectopia ventricular": "Latidos ventriculares prematuros",
           "Ectopia auricular": "Latidos auriculares prematuros", "Disfuncion nodo sinusal": "Disfunción del nodo sinusal",
           "WAP/MAT": "Ritmo auricular multifocal", "Bloqueo AV": "Bloqueo auriculoventricular",
           "Ruido": "Señal con ruido (no interpretable)"}
ALERTA = {"AFib", "SVTA", "TV", "Bloqueo AV"}     # arritmias que requieren atención inmediata
COLOR = {"Alerta": "#d9363e", "Revisar": "#d99a00", "Sin alerta": "#2f9e62", "Ruido": "#8a918d"}

ruta_ejemplos = RAIZ / "app" / "ejemplos" / "ejemplos.json"
ejemplos = json.loads(ruta_ejemplos.read_text(encoding="utf-8")) if ruta_ejemplos.exists() else []

col_datos, col_ajustes = st.columns([2, 1])
with col_datos:
    origen = st.radio("¿Qué ECG quiere analizar?", ["Un ejemplo incluido", "Mi propio archivo"], horizontal=True)
    archivo, ejemplo = None, None
    if origen == "Un ejemplo incluido":
        titulos = {e["titulo"]: e for e in ejemplos}
        if not titulos:
            st.error("No hay ejemplos. Ejecute generar_demo.py."); st.stop()
        ejemplo = titulos[st.selectbox("Ejemplo", list(titulos))]
        st.caption(ejemplo["nota"] + " El análisis lo hace una copia del modelo que nunca vio a este paciente.")
    else:
        archivo = st.file_uploader("ECG en CSV o TXT", type=["csv", "txt"])
with col_ajustes:
    fs = st.number_input("Frecuencia de muestreo (Hz)", 100, 1000, 500, help="Muestras por segundo de su archivo. Los ejemplos usan 500 Hz.")
    umbral = st.slider("Confianza mínima", 0.3, 0.95, 0.6, help="Los tramos con menos confianza se marcan para revisión.")
with st.expander("Formato del archivo y cómo funcionaría en un monitor"):
    st.markdown("""
- **Archivo:** una columna con el voltaje del ECG en derivación II (mV), un valor por fila y sin encabezado. Si hay varias columnas se usa la última. Mínimo 10 segundos.
- **Muestreo:** indique la frecuencia real; la señal se convierte a 500 Hz, que es la usada para entrenar.
- **Monitor real:** un monitor transmite la señal en vivo por protocolos del fabricante o HL7. Una integración recibiría ese flujo, guardaría los últimos 10 s en memoria y repetiría este mismo análisis. El CSV es para la demostración.
""")

if ejemplo:
    x = pd.read_csv(RAIZ / "app" / "ejemplos" / ejemplo["archivo"], header=None)
    modelo, meta = cargar("modelo_evaluacion.joblib")
elif archivo is not None:
    try:
        x = pd.read_csv(archivo, header=None)
    except Exception:
        st.error("No se pudo leer el archivo. Revise que sea un CSV o TXT con una columna de números."); st.stop()
    modelo, meta = cargar("modelo_arritmias.joblib")
else:
    st.info("Suba un archivo para empezar, o elija un ejemplo."); st.stop()

x = pd.to_numeric(x.iloc[:, -1], errors="coerce").values.astype(float)
if np.isnan(x).mean() > 0.5:
    st.error("No se encontró una columna numérica con la señal. Revise el formato del archivo."); st.stop()
if fs != meta["fs"]:
    from scipy.signal import resample
    x = resample(np.nan_to_num(x, nan=float(np.nanmedian(x))), int(len(x) * meta["fs"] / fs))
fs = meta["fs"]
vent = int(meta["ventana_s"] * fs)
if len(x) < vent:
    st.error(f"La señal dura {len(x) / fs:.1f} s y se necesitan al menos {meta['ventana_s']} s."); st.stop()

filas = []
for i in range(0, len(x) - vent + 1, vent):
    seg = x[i:i + vent]
    picos = eu.detectar_picos_r(seg, fs)
    f = eu.features_ventana(picos, seg if meta["usa_senal"] else None, fs)
    p = modelo.predict_proba(pd.DataFrame([f])[meta["columnas"]])[0]
    k = int(np.argmax(p))
    filas.append({"Inicio (s)": int(i / fs), "clase": str(modelo.classes_[k]), "Confianza": float(p[k]), "fc": f["fc_media"]})
res = pd.DataFrame(filas)
res["Ritmo"] = res.clase.map(NOMBRES).fillna(res.clase)
res["Frecuencia (lpm)"] = res.fc.round(0)
res["Estado"] = np.select([res.clase.isin(ALERTA), res.clase.eq("Ruido"), res.clase.ne("Normal") | (res.Confianza < umbral)],
                          ["Alerta", "Ruido", "Revisar"], default="Sin alerta")

# Primero la conclusión, luego el detalle
n = res.Estado.value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tramos analizados", len(res))
c2.metric("Con alerta", int(n.get("Alerta", 0)))
c3.metric("Para revisar", int(n.get("Revisar", 0)))
c4.metric("Señal con ruido", int(n.get("Ruido", 0)))
if n.get("Alerta", 0):
    primera = res[res.Estado == "Alerta"].iloc[0]
    st.error(f"Alerta: {primera.Ritmo} a partir del segundo {primera['Inicio (s)']}. Confirme en el monitor antes de actuar.", icon="⚠️")
elif n.get("Revisar", 0):
    st.warning("Hay tramos que conviene revisar.", icon="🔎")
else:
    st.success("Ningún tramo requiere revisión según el modelo.", icon="✅")

t = np.arange(len(x)) / fs
xf = eu.filtrar_ecg(x, fs)
fig, ax = plt.subplots(figsize=(13, 3.2))
for _, r in res.iterrows():
    ax.axvspan(r["Inicio (s)"], r["Inicio (s)"] + meta["ventana_s"], color=COLOR[r.Estado], alpha=0.2, lw=0)
ax.plot(t, xf, lw=0.5, color="#1c1f1e")
lim = np.percentile(np.abs(xf), 99.8) * 1.15
ax.set_ylim(-lim, lim); ax.set_xlim(0, t[-1]); ax.set_xlabel("Segundos"); ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(color=COLOR[e], alpha=0.35, label=e) for e in COLOR if e in set(res.Estado)], loc="upper center",
          bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False)
fig.tight_layout()
st.pyplot(fig)

st.subheader(f"Detalle por tramo de {meta['ventana_s']} segundos")
tabla = res[["Inicio (s)", "Estado", "Ritmo", "Frecuencia (lpm)", "Confianza"]]
st.dataframe(tabla.style.format({"Confianza": "{:.0%}", "Frecuencia (lpm)": "{:.0f}"}, na_rep="sin dato"), width="stretch", hide_index=True)
st.download_button("Descargar resultados (CSV)", tabla.to_csv(index=False).encode("utf-8"), "resultados_ecg.csv", "text/csv")
st.caption(f"Modelo {meta['modelo']} v{meta['version']}. Entrenado con VitalDB Arrhythmia (un hospital de Corea). "
           "Bloqueo AV y ritmo multifocal se reconocen poco; un tramo sin alerta no descarta una arritmia.")
