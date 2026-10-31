"""Utilidades de ECG para el proyecto VitalDB Arrhythmia.

Las usan el notebook model_development.ipynb y la app de Streamlit, para que el
preprocesamiento de entrenamiento y el de produccion sean exactamente el mismo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal, stats

# Compatibilidad: np.trapezoid existe en NumPy 2, np.trapz en versiones anteriores
_trapz = getattr(np, "trapezoid", None) or np.trapz
FS = 500  # Hz, frecuencia de muestreo de SNUADC/ECG_II en VitalDB
CANAL_ECG = "SNUADC/ECG_II"

# Nombres unificados de las clases, usados en todo el proyecto
CLASES_ORDEN = [
    "Normal",
    "AFib",
    "SVTA",
    "TV",
    "Ectopia ventricular",
    "Ectopia auricular",
    "Disfuncion nodo sinusal",
    "WAP/MAT",
    "Bloqueo AV",
    "Ruido",
    "Inclasificable",
]


def normalizar_ritmo(valor) -> str:
    """Convertir la etiqueta original de rhythm_label a un nombre unificado.

    El orden de las reglas importa: 'supraventricular' contiene 'ventricular'.
    """
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "Sin etiqueta"
    s = str(valor).strip().lower()
    if "noise" in s or "ruido" in s:
        return "Ruido"
    if "unclass" in s:
        return "Inclasificable"
    if "afib" in s or "fibrillation" in s or "flutter" in s:
        return "AFib"
    # Supraventricular va antes que ventricular para no confundirlas
    if "supraventricular" in s or s == "svta":
        return "SVTA"
    if "patterned ventricular" in s or s == "pve":
        return "Ectopia ventricular"
    if "ventricular tach" in s or s == "vt":
        return "TV"
    if "patterned atrial" in s or s == "pae":
        return "Ectopia auricular"
    if "sinus node" in s or s == "snd":
        return "Disfuncion nodo sinusal"
    if "wandering" in s or "multifocal" in s or s in ("wap/mat", "wap", "mat"):
        return "WAP/MAT"
    if "atrioventricular" in s or "av block" in s or s == "avb":
        return "Bloqueo AV"
    if "normal" in s or s == "n":
        return "Normal"
    return "Otro"  # etiqueta no prevista: el notebook la lista para revisarla


# ----------------------------------------------------------------------------
# Senal
# ----------------------------------------------------------------------------
def _sos(low, high, fs, order=3):
    """Disenar un filtro pasa banda Butterworth en formato SOS (numericamente estable)."""
    return signal.butter(order, [low, high], btype="band", fs=fs, output="sos")


def filtrar_ecg(x: np.ndarray, fs: int = FS, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Filtrar con pasa banda de fase cero. Rellena NaN por interpolacion lineal."""
    x = np.asarray(x, dtype=float)
    # Rellenar huecos (electrodo desconectado) antes de filtrar, o el NaN contaminaria todo
    if np.isnan(x).any():
        idx = np.arange(len(x))
        ok = ~np.isnan(x)
        if ok.sum() < 2:
            return np.zeros_like(x)
        x = np.interp(idx, idx[ok], x[ok])
    # sosfiltfilt filtra hacia adelante y atras: no desplaza los picos en el tiempo
    return signal.sosfiltfilt(_sos(low, high, fs), x)


def detectar_picos_r(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Detectar picos R con un Pan-Tompkins simplificado.

    Devuelve los tiempos de los picos en segundos (relativos al inicio de x).
    """
    xf = filtrar_ecg(x, fs)
    if len(xf) < fs:
        return np.array([])
    # Resaltar el QRS: banda 5-15 Hz, derivada al cuadrado y media movil de ~120 ms
    y = signal.sosfiltfilt(_sos(5, 15, fs), xf)
    e = np.diff(y, prepend=y[0]) ** 2
    w = max(int(0.12 * fs), 1)
    e = np.convolve(e, np.ones(w) / w, mode="same")
    # Umbral adaptativo por bloques de 10 s, para tolerar cambios de amplitud
    bloque = 10 * fs
    picos = []
    for ini in range(0, len(e), bloque):
        seg = e[ini : ini + bloque]
        if len(seg) < fs:
            continue
        thr = 0.25 * np.percentile(seg, 99.5)
        if thr <= 0:
            continue
        # distance: periodo refractario de 250 ms (evita doble deteccion del mismo latido)
        p, _ = signal.find_peaks(seg, height=thr, distance=int(0.25 * fs))
        picos.extend((p + ini).tolist())
    picos = np.array(sorted(picos), dtype=int)
    if len(picos) == 0:
        return np.array([])
    # Refinar cada pico al maximo absoluto de la senal filtrada en +-50 ms
    r = int(0.05 * fs)
    ref = []
    for p in picos:
        a, b = max(p - r, 0), min(p + r + 1, len(xf))
        ref.append(a + int(np.argmax(np.abs(xf[a:b]))))
    ref = np.unique(ref)
    return ref / fs


def emparejar_picos(ref: np.ndarray, det: np.ndarray, tol: float = 0.05):
    """Comparar picos detectados contra anotados. Devuelve (TP, FP, FN).

    Un pico detectado cuenta como acierto si esta a menos de `tol` segundos de uno anotado.
    """
    ref, det = np.asarray(ref), np.asarray(det)
    if len(ref) == 0 or len(det) == 0:
        return 0, len(det), len(ref)
    usados = np.zeros(len(det), dtype=bool)  # cada deteccion se usa una sola vez
    tp = 0
    for r in ref:
        j = np.searchsorted(det, r)
        # Candidatos: la deteccion inmediatamente anterior y la posterior
        cand = [k for k in (j - 1, j) if 0 <= k < len(det) and not usados[k]]
        if not cand:
            continue
        k = min(cand, key=lambda k: abs(det[k] - r))
        if abs(det[k] - r) <= tol:
            usados[k] = True
            tp += 1
    return tp, len(det) - tp, len(ref) - tp


# ----------------------------------------------------------------------------
# Caracteristicas
# ----------------------------------------------------------------------------
def entropia_muestral(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Calcular la entropia muestral: mide cuan impredecible es la serie (alta en AFib)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2 or np.std(x) == 0:
        return np.nan
    r = r_frac * np.std(x)  # tolerancia de coincidencia

    def _cuenta(mm):
        # Contar pares de plantillas de largo mm que difieren menos que r (sin contarse a si mismas)
        plantillas = np.array([x[i : i + mm] for i in range(n - m)])
        d = np.max(np.abs(plantillas[:, None, :] - plantillas[None, :, :]), axis=2)
        return (np.sum(d <= r) - len(plantillas)) / 2

    b, a = _cuenta(m), _cuenta(m + 1)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


# Orden fijo de las columnas de caracteristicas (el modelo depende de este orden)
NOMBRES_RR = [
    "n_latidos", "fc_media", "fc_pico", "rr_media", "rr_mediana", "sdnn", "rmssd", "pnn50", "pnn20",
    "cv_rr", "rr_min", "rr_max", "rr_iqr", "rr_mad", "rr_skew", "rr_kurt", "sd1", "sd2",
    "sd1_sd2", "frac_rr_corto", "frac_rr_largo", "tpr",
]
# v0 incluia "sampen" (entropia muestral): se elimino porque faltaba en el 62 % de las ventanas de 10 s
NOMBRES_SENAL = [
    "sig_std", "sig_p2p", "sig_skew", "sig_kurt", "sig_sqi_qrs", "sig_ruido_alto",
    "sig_deriva", "sig_frac_plana",
]


def features_rr(picos: np.ndarray) -> dict:
    """Calcular caracteristicas de variabilidad del ritmo a partir de los tiempos de los picos R."""
    f = dict.fromkeys(NOMBRES_RR, np.nan)  # todo NaN por defecto: ventanas sin latidos suficientes
    picos = np.asarray(picos, dtype=float)
    f["n_latidos"] = float(len(picos))
    if len(picos) < 4:
        return f
    rr = np.diff(picos)
    rr = rr[(rr > 0.2) & (rr < 3.0)]  # descartar intervalos fisiologicamente imposibles
    if len(rr) < 3:
        return f
    d = np.diff(rr)  # diferencias entre RR sucesivos
    med = np.median(rr)
    f.update(
        fc_media=60.0 / rr.mean(), rr_media=rr.mean(), rr_mediana=med, sdnn=rr.std(ddof=1),
        rmssd=np.sqrt(np.mean(d**2)), pnn50=np.mean(np.abs(d) > 0.05), pnn20=np.mean(np.abs(d) > 0.02),
        cv_rr=rr.std(ddof=1) / rr.mean(), rr_min=rr.min(), rr_max=rr.max(),
        rr_iqr=np.subtract(*np.percentile(rr, [75, 25])), rr_mad=np.median(np.abs(rr - med)),
        rr_skew=float(stats.skew(rr)), rr_kurt=float(stats.kurtosis(rr)),
        # SD1 y SD2: ejes de la elipse del grafico de Poincare (variabilidad corta y larga)
        sd1=np.std(d, ddof=1) / np.sqrt(2),
        sd2=np.sqrt(max(2 * np.var(rr, ddof=1) - 0.5 * np.var(d, ddof=1), 0)),
        # Fraccion de intervalos muy cortos o largos respecto a la mediana (ectopias, pausas)
        frac_rr_corto=np.mean(rr < 0.8 * med), frac_rr_largo=np.mean(rr > 1.2 * med),
    )
    # Frecuencia maxima sostenida durante 3 latidos seguidos: capta salvas breves de TV o SVTA
    if len(rr) >= 3:
        f["fc_pico"] = 60.0 / np.convolve(rr, np.ones(3) / 3, mode="valid").min()
    f["sd1_sd2"] = f["sd1"] / f["sd2"] if f["sd2"] > 0 else np.nan
    # Turning point ratio: fraccion de maximos y minimos locales (~0.67 en ruido blanco)
    if len(rr) >= 3:
        a, b, c = rr[:-2], rr[1:-1], rr[2:]
        f["tpr"] = float(np.mean(((b > a) & (b > c)) | ((b < a) & (b < c))))
    return f


def features_senal(x_crudo: np.ndarray, fs: int = FS) -> dict:
    """Calcular caracteristicas de calidad y forma de la senal. Ayudan a detectar Ruido."""
    f = dict.fromkeys(NOMBRES_SENAL, np.nan)
    x = np.asarray(x_crudo, dtype=float)
    if len(x) < 4 * fs or np.isnan(x).mean() > 0.5:
        return f
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    xf = filtrar_ecg(x, fs)
    # Espectro de potencia de la senal cruda (Welch) para medir cuanta energia hay en cada banda
    fr, pxx = signal.welch(x - x.mean(), fs=fs, nperseg=min(len(x), 4 * fs))

    def banda(a, b):
        m = (fr >= a) & (fr < b)
        return float(_trapz(pxx[m], fr[m])) if m.any() else 0.0

    total = banda(0.5, 100) + 1e-12
    f.update(
        sig_std=float(xf.std()), sig_p2p=float(np.percentile(xf, 99) - np.percentile(xf, 1)),
        sig_skew=float(stats.skew(xf)), sig_kurt=float(stats.kurtosis(xf)),
        sig_sqi_qrs=banda(5, 15) / total,          # fraccion de energia en la banda del QRS
        sig_ruido_alto=banda(40, 100) / total,     # ruido de alta frecuencia (electrobisturi, EMG)
        sig_deriva=banda(0, 0.5) / (total + banda(0, 0.5)),  # deriva de la linea base
        sig_frac_plana=float(np.mean(np.abs(np.diff(x)) < 1e-6)),  # tramos sin senal (saturacion)
    )
    return f


NOMBRES_MORF = [
    "qrs_ancho_med", "qrs_ancho_max", "corr_plantilla_med", "corr_plantilla_min", "frac_latidos_atipicos",
    "r_amp_cv", "p_amp_rel", "p_presente_frac", "pr_intervalo", "pr_std", "p_corr_med", "p_sin_qrs", "t_amp_rel",
]
NOMBRES_EXTRA = ["sin_rr_valido"]

# Ventana alrededor de cada pico R usada para describir la forma del latido (segundos)
_ANTES, _DESPUES = 0.35, 0.45


def _quitar_tendencia(seg: np.ndarray) -> np.ndarray:
    """Restar la recta que une los extremos del tramo (linea base local)."""
    if len(seg) < 2:
        return seg
    return seg - np.linspace(seg[0], seg[-1], len(seg))


def _ancho_qrs(latido: np.ndarray, centro: int, fs: int) -> float:
    """Estimar el ancho del QRS: tramo alrededor del pico donde la pendiente supera el 20 % de su maximo."""
    a, b = centro - int(0.12 * fs), centro + int(0.12 * fs)
    d = np.abs(np.diff(latido[a:b]))
    if d.max() <= 0:
        return np.nan
    sobre = np.where(d > 0.2 * d.max())[0]
    return (sobre[-1] - sobre[0]) / fs


def features_morfologia(x_crudo: np.ndarray, picos: np.ndarray, fs: int = FS) -> dict:
    """Describir la forma de las ondas P, QRS y T a partir de la senal y de los picos R detectados.

    Son las caracteristicas que usa un medico para separar ritmos con intervalos RR parecidos:
    ancho del QRS (ventricular o no), presencia y forma de la onda P (sinusal, multifocal o ausente),
    intervalo PR y ondas P sin QRS detras (bloqueo AV).
    """
    f = dict.fromkeys(NOMBRES_MORF, np.nan)
    if x_crudo is None or len(picos) < 3:
        return f
    x = np.nan_to_num(np.asarray(x_crudo, dtype=float), nan=np.nanmedian(x_crudo))
    xf = filtrar_ecg(x, fs)                 # 0.5-40 Hz: forma general y QRS
    xp = filtrar_ecg(x, fs, 0.5, 15.0)      # 0.5-15 Hz: ondas P y T, mas suaves
    antes, despues = int(_ANTES * fs), int(_DESPUES * fs)
    idx = np.round(np.asarray(picos) * fs).astype(int)
    idx = idx[(idx - antes >= 0) & (idx + despues < len(xf))]   # solo latidos completos
    if len(idx) < 3:
        return f

    # Matriz de latidos alineados en el pico R (un latido por fila), centrados en su mediana
    lat = np.stack([xf[i - antes:i + despues] for i in idx])
    lat = lat - np.median(lat, axis=1, keepdims=True)
    latp = np.stack([xp[i - antes:i + despues] for i in idx])
    amp = np.abs(lat[:, antes - int(0.04 * fs):antes + int(0.04 * fs)]).max(axis=1)   # amplitud del QRS
    amp_ref = np.median(amp) + 1e-9

    # QRS: ancho y parecido con la plantilla (latido mediano de la ventana)
    anchos = np.array([_ancho_qrs(b, antes, fs) for b in lat])
    qrs = slice(antes - int(0.10 * fs), antes + int(0.12 * fs))
    plantilla = np.median(lat, axis=0)
    corr = np.array([np.corrcoef(b[qrs], plantilla[qrs])[0, 1] if b[qrs].std() > 0 else np.nan for b in lat])

    # Onda P: buscar en 300 a 80 ms antes del pico R, sobre la senal suave y sin tendencia
    p_a, p_b = antes - int(0.30 * fs), antes - int(0.08 * fs)
    p_seg = np.stack([_quitar_tendencia(b[p_a:p_b]) for b in latp])
    p_pos = np.abs(p_seg).argmax(axis=1)
    p_amp = np.abs(p_seg)[np.arange(len(p_seg)), p_pos] / amp_ref
    p_pres = p_amp > 0.06                       # P visible: al menos 6 % de la amplitud del QRS
    pr = (p_b - p_a - p_pos + int(0.08 * fs)) / fs   # distancia del pico de la P al pico R
    p_plant = np.median(p_seg, axis=0)
    p_corr = np.array([np.corrcoef(s, p_plant)[0, 1] if s.std() > 0 and p_plant.std() > 0 else np.nan for s in p_seg])

    # Onda T: 150 a 450 ms despues del pico R
    t_a, t_b = antes + int(0.15 * fs), antes + int(0.45 * fs)
    t_amp = np.array([np.abs(_quitar_tendencia(b[t_a:t_b])).max() for b in latp]) / amp_ref

    # Ondas P sin QRS detras: en pausas largas, buscar una onda tipo P lejos de la T anterior y del QRS siguiente
    rr = np.diff(idx) / fs
    largas = np.where(rr > 1.5 * np.median(rr))[0]
    bloqueadas = 0
    p_ref = np.median(p_amp[p_pres]) if p_pres.any() else np.nan
    for k in largas:
        a, b = idx[k] + int(0.45 * fs), idx[k + 1] - int(0.30 * fs)
        if b - a < int(0.10 * fs) or np.isnan(p_ref):
            continue
        tramo = np.abs(_quitar_tendencia(xp[a:b])) / amp_ref
        pk, _ = signal.find_peaks(tramo, height=0.6 * p_ref, distance=int(0.2 * fs))
        bloqueadas += len(pk)

    f.update(
        qrs_ancho_med=float(np.nanmedian(anchos)), qrs_ancho_max=float(np.nanmax(anchos)),
        corr_plantilla_med=float(np.nanmedian(corr)), corr_plantilla_min=float(np.nanmin(corr)),
        frac_latidos_atipicos=float(np.nanmean(corr < 0.8)), r_amp_cv=float(amp.std() / amp_ref),
        p_amp_rel=float(np.median(p_amp)), p_presente_frac=float(p_pres.mean()),
        pr_intervalo=float(np.median(pr[p_pres])) if p_pres.any() else np.nan,
        pr_std=float(np.std(pr[p_pres])) if p_pres.sum() > 1 else np.nan,
        p_corr_med=float(np.nanmedian(p_corr)), p_sin_qrs=float(bloqueadas / len(rr)),
        t_amp_rel=float(np.median(t_amp)),
    )
    return f


def features_ventana(picos_rel: np.ndarray, ecg_ventana: np.ndarray | None = None, fs: int = FS) -> dict:
    """Unir las caracteristicas de ritmo (RR), senal y morfologia de una ventana en un solo diccionario."""
    f = features_rr(picos_rel)
    f.update(features_senal(ecg_ventana, fs) if ecg_ventana is not None else dict.fromkeys(NOMBRES_SENAL, np.nan))
    f.update(features_morfologia(ecg_ventana, picos_rel, fs) if ecg_ventana is not None
             else dict.fromkeys(NOMBRES_MORF, np.nan))
    # Indicador explicito: 1 si no hubo latidos suficientes para calcular el ritmo (nulos informativos)
    f["sin_rr_valido"] = float(np.isnan(f["fc_media"]))
    return f


# ----------------------------------------------------------------------------
# E/S de ECG (cache local)
# ----------------------------------------------------------------------------
def ruta_cache(case_id: int, ecg_dir) -> Path:
    """Ruta del archivo de cache de un caso."""
    return Path(ecg_dir) / f"ecg_{int(case_id)}.npz"


def cargar_ecg_cache(case_id: int, ecg_dir):
    """Devolver (ecg, offset_s) o (None, None) si el caso no esta en cache.

    ecg[i] corresponde al tiempo offset_s + i / FS (mismo origen que time_second).
    """
    p = ruta_cache(case_id, ecg_dir)
    if not p.exists():
        return None, None
    z = np.load(p)
    return z["ecg"].astype(np.float32), float(z["offset_s"])


def descargar_ecg(case_id: int, ecg_dir, t_ini: float, t_fin: float, fs: int = FS, margen: float = 5.0):
    """Bajar el ECG de VitalDB, recortar [t_ini-margen, t_fin+margen] y guardarlo comprimido."""
    import vitaldb  # import tardio: solo hace falta para descargar

    vals = vitaldb.load_case(int(case_id), [CANAL_ECG], 1.0 / fs)
    # Segun la version y el numero de pistas, load_case devuelve un dict, un vector o una matriz
    arr = vals[CANAL_ECG] if isinstance(vals, dict) else np.asarray(vals)
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr.ravel() if arr.ndim == 1 else arr[:, 0]
    # Guardar solo el tramo anotado (~20 min) en vez de toda la cirugia: ahorra disco
    a = max(int((t_ini - margen) * fs), 0)
    b = min(int((t_fin + margen) * fs), len(arr))
    Path(ecg_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta_cache(case_id, ecg_dir), ecg=arr[a:b], offset_s=a / fs)
    return b - a
