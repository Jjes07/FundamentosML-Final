"""Descarga los datos de VitalDB Arrhythmia a la carpeta data/.

Uso (desde la raiz del proyecto):
    python descargar_datos.py                  # anotaciones + ECG de los 482 casos
    python descargar_datos.py --solo-anotaciones
    python descargar_datos.py --max-casos 20   # prueba rapida
    python descargar_datos.py --workers 6

Es reanudable: lo que ya esta en data/ecg_cache no se vuelve a bajar.
Requisitos: pip install vitaldb pandas numpy scipy tqdm
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Rutas del proyecto, relativas a la ubicacion de este script
RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))  # permite importar ecg_utils
from ecg_utils import FS, descargar_ecg, ruta_cache  # noqa: E402

DATA = RAIZ / "data"
ANN_DIR = DATA / "Annotation_Files"
ECG_DIR = DATA / "ecg_cache"
URL_ZIP = "https://physionet.org/static/published-projects/vitaldb-arrhythmia/vitaldb-arrhythmia-1.0.0.zip"


def bajar_anotaciones():
    """Descargar el ZIP de PhysioNet y dejar metadata.csv y Annotation_Files/ dentro de data/."""
    # Omitir si ya esta todo descargado
    if (DATA / "metadata.csv").exists() and ANN_DIR.exists() and any(ANN_DIR.glob("*.csv")):
        print("Anotaciones ya presentes, se omite la descarga.")
        return
    DATA.mkdir(exist_ok=True)
    zip_path = DATA / "vitaldb-arrhythmia-1.0.0.zip"
    print(f"Descargando {URL_ZIP} (3.4 MB)...")
    # timeout evita quedarse colgado si el servidor no responde
    with urllib.request.urlopen(URL_ZIP, timeout=60) as resp, open(zip_path, "wb") as salida:
        salida.write(resp.read())

    # El ZIP trae una carpeta raiz con un nombre larguisimo. Extraer ahi desde Windows supera el
    # limite de 260 caracteres, asi que se extrae archivo por archivo quitando esa carpeta raiz.
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            partes = Path(info.filename).parts
            relativa = Path(*partes[1:]) if len(partes) > 1 else Path(partes[0])
            destino = DATA / relativa
            destino.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as origen, open(destino, "wb") as salida:
                salida.write(origen.read())
    zip_path.unlink()  # el ZIP ya no hace falta
    print("Anotaciones listas en", DATA)


def verificar_sha256():
    """Comparar cada archivo con SHA256SUMS.txt para detectar descargas corruptas."""
    archivo = DATA / "SHA256SUMS.txt"
    if not archivo.exists():
        return
    malos = 0
    for linea in archivo.read_text().splitlines():
        if not linea.strip():
            continue
        esperado, nombre = linea.split(maxsplit=1)
        ruta = DATA / nombre.strip().lstrip("*")
        if ruta.exists() and hashlib.sha256(ruta.read_bytes()).hexdigest() != esperado:
            print("Checksum distinto:", ruta.name)
            malos += 1
    print("Verificacion SHA256:", "OK" if malos == 0 else f"{malos} archivos con problemas")


def rango_caso(case_id: int):
    """Leer el primer y ultimo instante anotado de un caso, para recortar solo ese tramo de ECG."""
    ruta = ANN_DIR / f"Annotation_file_{case_id}.csv"
    if not ruta.exists():
        return None
    t = pd.read_csv(ruta, usecols=["time_second"])["time_second"]
    return float(t.min()), float(t.max())


def bajar_caso(case_id: int, reintentos: int = 3):
    """Descargar y guardar en cache el ECG de un caso. Devuelve (case_id, estado, n_muestras)."""
    if ruta_cache(case_id, ECG_DIR).exists():
        return case_id, "ya estaba", 0
    r = rango_caso(case_id)
    if r is None:
        return case_id, "sin archivo de anotacion", 0
    ultimo = ""
    for _ in range(reintentos):  # reintentar ante fallos de red
        try:
            n = descargar_ecg(case_id, ECG_DIR, r[0], r[1], FS)
            return case_id, "ok", n
        except Exception as e:  # red, caso sin el canal de ECG, etc.
            ultimo = f"{type(e).__name__}: {e}"
    return case_id, f"fallo: {ultimo}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-anotaciones", action="store_true")
    ap.add_argument("--max-casos", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    print("1/3 Anotaciones...", flush=True)
    bajar_anotaciones()
    print("2/3 Verificando checksums...", flush=True)
    verificar_sha256()
    if a.solo_anotaciones:
        return

    # Lista de casos tomada del metadata; --max-casos permite probar con pocos
    ids = pd.read_csv(DATA / "metadata.csv")["case_id"].astype(int).tolist()
    if a.max_casos:
        ids = ids[: a.max_casos]
    ECG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"3/3 Bajando ECG de {len(ids)} casos con {a.workers} hilos...", flush=True)
    fallos = []
    try:
        from tqdm import tqdm  # barra de progreso opcional
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    # Descargas en paralelo: la espera es de red, por eso bastan hilos
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(bajar_caso, c) for c in ids]
        for f in tqdm(as_completed(futs), total=len(futs)):
            cid, estado, _ = f.result()
            print(f"  caso {cid}: {estado}", flush=True)
            if estado.startswith(("fallo", "sin")):
                fallos.append((cid, estado))

    # Guardar los casos fallidos para poder reintentarlos
    (ECG_DIR / "fallidos.txt").write_text("\n".join(f"{c}\t{e}" for c, e in fallos))
    tam = sum(p.stat().st_size for p in ECG_DIR.glob("*.npz")) / 1e9
    print(f"Listo. {len(ids) - len(fallos)} casos, {tam:.2f} GB en {ECG_DIR}. Fallidos: {len(fallos)}")


if __name__ == "__main__":
    main()
