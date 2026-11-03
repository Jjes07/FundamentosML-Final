# Detección de arritmias intraoperatorias con ML tradicional

Proyecto final de **Fundamentos de Aprendizaje Automático**, EAFIT.

Clasificación del ritmo cardíaco de pacientes en cirugía a partir de ventanas de 10 s de ECG (derivación II), usando aprendizaje supervisado tradicional. El modelo distingue ritmo normal, arritmias relevantes (fibrilación auricular, taquicardias, ectopias, bloqueo) y tramos con ruido. Es una herramienta de **apoyo** para el personal de quirófano: no diagnostica.

## Resultado

| | v2 (base) | **v3 (final, XGBoost)** |
|---|---|---|
| F1 macro, validación cruzada externa por paciente | 0.534 ± 0.019 | **0.545 ± 0.014** |
| F1 macro, test fijo (93 pacientes) | 0.526 | **0.541** (IC95 % 0.474 a 0.573) |
| Línea base (siempre Normal) | 0.077 | 0.077 |

Por clase: AFib 0.95, Normal 0.92, Ectopia ventricular 0.75, Ruido 0.67, Ectopia auricular 0.65, SVTA 0.47, TV 0.40, Disfunción del nodo 0.38, Bloqueo AV 0.12 y WAP/MAT 0.05. Agrupado en normal y anormal: F1 0.878, sensibilidad 0.919, AUROC 0.968. Las limitaciones están en la sección 17 del notebook.

## Probar la demo

```
pip install -r requirements.txt
streamlit run app/app.py
```

- **App:** `app/app.py`. Trae tres ejemplos y acepta archivos propios. No requiere descargar datos ni ejecutar el notebook.
- **Monitor animado:** abrir `demo/monitor.html` en el navegador. No requiere instalar nada.
- **Reentrenamiento:** `python reentrenar.py --simular-nuevos 60 --sin-guardar` hace un ensayo completo (requiere los datos).

## Contexto

Durante una cirugía la anestesia, los fármacos y el estímulo quirúrgico provocan arritmias con frecuencia. Algunas requieren intervención inmediata, pero el anestesiólogo atiende muchas tareas a la vez. Las bases públicas de ECG existentes son ambulatorias o de cuidados intensivos y no representan el entorno quirúrgico. Este proyecto usa una base anotada específicamente en cirugía.

## Datos

**VitalDB Arrhythmia Database** (PhysioNet, 2026, CC BY 4.0). DOI: [10.13026/axd6-wm13](https://doi.org/10.13026/axd6-wm13).

- 482 casos (473 pacientes) quirúrgicos del Hospital Universitario Nacional de Seúl, unos 20 minutos anotados por caso.
- Más de 660,000 latidos con tipo de latido y 10 categorías de ritmo, validados por anestesiólogos (kappa de Cohen 0.93).
- El paquete de PhysioNet trae **solo anotaciones y metadatos** (21 MB). La señal ECG (`SNUADC/ECG_II`, 500 Hz) se descarga del [VitalDB](https://physionet.org/content/vitaldb/1.0.0/) original con el `case_id`.
- Limitación importante: los segmentos fueron preseleccionados por un modelo de IA como candidatos a arritmia (sesgo de selección), y todos vienen de un solo hospital.

## Estructura

```
FundamentosML-Final/
  model_development.ipynb    notebook principal: EDA, modelado, experimentos, evaluación, despliegue, drift
  PENDIENTES.md              lista de cambios acordados y su estado
  descargar_datos.py         descarga anotaciones y ECG a data/
  reentrenar.py              pipeline de reentrenamiento con regla de promoción
  generar_demo.py            genera los ejemplos y el monitor de la demo
  requirements.txt
  src/ecg_utils.py           filtro, detector de picos R, características (compartido con la app)
  src/modelos.py             XGBoost con pesos por clase (lo necesita el modelo guardado)
  app/app.py                 interfaz Streamlit (la genera el notebook)
  app/ejemplos/              tres ECG de 1 minuto de pacientes del test
  demo/monitor.html          monitor animado (lo genera generar_demo.py)
  models/                    modelos entrenados, referencias de monitoreo y registro de experimentos
  figures/                   figuras del EDA, los experimentos y la evaluación
  data/                      datos descargados (ver abajo)
```

## Cómo reproducir

1. Descargar los datos (sección siguiente).
2. Abrir `model_development.ipynb` desde la raíz del proyecto y ejecutarlo en orden. También funciona en Google Colab (monta Drive y ajusta las rutas).
3. Para la interfaz: `streamlit run app/app.py`. El modelo ya viene en `models/`; el notebook lo vuelve a generar.

Ejecutar el notebook completo toma unos 30 minutos si existe `data/cache_experimentos/`. Esa carpeta no se versiona, así que la primera vez los experimentos de la sección 12.3 suman más de una hora.

Probado con Python 3.13. `models/*.joblib` requiere las versiones de scikit-learn y XGBoost fijadas en `requirements.txt`.

Parámetros como `MAX_CASOS`, `VENTANA_S` o `FUENTE_PICOS` se cambian en la celda de configuración del notebook.

---

# Descarga de los datos (VitalDB Arrhythmia)

Todo queda dentro de `data/`. Resultado final esperado:

```
data/
  metadata.csv                  (218 KB)
  Annotation_Files/             (482 archivos Annotation_file_<case_id>.csv)
  LICENSE.txt, README.md, SHA256SUMS.txt
  ecg_cache/                    (un ecg_<case_id>.npz por caso, ~130 MB en total)
```

El script solo guarda el tramo anotado (~20 min) de cada caso, no la cirugía completa.

## Opción A: un solo comando (recomendada)

1. Abrir una terminal en la raíz del proyecto.
2. Instalar las dependencias. Un entorno virtual es **opcional**; si ya tienes Python con pandas, scikit-learn, scipy, matplotlib y seaborn, basta con instalar lo que falte:

   ```
   pip install -r requirements.txt
   ```

3. Probar primero con pocos casos:

   ```
   python descargar_datos.py --max-casos 5
   ```

   Descarga las anotaciones, verifica los checksums y baja el ECG de 5 casos. Deben aparecer archivos en `data/ecg_cache/`.
4. Si todo salió bien, descargar todo (entre 30 minutos y unas horas según la conexión):

   ```
   python descargar_datos.py --workers 4
   ```

   Es **reanudable**: si se corta, volver a ejecutarlo continúa donde quedó. Los casos que fallen quedan en `data/ecg_cache/fallidos.txt`; reejecutar el comando los reintenta.

Opciones: `--solo-anotaciones` (solo los 21 MB), `--max-casos N`, `--workers N`.

## Opción B: anotaciones a mano

1. En https://physionet.org/content/vitaldb-arrhythmia/1.0.0/ pulsar **Download the ZIP file** (3.4 MB).
2. Descomprimir y dejar **dentro de `data/`**: `metadata.csv`, la carpeta `Annotation_Files/` y, opcionalmente, `SHA256SUMS.txt`.
3. Correr `python descargar_datos.py`, que omite las anotaciones si ya existen y baja solo los ECG.

Otras vías oficiales: `wget -r -N -c -np https://physionet.org/files/vitaldb-arrhythmia/1.0.0/` o `aws s3 sync --no-sign-request s3://physionet-open/vitaldb-arrhythmia/1.0.0/ data/`.

## Opción C: Google Colab

1. Subir la carpeta del proyecto a Google Drive en `MyDrive/FundamentosML-Final`.
2. En Colab, ejecutar el notebook: la celda de configuración monta Drive y ajusta las rutas.
3. Para descargar los datos desde Colab, en una celda:

   ```
   !pip install -q vitaldb tqdm
   %cd /content/drive/MyDrive/FundamentosML-Final
   !python descargar_datos.py --workers 4
   ```

   Los datos quedan en Drive y no hay que repetirlo.

## Verificaciones rápidas

- `python -c "import pandas as pd; print(pd.read_csv('data/metadata.csv').shape)"` debe dar unas 482 filas.
- `data/Annotation_Files` debe tener 482 archivos.
- `data/ecg_cache` debe tener hasta 482 archivos `.npz`; algunos casos pueden faltar si VitalDB no tiene ese canal.
- Prueba mínima de la señal: `python -c "import vitaldb; print(len(vitaldb.load_case(337, ['SNUADC/ECG_II'], 1/500)))"`

## Problemas comunes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `URLError` o timeout al bajar el ECG | Red inestable o servidor ocupado | Bajar `--workers` a 2 y reintentar; el script reintenta 3 veces cada caso |
| El ZIP no se descarga | La URL de PhysioNet cambió | Usar la Opción B |
| Un caso falla siempre | Ese caso no trae el canal `SNUADC/ECG_II` | Se omite; el notebook trabaja con los casos disponibles |
| `vitaldb` o `shap` no se instalan | Versión de Python posterior a 3.13, sin paquetes publicados aún | Usar Python 3.13, que es la probada, o Colab |
| Poco espacio | ~130 MB para los 482 casos | Usar `--max-casos 100` |

`data/ecg_cache/` está en `.gitignore`. Las anotaciones (21 MB) pueden versionarse, pero conviene citar la fuente en lugar de duplicarla.

## Referencias

- Eun, D., et al. (2026). *VitalDB Arrhythmia Database* (v1.0.0). PhysioNet. https://doi.org/10.13026/axd6-wm13
- Lee, H.-C., et al. (2022). VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. *Scientific Data*, 9(1), 279.
- Pollard, T., et al. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z
- Repositorio de los autores: https://github.com/vitaldb/arrdb
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
