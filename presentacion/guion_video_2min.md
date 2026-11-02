# Guion Técnico Cronometrado: Video Demostrativo (2 Minutos)
## Sistema de Apoyo a la Decisión Clínica para Monitoreo de ECG Intraoperatorio
### Demostración del Modelo Desplegado y su Interfaz de Usuario (`app/app.py`)

---

> **Especificaciones del Video:**
> - **Duración total:** 120 segundos (02:00 exactos).
> - **Requisito curricular:** Sección 4 de `docs/Requeriments_FinalP.pdf` ("video de máximo 2 minutos mostrando el funcionamiento del modelo desplegado y su interfaz").
> - **Pacing de voz en off:** ~135 palabras por minuto. Conteo de locución diseñado para **~270 palabras** de narración continua, pausada y profesional.
> - **Versión del modelo en producción:** `[PENDIENTE: confirmar si el modelo cargado en el video sigue siendo HistGradientBoosting v2 (2026.11.01) o si cambia tras el cierre del bloque F]`.

---

## Estructura Temporal del Video (120 Segundos)

| Bloque | Intervalo de Tiempo | Duración | Eje Temático Demostrado | Enfoque de Pantalla |
| :---: | :---: | :---: | :--- | :--- |
| **B1** | **00:00 – 00:20** | 20 s | Introducción al Reto Quirúrgico y Portada | Banner clínico de la aplicación y contexto hospitalario |
| **B2** | **00:20 – 00:45** | 25 s | Carga de Trazas ECG y Parámetros de Inferencia | Sidebar interactivo: Frecuencia de muestreo y archivo CSV |
| **B3** | **00:45 – 01:10** | 25 s | Procesamiento en Vivo y Trazas con Código Cromático | Gráfica continua interactiva con bandas Verde / Gris / Rojo |
| **B4** | **01:10 – 01:40** | 30 s | Auditoría Tabular, Umbral de Confianza y Casos Críticos | Slider de confianza, métricas de frecuencia y bandera `Revisar` |
| **B5** | **01:40 – 02:00** | 20 s | Arquitectura MLOps, Detección de Deriva y Cierre | Monitoreo PSI continuo y despedida institucional |

---

## Guion Detallado: Acción Visual vs. Locución

### Bloque 1: Introducción y Contexto Quirúrgico (00:00 – 00:20 | 20 segundos)

#### [PANTALLA / ACCIÓN VISUAL]
- **00:00 – 00:05:** Primer plano de la interfaz web abierta en el navegador: Título destacado `🏥 Monitoreo Inteligente de ECG Intraoperatorio` y subtítulo con metadatos del modelo (`HistGradientBoosting v2`, 500 Hz, Derivación II).
- **00:05 – 00:12:** Zoom hacia el banner de seguridad clínica superior en tono ámbar/azul:  
  *«⚠️ Herramienta de apoyo a la decisión para el anestesiólogo. No sustituye el juicio clínico ni la monitorización estándar.»*
- **00:12 – 00:20:** Paneado general hacia la barra lateral derecha y el lienzo principal vacío esperando la carga del paciente.

#### [VOZ EN OFF / NARRACIÓN] (44 palabras)
> *"En quirófano, la fatiga por falsas alarmas y el ruido de electrocauterio dificultan detectar arritmias críticas. Presentamos un sistema de apoyo a la decisión médica para monitoreo de electrocardiograma intraoperatorio, diseñado para clasificar el ritmo cardíaco en ventanas de diez segundos sin depender de anotaciones humanas."*

---

### Bloque 2: Carga de Señal y Configuración de Inferencia (00:20 – 00:45 | 25 segundos)

#### [PANTALLA / ACCIÓN VISUAL]
- **00:20 – 00:28:** El cursor interactúa con el panel lateral derecho (`sidebar`):
  - Muestra el selector de **Frecuencia de muestreo** configurado en `500 Hz` (con opción de remuestreo automático si la fuente difiere).
  - Puntero resalta el control deslizante **Umbral de confianza mínima** posicionado en `0.60`.
- **00:28 – 00:37:** Clic en el componente `st.file_uploader`:  
  `[PENDIENTE: selección de los 3 archivos CSV de 1 minuto del conjunto de test — ej. normal con ectopias, AFib, salva de TV con ruido — del bloque H2]`.  
  Se arrastra y suelta un archivo de prueba representativo (ejemplo: `caso_test_afib_tv_1min.csv`).
- **00:37 – 00:45:** Animación breve de carga (`Extrayendo 45 características...`). Se activa el pipeline modular de `src/ecg_utils.py` ejecutando el filtrado Butterworth SOS (0.5–40 Hz) y el detector QRS adaptativo v2 con piso robusto.

#### [VOZ EN OFF / NARRACIÓN] (56 palabras)
> *"Desde el panel lateral ajustamos la frecuencia de muestreo y definimos un umbral de confianza operativa del sesenta por ciento. Cargamos una señal electrocardiográfica cruda de un minuto tomada de nuestro conjunto de prueba independiente de pacientes nuevos. De inmediato, el sistema filtra artefactos y extrae cuarenta y cinco características de ritmo, calidad de señal y morfología."*

---

### Bloque 3: Inferencia y Visualización con Código de Alertas (00:45 – 01:10 | 25 segundos)

#### [PANTALLA / ACCIÓN VISUAL]
- **00:45 – 00:55:** La pantalla despliega la gráfica interactiva superior con la señal ECG filtrada y las **bandas de fondo coloreadas por tramo de 10 segundos**:
  - Tramos iniciales con fondo **Verde suave**: Ritmo Normal y ectopias supraventriculares aisladas.
  - Tramo central con fondo **Gris**: Artefacto/Ruido (desconexión o interferencia del bisturí).
  - Tramo posterior con fondo **Rojo**: Alerta crítica de arritmia de alto riesgo (Fibrilación Auricular con respuesta ventricular rápida o salva de Taquicardia Ventricular).
- **00:55 – 01:10:** Demostración de interactividad: el cursor hace zoom dinámico sobre un complejo QRS dentro del tramo rojo, mostrando la preservación morfológica de fase cero sin retardo temporal.
- `[PENDIENTE: demostración de la vista de monitor quirúrgico en tiempo real (doble pantalla: traza cruda a la izquierda y traza con predicción/alertas por ventana deslizante a la derecha) del bloque H4]`.

#### [VOZ EN OFF / NARRACIÓN] (57 palabras)
> *"La interfaz grafica la traza continua estratificada con un código cromático intuitivo: verde para ritmos normales o ectopias estables; gris para segmentos con ruido excesivo o electrobisturí que no deben interpretarse; y rojo para arritmias de riesgo vital. Esto permite al especialista priorizar de un solo vistazo el estado hemodinámico del paciente en el campo operatorio."*

---

### Bloque 4: Triage Tabular, Confianza y Casos Críticos (01:10 – 01:40 | 30 segundos)

#### [PANTALLA / ACCIÓN VISUAL]
- **01:10 – 01:22:** Scroll hacia la tabla inferior `Resumen por Tramo (Ventanas de 10 s)`:
  - Columnas visibles: `Inicio (s)`, `Ritmo Predicho`, `Confianza (%)`, `FC Media (lpm)`, `Revisar`.
  - La fila del segundo 20.0 destaca en rojo con `Ritmo: Taquicardia Ventricular`, `Confianza: 78.4%`, `FC: 148 lpm` y bandera `Revisar: SI`.
  - La fila de ruido muestra `Ritmo: Ruido / Artefacto`, `Confianza: 92.1%`, bandera `Revisar: SI`.
- **01:22 – 01:32:** El usuario mueve el control deslizante de confianza desde `0.60` hasta `0.85`:
  - Instantáneamente, un tramo ambiguo de ectopia auricular con 72% de confianza pasa su bandera `Revisar` a `SI` y se ilumina un mensaje de advertencia preventivo en la interfaz.
- **01:32 – 01:40:** Vista del resumen superior de métricas agregadas: 6 ventanas procesadas, 1 alerta crítica activa, 0 asistolias.

#### [VOZ EN OFF / NARRACIÓN] (68 palabras)
> *"En la tabla inferior, cada tramo desglosa el ritmo estimado, la frecuencia cardíaca y la probabilidad del modelo. Si se detecta una arritmia grave como taquicardia ventricular, o si la confianza del clasificador cae por debajo del umbral exigido por el médico, la bandera 'Revisar' se activa automáticamente. Esto evita decisiones a ciegas y focaliza la atención médica exactamente en las ventanas dudosas o inestables."*

---

### Bloque 5: Arquitectura MLOps, Deriva y Cierre (01:40 – 02:00 | 20 segundos)

#### [PANTALLA / ACCIÓN VISUAL]
- **01:40 – 01:48:** Transición hacia una lámina sintética que muestra la arquitectura de despliegue y observabilidad:
  - Diagrama de flujo: `VitalDB (473 Pacientes)` $\to$ `Validación GroupKFold` $\to$ `HistGradientBoosting (models/modelo_arritmias.joblib)` $\to$ `Monitoreo PSI continuo (models/referencia_features.csv)`.
  - Semáforo de deriva indicando estado verde ($\text{PSI} < 0.10$) en condiciones normales y alerta ante electrodos degradados ($\text{PSI} > 0.20$).
- **01:48 – 01:55:** Regreso a la pantalla principal de Streamlit con el equipo de trabajo acreditado en el pie de página.
- **01:55 – 02:00:** Cierre con logotipos institucionales de la Universidad EAFIT y código abierto en GitHub.

#### [VOZ EN OFF / NARRACIÓN] (45 palabras)
> *"El sistema opera respaldado por un protocolo estricto de validación por paciente, garantizando cero fuga de información. Además, incorpora monitoreo de deriva estadística con el índice PSI para alertar degradación de electrodos en quirófano. Machine learning robusto, transparente y centrado en la seguridad del paciente."*

---

## Balance del Conteo de Palabras y Tiempos de Locución

| Segmento | Duración | Palabras Locución | Palabras por Minuto (PPM) | Estado de Viabilidad |
| :--- | :---: | :---: | :---: | :---: |
| **Bloque 1:** Introducción Quirúrgica | 20 s | 44 palabras | 132 PPM | Ritmo óptimo (pausado, claro) |
| **Bloque 2:** Configuración y Carga | 25 s | 56 palabras | 134 PPM | Ritmo natural conversacional |
| **Bloque 3:** Trazas y Código de Colores | 25 s | 57 palabras | 136 PPM | Ritmo dinámico descriptivo |
| **Bloque 4:** Triage Tabular y Confianza | 30 s | 68 palabras | 136 PPM | Ritmo técnico explicativo |
| **Bloque 5:** MLOps, PSI y Cierre | 20 s | 45 palabras | 135 PPM | Ritmo conclusivo institucional |
| **TOTAL** | **120 s (02:00)** | **270 palabras** | **135.0 PPM** | **Cumplimiento exacto de especificación** |

---

## Lista de Marcadores Pendientes para Producción Audiovisual

1. **`[PENDIENTE: selección de los 3 archivos CSV de 1 minuto del conjunto de test — ej. normal con ectopias, AFib, salva de TV con ruido — del bloque H2]`**:
   - Requiere aislar 3 clips de 60 segundos (30,000 muestras cada uno a 500 Hz) a partir de los 93 pacientes exclusivos del Pliegue 0 de test para su uso durante la grabación en vivo.
2. **`[PENDIENTE: demostración de la vista de monitor quirúrgico en tiempo real (doble pantalla: traza cruda a la izquierda y traza con predicción/alertas por ventana deslizante a la derecha) del bloque H4]`**:
   - Una vez finalizada la vista de monitor de quirófano en tiempo real del bloque H4, grabar una toma adicional de 8 segundos para insertarla en el segundo 00:55 del video.
3. **`[PENDIENTE: confirmar si el modelo cargado en el video sigue siendo HistGradientBoosting v2 (2026.11.01) o si cambia tras el cierre del bloque F]`**:
   - En caso de que algún experimento del Bloque F supere la regla de adopción ($+0.005$ CV externa en 4/5 pliegues sin degradar recall de TV/Bloqueo AV/AFib), actualizar la referencia del modelo en el banner de la interfaz y en los metadatos visuales.
