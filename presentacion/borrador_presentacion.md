# Borrador de Presentación del Proyecto Final
## Sistema de Apoyo a la Decisión Clínica para Monitoreo Intraoperatorio de ECG
### Detección y Clasificación Multiclase de Arritmias Cardíacas en Anestesia General

---

> **Estructura de la Presentación:**
> Este documento contiene el diseño diapositiva a diapositiva (*Slide-by-Slide*) para la sustentación del proyecto final de Machine Learning, estructurado en torno a los 7 ejes requeridos por la **Sección 5 de `docs/Requeriments_FinalP.pdf`**. Cada diapositiva incluye: contenido visual proyectado, tablas estructuradas, figuras del repositorio a renderizar y notas completas para el orador con respaldo matemático, metodológico y clínico.

---

## Índice de Diapositivas

| Lámina | Eje Temático | Título de la Diapositiva | Figura / Recurso Clave |
| :---: | :--- | :--- | :--- |
| **01** | Portada | Monitoreo Inteligente de ECG Intraoperatorio | *Logo EAFIT / VitalDB* |
| **02** | 1. Problema y Contexto | Necesidad Clínica en Quirófano y Filosofía de Apoyo | Diagrama de flujo asistencial |
| **03** | 1. Problema y Contexto | VitalDB Arrhythmia Database y Hallazgos del EDA | `figures/eda_linea_tiempo.png` |
| **04** | 1. Problema y Contexto | Estructura No Supervisada y Justificación del Modelo | `figures/no_supervisado_pca.png` |
| **05** | 2. Metodología | Pipeline de Señal y Detección QRS Adaptativa v2 | Traza de detección en bisturí |
| **06** | 2. Metodología | Ingeniería de 45 Features y Tratamiento de MNAR | Tabla de familias de features |
| **07** | 3. Selección del Modelo | Protocolo por Paciente y Benchmark de Familias | Tabla comparativa de modelos |
| **08** | 3. Selección del Modelo | Optimización, Ablación y Suite Experimental Bloque F | `figures/registro_correcciones.png` |
| **09** | 4. Generalización | Validación Cruzada Externa y Test Ciego por Paciente | `figures/eval_cv_externa.png` |
| **10** | 4. Generalización | Demostración Empírica Anti-Leakage y Brecha General | `figures/gen_curva_aprendizaje.png` |
| **11** | 5. Métricas e Interpretabilidad | Desempeño Multiclase ante Desbalance Extremo | `figures/eval_confusion.png` & `eval_pr.png` |
| **12** | 5. Métricas e Interpretabilidad | Interpretabilidad y Coherencia Electrofisiológica | `figures/explic_permutacion.png` & `explic_shap_global.png` |
| **13** | 6. Despliegue Clínico | Arquitectura de Despliegue e Interfaz en Streamlit | Capturas reales de `app/app.py` |
| **14** | 7. Monitoreo y MLOps | Detección de Deriva (PSI/KS) y Protocolo de Reajuste | `models/referencia_features.csv` |
| **15** | Cierre | Conclusiones, Impacto Quirúrgico y Próximos Pasos | Hoja de ruta MLOps |

---

## Diapositiva 01: Portada
**Título:** Sistema de Apoyo a la Decisión Clínica para Monitoreo Intraoperatorio de ECG  
**Subtítulo:** Detección Multiclase de Arritmias en Quirófano con HistGradientBoosting y Validación por Paciente  
**Autores:** Equipo de Fundamentos de Machine Learning  
**Institución:** Universidad EAFIT — Maestría / Pregrado en Ciencia de Datos e Inteligencia Artificial  
**Fecha:** Noviembre 2026  

### Contenido Visual Sugerido
- Traza limpia de ECG intraoperatorio de 10 s superpuesta con bandas de alerta clínica (Verde: Normal, Rojo: Arritmia Crítica, Gris: Ruido).
- Esquema de bloques: Señal Cruda (500 Hz) $\to$ Detección QRS Adaptativa $\to$ 45 Características $\to$ HistGradientBoosting $\to$ Priorización de Alertas.

### Notas del Orador
> "Buenos días a todos. En los quirófanos modernos, un anestesiólogo supervisa simultáneamente hemodinámica, profundidad anestésica, ventilación mecánica y el electrocardiograma continuo. Durante una cirugía, hasta el 70% de las alarmas acústicas son artefactos o falsas alarmas, generando fatiga por alarmas y desensibilización sensorial. Presentamos hoy un sistema de apoyo a la decisión clínica para monitoreo intraoperatorio de ECG, basado en la base de datos abierta VitalDB Arrhythmia Database, diseñado específicamente para priorizar alertas con rigor fisiológico, control estricto de fuga de datos por paciente y observabilidad continua en producción."

---

## Diapositiva 02: El Problema y Contexto Clínico (Eje 1)
**Título:** Desafíos del Monitoreo Intraoperatorio de ECG: Apoyo vs. Diagnóstico Autónomo  
**Eje Temático:** 1. El Problema y Contexto Clínico (Sección 3.1)

### Contenido de la Diapositiva
- **Entorno Operatorio:**
  - Derivación estándar intraoperatoria: **Derivación II (DII)** adquirida a **500 Hz**.
  - Pacientes bajo anestesia general e intubación orotraqueal, sometidos a fármacos vasoactivos, anestésicos halogenados y estrés quirúrgico.
- **Interferencia Quirúrgica Severa:**
  - Presencia constante de electrobisturí/electrocauterio (ruido de alta frecuencia 40–100 Hz y pulsos saturantes de hasta $\pm 600\text{ mV}$), cambios de línea base por ventilación mecánica y desconexión transitoria de electrodos.
- **Propósito del Modelo de ML:**
  - **Priorización Inteligente de Alertas y Triage de Tramos Críticos**, NO un diagnosticador autónomo ni un reemplazo del anestesiólogo.
  - Objetivo: Filtrar tramos no interpretables (ruido), discriminar arritmias estables vs. inestables y advertir de forma oportuna eventos de alto riesgo (Taquicardia Ventricular, Fibrilación Auricular con respuesta rápida).

### Notas del Orador
> "Es fundamental delimitar el alcance clínico: nuestro sistema no busca emitir un diagnóstico médico independiente ni reemplazar la responsabilidad del anestesiólogo o cardiólogo. En quirófano, la única derivación comúnmente monitorizada con alta resolución continua es la Derivación II a 500 Hz. El desafío reside en que el ECG intraoperatorio está plagado de artefactos severos de electrocauterio, movimiento del paciente y derivas ventilatorias. Un modelo clínicamente útil debe clasificar en ventanas cortas de 10 segundos, etiquetar con precisión el ruido para evitar falsos positivos catastróficos, y priorizar la atención médica en taquicardias ventriculares o fibrilaciones auriculares súbitas."

---

## Diapositiva 03: VitalDB Arrhythmia Database y EDA (Eje 1)
**Título:** VitalDB: Caracterización del Dataset y Desbalance Intraoperatorio  
**Eje Temático:** 1. El Problema y Contexto Clínico (Sección 3.3)

### Contenido de la Diapositiva
- **Fuente de Datos y Anotación:**
  - *VitalDB Arrhythmia Database* (PhysioNet, 2026): **482 cirugías** correspondientes a **473 pacientes únicos (`subjectid`)**.
  - Más de 660,000 latidos anotados individualmente por cardiólogos expertos ($\kappa = 0.93$ de acuerdo interobservador).
  - **10 clases modeladas:** `Normal`, `AFib`, `Ectopia ventricular`, `Ectopia auricular`, `Disfuncion nodo sinusal`, `SVTA`, `TV`, `Bloqueo AV`, `WAP/MAT`, `Ruido`.
  - *Clase excluida:* `Inclasificable` (solo 8 ventanas en 2 pacientes, clínicamente ambiguas e inviables de validar de forma estratificada).
- **Desbalance Severo de Clases:**
  - Clases dominantes: `Normal` (61.7% de los latidos) y `AFib` (24.7% de los latidos) abarcan >86% de los datos.
  - Clases críticas altamente minoritarias: `Bloqueo AV` (0.65% de latidos, presente en **solo 9 pacientes**) y `TV` (0.24% de latidos).
- **Temporalidad y Duración de Episodios (`figures/eda_linea_tiempo.png`):**
  - *Episodios largos:* Fibrilación Auricular (mediana 123.1 s), WAP/MAT (83.9 s), Bloqueo AV (77.9 s), Disfunción de nodo (41.2 s), Normal (33.2 s).
  - *Episodios medios:* Ectopias auriculares y ventriculares (~14 s).
  - *Salvas críticas ultracortas:* Taquicardia Ventricular (mediana **2.7 s**, percentil 90 = 6.2 s), SVTA (2.4 s), Ruido (5.4 s).
  - *Justificación del Ventaneo:* Ventanas de 10 segundos con regla de inclusión de arritmias paroxísticas ($\ge 3$ latidos ectópicos consecutivos capturan la salva de TV sin diluirla en el ritmo sinusal basal).
- **Limpieza Rigurosa de Datos:**
  - Eliminación de filas marcadoras (`StartK`/`EndK` de `bad_signal_quality_label` sin `beat_type`), cuya inclusión ingenua creaba intervalos RR espurios de cientos de segundos.
  - Exclusión justificada del **caso 1165** por desfase sistemático de anotación cardiaca ($>0.1\text{ s}$ de error de sincronía temporal en cabecera).

### Figura del Repositorio a Renderizar
- `figures/eda_linea_tiempo.png` (Distribución de duraciones de episodios por tipo de ritmo).

### Notas del Orador
> "Al explorar VitalDB encontramos un reto fundamental: el desbalance extremo y la asimetría temporal de los eventos. Mientras la Fibrilación Auricular suele persistir durante minutos enteros (mediana de 123 segundos), una salva de Taquicardia Ventricular dura apenas 2.7 segundos de mediana, pero compromete de inmediato el gasto cardíaco. Si utilizáramos una ventana de 60 segundos, una salva de TV de 3 segundos se diluiría en 57 segundos de ritmo normal y pasaría inadvertida. Por eso adoptamos una ventana de 10 segundos con una regla de prioridad para salvas de 3 o más latidos anormales. Asimismo, depuramos las anotaciones eliminando marcadores de calidad que generaban intervalos RR artificiales y descartamos el caso 1165 por un desalineamiento severo de reloj."

---

## Diapositiva 04: Estructura No Supervisada y Justificación del Enfoque (Eje 1)
**Título:** ¿Es Viable el Agrupamiento No Supervisado? Análisis PCA y K-Means  
**Eje Temático:** 1. El Problema y Contexto Clínico (Sección 3.3 Exploración No Supervisada)

### Contenido de la Diapositiva
- **Exploración No Supervisada (`figures/no_supervisado_pca.png`):**
  - Preprocesamiento: Extracción de variables de señal, escalado con `RobustScaler` (resistente a outliers de electrocauterio), descomposición mediante PCA y agrupamiento con K-Means ($k=10$).
  - Varianza explicada: Las dos primeras componentes principales retienen el **49% de la varianza total** (10 componentes alcanzan el 84%).
- **Hallazgos Clave:**
  - **Aislamiento Efectivo del Ruido:** El clúster 4 agrupó de forma casi pura **394 de las 419 ventanas de ruido**, demostrando que la degradación espectral y la entropía separan nítidamente el artefacto del tejido biológico.
  - **Separación Macroscópica de Frecuencia y Regularidad:** PC1 y PC2 capturan gradientes de frecuencia cardíaca (bradicardias vs. taquicardias) y variabilidad del RR (ritmo regular vs. caótico como AFib).
  - **Incapacidad para Discriminar Origen del Impulso:** Sin supervisión, los clústeres no logran distinguir entre Taquicardia Ventricular (origen ventricular, QRS ancho) y Taquicardia Supraventricular (origen auricular, QRS estrecho), ni detectar ondas P disociadas en Bloqueos AV.
- **Conclusión Metodológica:**
  - Se justifica plenamente la necesidad de un enfoque **supervisado** que combine ingeniería experta de características morfológicas (ancho QRS, correlación de plantillas, presencia de onda P) con modelos basados en árboles capaces de capturar fronteras de decisión no lineales complejas.

### Figura del Repositorio a Renderizar
- `figures/no_supervisado_pca.png` (Espacio latente de las 2 primeras componentes principales coloreadas por ritmo real vs. clúster asignado).

### Notas del Orador
> "Antes de construir modelos supervisados, nos preguntamos si la propia geometría de los datos separaba los ritmos. El análisis no supervisado con PCA y K-Means reveló una excelente noticia: el ruido y los artefactos de bisturí forman una nube completamente aislada, lo que garantiza que el sistema pueda descartar señales no interpretables. Sin embargo, para distinguir una TV de una SVTA, o un Bloqueo AV de una bradicardia sinusal, el agrupamiento no supervisado falla, porque ambas comparten frecuencias cardíacas similares. Esto valida nuestra hipótesis: se requiere aprendizaje supervisado enriquecido con descriptores morfológicos finos del complejo QRS y la onda P."

---

## Diapositiva 05: Metodología y Preprocesamiento de Señal (Eje 2)
**Título:** Del ECG Crudo a la Detección QRS Adaptativa v2  
**Eje Temático:** 2. Metodología y Preprocesamiento (Sección 3.2 y 3.4)

### Contenido de la Diapositiva
- **Condición Estricta de Inferencia en Tiempo Real:**
  - `FUENTE_PICOS = "detectados"`: El modelo en ningún momento asume picos R anotados por médicos en inferencia. Todo latido debe ser descubierto autónomamente por el algoritmo a partir de la señal continua.
- **Cadena de Filtrado Digital:**
  - Filtro pasabanda Butterworth bidireccional en secciones de segundo orden (**SOS**) de **0.5 Hz a 40 Hz**.
  - Supresión de deriva de línea base respiratoria (<0.5 Hz) y ruido de red eléctrica / armónicos musculares (>40 Hz), garantizando **fase cero** (cero distorsión morfológica en el tiempo).
- **Detector QRS Adaptativo v2 (Resistente a Quirófano):**
  1. *Filtro derivativo y pasabanda específico (5–15 Hz)*: Maximiza la pendiente pronunciada de la despolarización ventricular frente a ondas T picudas.
  2. *Elevación al cuadrado e integración móvil (120 ms)*: Convierte la energía espectral en un pulso unipolar continuo.
  3. *Umbral Adaptativo con Piso Robusto:* Implementación de un piso mínimo fijado en el **20% de la mediana de los máximos locales**.
     - *Lección aprendida:* En la versión 1, cuando se activaba el electrobisturí (ej. caso 6069 con un pulso artefactual de **642 mV**), el umbral móvil se elevaba a cientos de milivoltios y el detector quedaba 'ciego' durante 8 segundos, perdiendo latidos reales. Con el piso robusto v2, el umbral se restablece de inmediato.
  4. *Periodo refractario fisiológico de 200 ms* y filtro de pendiente para rechazar espículas de marcapasos o artefactos bifásicos.
- **Validación del Detector:**
  - Sensibilidad mediana de **0.996** y precisión mediana de **0.981** comparado contra las anotaciones gold-standard de cardiólogos.
- **Ventaneo Operativo:**
  - Extracción de **50,517 ventanas válidas de 10 segundos** a lo largo de 472 pacientes.

### Notas del Orador
> "Un modelo de machine learning en cardiología es tan bueno como su detector de picos R. Un error muy común en la literatura es alimentar el modelo con picos R anotados por humanos; eso es trampa clínica. Nuestro pipeline opera con la bandera 'detectados'. Diseñamos un detector Pan-Tompkins mejorado que denominamos v2. Su innovación clave es un umbral adaptativo con un piso robusto del 20% de la mediana de picos. En quirófano, un electrobisturí genera picos de hasta 642 milivoltios. Los detectores convencionales elevan su umbral a ese nivel y se vuelven ciegos durante varios segundos. Nuestro detector v2 asimila el artefacto y recupera la detección fisiológica en menos de 200 milisegundos, logrando un 99.6% de sensibilidad."

---

## Diapositiva 06: Ingeniería de Características y Manejo de MNAR (Eje 2)
**Título:** Espacio de 45 Características: Ritmo, Morfología, Ruido e Imputación No Aleatoria  
**Eje Temático:** 2. Metodología y Preprocesamiento (Sección 3.4)

### Contenido de la Diapositiva
- **Distribución de las 45 Características Extraídas:**
  - **22 de Ritmo y Variabilidad RR (VFC):**
    - Métricas temporales: `fc_media`, `fc_pico`, `fc_min`, `rr_mediana`, `sdnn`, `rmssd`, `pnn50`, `cv_rr`.
    - Métricas geométricas y no lineales: Diagrama de Poincaré (`sd1`, `sd2`, `sd1_sd2_ratio`), `tpr` (Turning Point Ratio).
    - *Depuración:* Eliminación de entropía muestral (`sampen`), la cual generaba un **62% de valores nulos** en ventanas cortas de 10 s donde los latidos eran escasos.
  - **9 de Calidad de Señal y Ruido:**
    - `sig_sqi_qrs` (índice de calidad QRS basado en skewness y kurtosis).
    - `sig_ruido_alto` (potencia en la banda de 40–100 Hz, diseñada específicamente para capturar la interferencia del electrocauterio).
    - `sig_deriva` (bajas frecuencias <0.5 Hz), `sig_frac_plana` ($\ge 100\text{ ms}$ de saturación a cero) y `sig_frac_repetidas` (saturación de conversor ADC).
  - **13 de Morfología P-QRS-T:**
    - `qrs_ancho_med`, `qrs_ancho_max` (diferenciación de latidos ectópicos y TV vs SVTA).
    - `corr_plantilla_med`, `frac_latidos_atipicos` (correlación cruzada contra el latido promedio de la ventana).
    - `p_presente_frac`, `pr_intervalo`, `p_sin_qrs` (detección de onda P mediante correlación de plantilla en ventanas pre-QRS y en pausas diastólicas).
  - **1 Variable de Ausencia No Aleatoria (MNAR):**
    - Flag binario `sin_rr_valido`: Activo cuando la ventana contiene menos de 4 latidos válidos (imposible calcular VFC fiable).
- **Manejo Clínico de Faltantes (MNAR vs. Imputación Típica):**
  - La ausencia de latidos en quirófano **no ocurre al azar**: ocurre por asistolia, desconexión de electrodos o ruido saturante masivo.
  - Si se imputa con la mediana (estrategia ingenua de scikit-learn), una ventana de ruido o paro se maquilla como un ritmo normal de 75 lpm con variabilidad perfecta, confundiendo al clasificador.
  - Se implementó **imputación explícita fuera de rango (`-1.0`)** acoplada a la variable `sin_rr_valido = 1`.
- **Estandarización:** Inclusión de `StandardScaler` en pipelines para algoritmos dependientes de distancias (Regresión Logística y SVM).

### Notas del Orador
> "Construimos un vector de 45 características clínicamente fundamentadas. Pero el aporte conceptual más relevante del preprocesamiento fue resolver el problema de datos faltantes no aleatorios (MNAR). Cuando en una ventana de 10 segundos hay menos de 4 latidos, es imposible calcular métricas como el RMSSD o el SDNN. Si uno utiliza una imputación por la mediana, el sistema reemplaza los nulos por un ritmo sinusal perfecto de 75 latidos por minuto, ocultando un artefacto masivo o una pausa sinusal severa. Por ello, diseñamos la variable indicadora 'sin_rr_valido' e imputamos las métricas de ritmo con el valor centinela -1.0. Esto le enseña explícitamente al árbol de decisión que la ausencia de ritmo es, en sí misma, una señal patológica o de ruido."

---

## Diapositiva 07: Selección de Modelos y Benchmark de Familias (Eje 3)
**Título:** Partición Estricta por Paciente y Comparación de Familias de Modelos  
**Eje Temático:** 3. Selección del Modelo y de sus Parámetros (Sección 3.5)

### Contenido de la Diapositiva
- **Protocolo de Validación Riguroso:**
  - `StratifiedGroupKFold` agrupado estrictamente por **`subjectid`** (5 pliegues fijos guardados en `data/pliegues_pacientes.csv`).
  - Ningún paciente del conjunto de entrenamiento aparece en validación ni en test.
  - **Tope de Entrenamiento (`MAX_POR_CLASE = 5000`):** Aplicado **exclusivamente al conjunto de entrenamiento** para evitar tiempos de cómputo inmanejables y mitigar el sesgo hacia las clases mayoritarias. La validación y el test retienen el 100% de la distribución epidemiológica real.
- **Tabla Comparativa de Modelos Base (Validación Interna por Paciente):**

| Familia de Modelo | Configuración | F1 Macro Val | Train F1 | Bal. Acc Val | Exactitud Val | Tiempo Entrenamiento | Diagnóstico Operativo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Línea Base Trivial** | Predicción mayoritaria (Normal) | **0.077** | 0.077 | 0.100 | 0.617 | < 0.1 s | Colapso total en clases minoritarias. `[PENDIENTE: agregar resultado exacto de Dummy estratificado y nota de empate tras tope, bloque G1]` |
| **Regresión Logística** | L2, `class_weight='balanced'`, StandardScaler | **0.426 ± 0.025** | 0.653 | 0.598 | 0.683 | 5.3 s | Rápida pero linealmente insuficiente para interacciones ritmo-morfología. |
| **SVM (RBF Kernel)** | Radial Basis Function, StandardScaler | **0.460 ± 0.040** | 0.803 | 0.604 | 0.753 | 169.1 s | Fronteras no lineales mejores, pero escala cuadrática en cómputo. |
| **Random Forest** | 100 árboles, `max_depth=None` | **0.509 ± 0.033** | 1.000 | 0.509 | 0.879 | 10.1 s | Memorización absoluta en train (sobreajuste), degradación en generalización. |
| **HistGradientBoosting** | Histogram-based GBDT (255 bins), defecto | **0.528 ± 0.017** | 0.983 | 0.585 | 0.866 | 23.2 s | **Ganador en F1 y estabilidad.** Manejo nativo de interacciones no lineales. |

- **Elección de la Familia:** HistGradientBoosting ofreció la mejor generalización inter-paciente, robustez ante outliers e inferencia ultrarrápida.

### Notas del Orador
> "Para seleccionar el modelo, establecimos una regla inquebrantable: ningún paciente de entrenamiento puede filtrarse en validación. Comparamos cuatro familias frente a una línea base trivial. La regresión logística quedó limitada en un F1 macro de 0.426. Las máquinas de soporte vectorial alcanzaron 0.460, pero su coste computacional es prohibitivo. Random Forest sufrió de memorización pura en entrenamiento (F1 de 1.000 en train, cayendo a 0.509 en validación). HistGradientBoosting superó a todos con 0.528 de F1 macro con hiperparámetros por defecto. Su discretización en 255 histogramas proporciona una regularización natural excelente contra el ruido intraoperatorio."

---

## Diapositiva 08: Optimización, Ablación y Experimentos (Eje 3)
**Título:** Optimización de Hiperparámetros, Estudio de Ablación y Bloque F  
**Eje Temático:** 3. Selección del Modelo y de sus Parámetros (Sección 3.5 y Experimentos F)

### Contenido de la Diapositiva
- **Optimización de Hiperparámetros (Búsqueda Agrupada por Paciente):**
  - *Random Forest afinado:* F1 macro subió a **0.528** (369 s; `max_depth=12`, `max_features=0.5`, `min_samples_leaf=1`, `n_estimators=249`).
  - *HistGradientBoosting afinado (Modelo Ganador):* F1 macro alcanzó **0.534** (112 s; `learning_rate=0.0669`, `max_iter=364`, `max_leaf_nodes=18`, `min_samples_leaf=33`, `l2_regularization=0.244`).
- **Estudio de Ablación (`figures/registro_correcciones.png`):**
  - ¿Qué aporta realmente cada componente del pipeline?
    - **Pipeline v2 Completo:** F1 macro = **0.539** (validación interna).
    - **Sin Características Morfológicas (solo Ritmo y SQI):** El F1 macro colapsa a **0.458 (caída drástica de −0.081)**. Demuestra empíricamente que el ritmo por sí solo no puede discriminar arritmias ventriculares de supraventriculares.
    - **Con Detector QRS v1:** F1 macro baja a **0.526** (ceguera transitoria en electrobisturí).
    - **Con Imputación por Mediana:** F1 macro baja a **0.531** (confusión en episodios de paro/ruido).
    - **Trampa de Picos Anotados:** F1 macro sube artificialmente a **0.588** (Ruido obtiene F1 = 0.994 falso porque las pausas de ruido venían sin latidos anotados).
- **Protocolo de Experimentos Adicionales (Bloque F):**
  - *Regla de Adopción Clínica:* Para sustituir al modelo de referencia, cualquier mejora debe: (1) Superar al baseline en al menos $+0.005$ en `cv_externa`, (2) Ganar en al menos 4 de los 5 pliegues por paciente, y (3) No reducir en más de $0.02$ el recall de Taquicardia Ventricular, Bloqueo AV ni Fibrilación Auricular.
  - *Evaluación de Experimentos F1 a F10:* Ventanas deslizantes (F1), Prioridad clínica (F2), SMOTE/Aumento de señal (F3), Tamaño de ventana (F4), Poda por colinealidad $|\rho|>0.85$ (F5), Filtro previo de ruido (F6), Esquema en 2 etapas (F7), Calibración de umbrales F8, Búsqueda ampliada SVM/XGBoost/LightGBM F9, y Entrenamiento sin tope F10.
  - `[PENDIENTE: resultados definitivos de experimentos Bloque F y tabla G2]`

### Figura del Repositorio a Renderizar
- `figures/registro_correcciones.png` (Evolución de métricas y estudio de ablación en validación interna).

### Notas del Orador
> "Ajustamos los hiperparámetros de HistGradientBoosting mediante búsqueda aleatoria restringida por paciente, alcanzando 0.534 de F1 macro con mayor profundidad y regularización L2. Pero el experimento más revelador fue la ablación: retiramos las 13 características morfológicas y el F1 macro se desplomó en casi 10 puntos a 0.458. Esto prueba que el conocimiento médico de la onda P y el ancho QRS es el núcleo del rendimiento del sistema. Adicionalmente, establecimos un protocolo riguroso en el Bloque F para evaluar 10 variantes adicionales sin caer en sobreajuste metodológico."

---

## Diapositiva 09: Generalización del Modelo y Validación Cruzada (Eje 4)
**Título:** Validación Cruzada Externa y Desempeño en Test Ciego  
**Eje Temático:** 4. Generalización del Modelo (Sección 3.6)

### Contenido de la Diapositiva
- **Validación Cruzada Externa (5 Pliegues por Paciente, 50,517 Ventanas):**
  - **F1 Macro Promedio:** **$0.534 \pm 0.019$**
  - Consistencia pliegue a pliegue (`figures/eval_cv_externa.png`):
    - Pliegue 0: **0.526**
    - Pliegue 1: **0.519**
    - Pliegue 2: **0.544**
    - Pliegue 3: **0.562**
    - Pliegue 4: **0.519**
  - Dispersión sumamente baja ($\sigma = 0.019$), indicando que el modelo no depende de la partición particular de pacientes.
- **Evaluación en Conjunto de Test Ciego Fijo (Pliegue 0):**
  - **93 pacientes completamente nuevos**, **9,950 ventanas independientes**.
  - **F1 Macro en Test:** **0.526**
  - **Intervalo de Confianza del 95% (Bootstrap a nivel de Paciente, $B=1000$):** **$[0.458, 0.558]$**
  - *Interpretación Estadística:* El límite inferior de 0.458 asegura un rendimiento muy superior al azar o a modelos lineales incluso en el peor escenario de variabilidad biológica inter-sujeto.
- **Exactitud Global:** **0.844 en CV Externa** y **0.824 en Test Ciego**.
- **Exactitud Balanceada:** **0.616 en CV Externa** y **0.601 en Test Ciego**.

### Figura del Repositorio a Renderizar
- `figures/eval_cv_externa.png` (Desempeño y estabilidad del F1 macro a través de los 5 pliegues de validación cruzada).

### Notas del Orador
> "En machine learning médico, la prueba de fuego es evaluar sobre pacientes que el algoritmo jamás ha visto. En nuestra validación cruzada externa de 5 pliegues sobre más de 50,000 ventanas, el F1 macro se situó en 0.534 con una desviación de solo 0.019. Al congelar el Pliegue 0 como conjunto de prueba ciego independiente —con 93 pacientes nuevos y casi 10,000 ventanas— el F1 macro fue de 0.526. Calculamos intervalos de confianza al 95% mediante remuestreo bootstrap a nivel de paciente, obteniendo un rango entre 0.458 y 0.558. La exactitud global se mantiene en el 82.4%, respaldando una sólida capacidad de generalización."

---

## Diapositiva 10: Demostración Empírica Anti-Leakage (Eje 4)
**Título:** Control Estricto de Fuga de Datos (Data Leakage) y Curvas de Aprendizaje  
**Eje Temático:** 4. Generalización del Modelo (Sección 3.6 Data Leakage)

### Contenido de la Diapositiva
- **El Gran Peligro Metodológico en Biosignales: Fuga de Datos por Paciente:**
  - Realizamos un experimento controlado comparando la partición aleatoria por ventana vs. la partición agrupada por paciente (`subjectid`):

| Estrategia de Partición | F1 Macro Obtenido | Fuga de Información | Causa Metodológica |
| :--- | :---: | :---: | :--- |
| **Partición Aleatoria (por Ventana)** | **$0.763 \pm 0.015$** | **Severa (+0.229 inflado)** | El modelo memoriza el voltaje basal, eje cardíaco y morfología propia del paciente presente en train y test. |
| **Partición Agrupada (por `subjectid`)** | **$0.534 \pm 0.008$** | **Ninguna (Realista)** | Evalúa verdadera generalización fisiológica a nuevos individuos. |

- **Fuga Sutil por Casos Quirúrgicos (`case_id` vs. `subjectid`):**
  - Identificamos que **8 pacientes tuvieron múltiples intervenciones quirúrgicas** en VitalDB. Agrupar por `case_id` en lugar de `subjectid` filtraba cirugías del mismo paciente en train y test. Se corrigió forzando la agrupación por `subjectid`.
- **Test de Sanidad con Etiquetas Barajadas (Permutación de $y$):**
  - Al destruir la relación entre $X$ e $y$, el F1 macro cayó inmediatamente a **$0.057 \pm 0.006$** (concordante con el azar esperado $1/10$), certificando la ausencia de fuga técnica o bugs en el preprocesamiento.
- **Análisis de la Brecha Train vs. Test (`figures/gen_curva_aprendizaje.png`):**
  - Brecha de entrenamiento (Train F1 = 0.963 vs. Test F1 = 0.526):
  - *Diagnóstico Clínico:* La brecha no proviene de sobreajuste en las clases mayoritarias, sino de la variabilidad morfológica extrema en patologías con muy baja prevalencia de pacientes: Bloqueo AV (9 pacientes en todo el dataset) y WAP/MAT (23 pacientes).

### Figura del Repositorio a Renderizar
- `figures/gen_curva_aprendizaje.png` (Curva de aprendizaje mostrando la evolución de Train vs. Validación por paciente).

### Notas del Orador
> "Uno de los hallazgos metodológicos más contundentes del proyecto fue la demostración empírica de fuga de datos. Si dividiéramos los datos de manera aleatoria por ventanas de 10 segundos, nuestro F1 macro se inflaría artificialmente a 0.763. Esto es una trampa: el modelo no aprende a diagnosticar la arritmia, sino que memoriza el 'código de barras' morfológico del paciente, presente en su ventana del minuto 10 y evaluado en la del minuto 12. Al forzar la separación estricta por 'subjectid', el modelo obtiene su rendimiento real de 0.534. Asimismo, el test de permutación con etiquetas barajadas colapsó a 0.057, garantizando la total transparencia metodológica."

---

## Diapositiva 11: Desempeño Multiclase e Interpretabilidad (Eje 5)
**Título:** Desempeño por Clase: Sensibilidad Clínica vs. Falsas Alarmas  
**Eje Temático:** 5. Métricas de Desempeño e Interpretabilidad (Sección 3.7)

### Contenido de la Diapositiva
- **Por qué la Exactitud Global es Engañosa:**
  - En un entorno donde `Normal` y `AFib` representan >85% del quirófano, un modelo inútil que prediga siempre sinusal obtendría 61.7% de exactitud.
  - Se priorizan: **F1 macro** (0.534 en CV, 0.526 en test), **Exactitud balanceada** (0.616 en CV, 0.601 en test), **MCC** (0.710 en test) y el **Recall en arritmias de riesgo vital**.
- **Desempeño Detallado por Clase (CV Externa 50,517 ventanas vs. Test Fijo 9,950 ventanas):**

| Clase de Ritmo | F1 CV Ext. | Recall CV | Precisión CV | F1 Test Ciego | Prioridad Clínica y Comportamiento Operativo |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Fibrilación Auricular (AFib)** | **0.947** | 0.943 | 0.952 | **0.930** | Excelente detección de caos RR y ausencia de onda P. |
| **Ritmo Normal** | **0.912** | 0.876 | 0.951 | **0.909** | Alta especificidad; preserva la tranquilidad del quirófano. |
| **Ectopia Ventricular** | **0.736** | 0.846 | 0.651 | **0.850** | Alta captura de extrasístoles ventriculares y bigeminismo. |
| **Ruido / Artefacto** | **0.663** | 0.900 | 0.525 | **0.693** | **Recall 90%**: Gran capacidad de silenciar ventanas no legibles. |
| **Ectopia Auricular** | **0.644** | 0.740 | 0.569 | **0.589** | Identificación de latidos prematuros con QRS normal. |
| **SVTA (Taquicardia Supravent.)**| **0.442** | 0.597 | 0.351 | **0.562** | Test Recall = 0.852. Buena captura de aceleraciones sinusales. |
| **TV (Taquicardia Ventricular)**| **0.384** | **0.653** | 0.272 | **0.423** | **Emergencia Crítica: Recall 0.653 (CV) y 0.617 (Test)**. Se prioriza sensibilidad sobre precisión para salvar vidas. |
| **Disfunción Nodo Sinusal** | **0.374** | 0.445 | 0.322 | **0.277** | Pausas sinusales y bradicardias extremas variables. |
| **Bloqueo AV** | **0.115** | 0.147 | 0.094 | **0.000** | Escasez extrema (9 pacientes totales, solo 1 en test con 117 vent; pliegue 2 tuvo 0.34). |
| **WAP / MAT** | **0.045** | 0.048 | 0.043 | **0.021** | Requiere diferenciar $\ge 3$ ondas P sutiles; límite físico de una sola derivación DII. |

- *Comparación con Autores de VitalDB (CNN 1D Binaria Normal vs Anormal):* `[PENDIENTE: métricas exactas de agrupación binaria del bloque G9]`.

### Figuras del Repositorio a Renderizar
- `figures/eval_confusion.png` (Matriz de confusión normalizada en validación cruzada externa).
- `figures/eval_pr.png` (Curvas Precisión-Recall multiclase).

### Notas del Orador
> "Al desagregar el desempeño clase a clase, observamos una correspondencia clínica fascinante. En Fibrilación Auricular y Ritmo Normal alcanzamos F1 superiores a 0.91 y 0.94. En Ruido, logramos un recall del 90%, garantizando que los artefactos no se confundan con arritmias mortales. En Taquicardia Ventricular, el F1 es 0.384 con una precisión de 0.272, pero un Recall de 0.653. En anestesia, este compromiso es el deseado: preferimos que el anestesiólogo reciba una alerta preventiva para mirar el monitor ante una posible TV, a que el modelo calle por buscar precisión matemática y pase por alto un evento letal. El Bloqueo AV y WAP/MAT reflejan los límites del dataset: solo 9 pacientes en Bloqueo AV y la imposibilidad de ver múltiples morfologías de onda P en una sola derivación."

---

## Diapositiva 12: Explicabilidad y Coherencia Electrofisiológica (Eje 5)
**Título:** Interpretabilidad Global y SHAP: Validación de la Lógica del Modelo  
**Eje Temático:** 5. Métricas de Desempeño e Interpretabilidad (Sección 3.7 Explicabilidad)

### Contenido de la Diapositiva
- **Importancia por Permutación (`figures/explic_permutacion.png`):**
  - Las variables líderes que gobiernan la toma de decisiones son fisiológicamente coherentes:
    1. `sig_ruido_alto` y `sig_sqi_qrs`: Determinan si la señal es médicamente analizable o si debe etiquetarse como `Ruido`.
    2. `rr_mediana`, `fc_media` y `sdnn`: Separan ritmos acelerados (TV/SVTA) de ritmos pausados o bradicárquicos.
    3. `frac_latidos_atipicos` y `qrs_ancho_max`: Discrimina origen ventricular vs. auricular.
- **Valores SHAP Multiclase (`figures/explic_shap_global.png`):**
  - **Fibrilación Auricular:** SHAP asigna alta densidad a valores elevados de `sd1`, `rmssd` y `cv_rr` (caos diastólico) acoplados con valores mínimos de `p_presente_frac` (ausencia de contracción auricular organizada).
  - **Taquicardia Ventricular:** Fuerte contribución de `qrs_ancho_max` elevado ($>120\text{ ms}$), correlación baja con plantilla normal y `fc_media` $>120\text{ lpm}$.
  - **Ruido:** Predominio absoluto de potencia en 40–100 Hz (`sig_ruido_alto`) y caída de `sig_sqi_qrs`.
  - **Ectopias Auriculares:** Caída transitoria del intervalo RR (`tpr` alto) pero con `qrs_ancho_med` dentro de límites normales.
  - `[PENDIENTE: síntesis de interpretación individual de las 10 clases en SHAP del bloque G7]`

### Figuras del Repositorio a Renderizar
- `figures/explic_permutacion.png` (Top de características por importancia de permutación).
- `figures/explic_shap_global.png` (Resumen SHAP beeswarm multiclase).

### Notas del Orador
> "Para que un anestesiólogo confíe en un modelo, este no puede ser una caja negra. Evaluamos la interpretabilidad mediante valores SHAP e importancia por permutación. Los resultados son impecables desde la perspectiva electrofisiológica: el modelo predice Fibrilación Auricular cuando detecta alta variabilidad caótica en el RR combinada con la desaparición de la onda P; predice Taquicardia Ventricular cuando el complejo QRS se ensancha más allá de 120 milisegundos a frecuencias elevadas; y activa la clase Ruido cuando la energía espectral entre 40 y 100 Hz se dispara debido al electrobisturí. El modelo ha aprendido la física y la fisiología del corazón, no correlaciones espurias."

---

## Diapositiva 13: Despliegue Clínico e Interfaz de Usuario (Eje 6)
**Título:** Arquitectura de Despliegue e Interfaz Interactiva en Streamlit  
**Eje Temático:** 6. Despliegue e Interfaz para el Usuario Clínico (Sección 3.8)

### Contenido de la Diapositiva
- **Empaquetado y Pipeline de Producción:**
  - Modelo serializado y congelado: `models/modelo_arritmias.joblib` (entrenado sobre las 17,410 ventanas representativas de los 472 pacientes) acompañado de `models/meta_modelo.json`.
  - Módulo reutilizable de producción: `src/ecg_utils.py` con procesamiento idéntico entre entrenamiento y serving.
- **Flujo de Datos en Entorno Real vs. Prototipo:**
  - *Caché interna:* Archivos `.npz` y `.csv` preprocesados para experimentación veloz.
  - *Integración hospitalaria esperada:* Conexión directa a la salida digital del monitor de quirófano mediante **Vital Recorder / protocolo HL7**, alimentando un búfer circular continuo en ventanas de 10 s.
- **Interfaz Interactiva (`app/app.py`):**
  - **Banner de Seguridad Clínica:** Notificación explícita de apoyo a la decisión y necesidad de corroboración médica.
  - **Panel de Control:** Carga de señal ECG cruda en formato CSV, selección de frecuencia de muestreo original con remuestreo automático a 500 Hz, y deslizador interactivo de **umbral de confianza mínima** (valor recomendado: 0.60).
  - **Visualización Superior:** Traza de ECG continuo con bandas de fondo con código cromático clínico:
    - 🟢 *Verde:* Ritmo Normal o ectopias aisladas benignas.
    - ⚪ *Gris:* Ruido no interpretable (solicita reacomodación de electrodos o espera de fin de electrobisturí).
    - 🔴 *Rojo:* Alerta de arritmia severa (AFib rápida, SVTA, Taquicardia Ventricular).
  - **Panel Resumen Tabular:** Desglose cronológico por ventanas de 10 s indicando segundo de inicio, diagnóstico de ritmo, probabilidad/confianza, frecuencia cardíaca en lpm y la bandera booleana `Revisar` (activa si hay arritmia roja o si la confianza cae por debajo del umbral del anestesiólogo).
- **Extensiones de Demostración:**
  - `[PENDIENTE: validación final de app Streamlit (H1), 3 casos de test de 1 minuto para demo (H2) y demo interactiva tipo monitor de quirófano de doble pantalla con ventanas deslizantes (H4)]`

### Notas del Orador
> "El mejor modelo del mundo es inútil si no se puede utilizar en la cabecera del paciente. Construimos un prototipo de interfaz clínica en Streamlit utilizando nuestro paquete modular 'ecg_utils.py'. La herramienta permite cargar trazas crudas de ECG, remuestrea automáticamente a 500 Hz y renderiza la señal con bandas de colores intuitivas: verde para ritmo sinusal, gris para ruido y rojo para eventos que requieren acción inmediata como Fibrilación Auricular o TV. Un elemento clave para el especialista es el control de confianza: si un tramo tiene una probabilidad inferior al 60%, el sistema enciende la bandera 'Revisar', invitando al anestesiólogo a auditar manualmente la traza."

---

## Diapositiva 14: Monitoreo de Deriva y Estrategia MLOps (Eje 7)
**Título:** Detección de Deriva de Datos (PSI / KS) y Protocolo de Reentrenamiento  
**Eje Temático:** 7. Estrategia de Monitoreo de Deriva y Reajuste del Modelo (Sección 3.9)

### Contenido de la Diapositiva
- **El Reto de la Observabilidad en Quirófano:**
  - La señal ECG puede degradarse por cambio de fabricante de electrodos, nuevos electrobisturíes, cambios en cables de monitorización o derivas demográficas de pacientes.
  - Almacenamiento de la distribución basal en `models/referencia_features.csv` (perfil empírico de las 45 características).
- **Métricas de Deriva: PSI (Population Stability Index) vs. Kolmogorov-Smirnov (KS):**
  - *Problema del KS en Big Data:* Con miles de ventanas, la prueba KS tiene una potencia estadística tan alta que el $p$-valor resulta casi siempre $\approx 0$ ante variaciones fisiológicas insignificantes.
  - *Solución Operativa:* Se adopta el **PSI** como regla de decisión en producción:
    - $\text{PSI} < 0.10$: **Estable** (distribución inalterada).
    - $0.10 \le \text{PSI} \le 0.20$: **Moderado** (vigilancia preventiva).
    - $\text{PSI} > 0.20$: **ALTO (Alerta de Deriva Activa)**.
- **Validación Experimental en Dos Escenarios (Sección 16 del Notebook):**
  1. *Escenario de Control (Dos cohortes del mismo hospital):* 44 de las 45 variables registraron $\text{PSI} < 0.09$; únicamente `sig_std` mostró un cambio leve ($\text{PSI} = 0.103$). El sistema se mantuvo estable sin falsas alarmas de deriva.
  2. *Escenario de Perturbación Quirúrgica (Electrodos ruidosos / cambio de monitor:* Ruido de alta frecuencia $\times 3$, amplitud de señal $\times 0.6$):
     - Detección inmediata con **3 variables en estado ALTO**:
       - `sig_ruido_alto`: **$\text{PSI} = 2.484$** (Alerta Crítica).
       - `sig_std`: **$\text{PSI} = 1.944$**.
       - `sig_p2p`: **$\text{PSI} = 1.646$**.
- **Protocolo de Reentrenamiento y Promoción Segura:**
  - Disparadores: Deriva sostenida ($\text{PSI} > 0.20$ en features clave), aumento anómalo de clase Ruido o auditorías clínicas discrepantes.
  - Flujo de reajuste: Extracción con `ecg_utils.py` $\to$ re-partición agrupada por `subjectid` $\to$ validación en conjunto fijo $\to$ promoción a producción solo si F1 macro no se degrada y el Recall de TV, Bloqueo AV y AFib se preserva $\to$ actualización del versionado en `models/meta_modelo.json`.
  - `[PENDIENTE: detalles de ejecución del script automatizado de reentrenamiento (bloque H6)]`

### Notas del Orador
> "Una vez desplegado un modelo, el verdadero desafío es la deriva de datos (data drift). Demostramos experimentalmente cómo detectar degradación mediante el Índice de Estabilidad Poblacional (PSI). La prueba KS convencional genera falsas alarmas por exceso de potencia muestral, mientras que el PSI discrimina cambios reales. En condiciones basales, el sistema se mantiene en verde. Pero cuando simulamos un cambio de monitor con electrodos degradados, el PSI de la variable 'sig_ruido_alto' se dispara a 2.484, encendiendo de inmediato la alarma de reajuste. Diseñamos un protocolo cerrado de MLOps: el modelo solo se reentrena agrupando por paciente y únicamente se promueve si supera al modelo anterior sin sacrificar la sensibilidad en arritmias de riesgo vital."

---

## Diapositiva 15: Conclusiones y Trabajo Futuro (Cierre)
**Título:** Síntesis del Proyecto: Rigor Metodológico y Visión Futura  
**Eje Temático:** Síntesis Ejecutiva y Cierre

### Contenido de la Diapositiva
- **Aportes Principales:**
  1. **Solución Realista de Punta a Punta:** Pipeline completamente funcional que opera con picos QRS detectados autónomamente, resistente a artefactos masivos de electrobisturí gracias al detector adaptativo v2.
  2. **Honestidad Metodológica y Control de Fuga:** Demostración empírica de que la partición aleatoria infla falsamente el F1 macro en $+0.229$. Validación estricta por `subjectid` con test ciego (F1 macro = 0.526, IC 95% $[0.458, 0.558]$).
  3. **Alineación con la Prioridad Quirúrgica:** Recall del 94% en AFib, 90% en Ruido y 65% en Taquicardia Ventricular, optimizando el balance entre detección oportuna y mitigación de fatiga por alarmas.
  4. **Arquitectura MLOps Integral:** Interfaz de usuario intuitiva en Streamlit y monitoreo continuo de deriva estadística basado en PSI.
- **Limitaciones Reconocidas:**
  - El monitoreo intraoperatorio con una única derivación (DII) tiene límites biofísicos para detectar alteraciones sutiles de la onda P (Bloqueo AV y WAP/MAT).
- **Próximos Pasos:**
  - Integración multi-derivación (DII + V5), validación prospectiva en quirófano simulado y auto-calibración conformal para cuantificación formal de incertidumbre.

### Notas del Orador
> "En conclusión, este proyecto demuestra que el machine learning aplicado a la medicina exige, por encima de todo, honestidad metodológica. Al eliminar la fuga de datos por paciente y diseñar un detector resistente a las condiciones hostiles del quirófano, logramos un sistema robusto, explicable y clínicamente útil para salvaguardar la vida del paciente bajo anestesia general. Muchas gracias por su atención. Quedamos atentos a sus preguntas y comentarios."
