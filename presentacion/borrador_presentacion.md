# Borrador de Presentación del Proyecto Final
## Sistema de Apoyo a la Decisión Clínica para Monitoreo Intraoperatorio de ECG
### Detección y Clasificación Multiclase de Arritmias Cardíacas en Anestesia General

---

> **Estructura de la presentación.** Diseño diapositiva a diapositiva para la sustentación, organizado según los 7 ejes de la **Sección 5 de `docs/Requeriments_FinalP.pdf`**. Cada diapositiva trae el contenido proyectado, las figuras del repositorio y las notas del orador.
>
> **Regla de este documento:** todo número sale de una salida visible de `model_development.ipynb` (versión v3, modelo final XGBoost). Si se cambia una cifra, verificarla antes en el notebook.

---

## Índice de Diapositivas

| Lámina | Eje Temático | Título de la Diapositiva | Figura / Recurso Clave |
| :---: | :--- | :--- | :--- |
| **01** | Portada | Monitoreo Inteligente de ECG Intraoperatorio | *Logo EAFIT / VitalDB* |
| **02** | 1. Problema y Contexto | Necesidad Clínica en Quirófano y Filosofía de Apoyo | Diagrama de flujo asistencial |
| **03** | 1. Problema y Contexto | VitalDB Arrhythmia Database y Hallazgos del EDA | `figures/eda_linea_tiempo.png` |
| **04** | 1. Problema y Contexto | Estructura No Supervisada y Justificación del Modelo | `figures/no_supervisado_pca.png` |
| **05** | 2. Metodología | Pipeline de Señal y Detección QRS v2 | `figures/eda_ecg_clinico.png` |
| **06** | 2. Metodología | 45 Características y Tratamiento de Faltantes | Tabla de familias de características |
| **07** | 3. Selección del Modelo | Protocolo por Paciente y Comparación de Familias | Tabla comparativa de modelos |
| **08** | 3. Selección del Modelo | Ablación y Experimentos del Bloque F | `figures/registro_correcciones.png` & `exp_resumen.png` |
| **09** | 4. Generalización | Validación Cruzada Externa y Test Fijo por Paciente | `figures/eval_final_confusion.png` |
| **10** | 4. Generalización | Demostración de Fuga de Datos y Curva de Aprendizaje | `figures/gen_curva_aprendizaje.png` |
| **11** | 5. Métricas e Interpretabilidad | Desempeño por Clase y Comparación con los Autores | `figures/eval_final_confusion.png` & `eval_pr.png` |
| **12** | 5. Métricas e Interpretabilidad | Explicabilidad y Coherencia Fisiológica | `figures/explic_permutacion.png` & `explic_shap_*.png` |
| **13** | 6. Despliegue | App en Streamlit y Monitor de Demostración | Capturas de `app/app.py` y `demo/monitor.html` |
| **14** | 7. Monitoreo y MLOps | Deriva (PSI/KS) y Reentrenamiento con `reentrenar.py` | Tablas de la sección 16 |
| **15** | Cierre | Conclusiones, Limitaciones y Trabajo Futuro | Hoja de ruta |

---

## Diapositiva 01: Portada
**Título:** Sistema de Apoyo a la Decisión Clínica para Monitoreo Intraoperatorio de ECG  
**Subtítulo:** Clasificación de 10 ritmos en quirófano con XGBoost y validación por paciente  
**Autores:** Equipo de Fundamentos de Aprendizaje Automático  
**Institución:** Universidad EAFIT, Fundamentos de Aprendizaje Automático  
**Fecha:** Noviembre 2026  

### Contenido Visual Sugerido
- Traza de ECG de 10 s con bandas de color por estado (verde: sin alerta, ámbar: revisar, rojo: alerta, gris: ruido), los mismos colores de la app.
- Esquema de bloques: Señal cruda (500 Hz) → Detección de picos R → 45 características → XGBoost → Alerta o revisión.

### Notas del Orador
> "Buenos días. Durante una cirugía, el anestesiólogo vigila al paciente en un monitor mientras atiende muchas otras tareas. Las arritmias intraoperatorias son frecuentes; la mayoría son benignas, pero algunas, como la taquicardia ventricular, exigen respuesta inmediata. Presentamos un sistema de apoyo que señala automáticamente los tramos de ECG con ritmo anormal, construido sobre la base abierta VitalDB Arrhythmia Database, con separación estricta por paciente y una ruta completa hasta producción."

---

## Diapositiva 02: El Problema y Contexto Clínico (Eje 1)
**Título:** Monitoreo Intraoperatorio de ECG: Apoyo, no Diagnóstico Autónomo  
**Eje Temático:** 1. El Problema y Contexto Clínico (Sección 3.1)

### Contenido de la Diapositiva
- **Entorno:**
  - Una derivación, **DII**, a **500 Hz**.
  - Pacientes bajo anestesia general: fármacos, estímulo quirúrgico y cambios autonómicos favorecen las arritmias.
- **Interferencia propia del quirófano:**
  - Electrobisturí (ruido de alta frecuencia), deriva de la línea base, movimiento y desconexión de electrodos.
- **Propósito del modelo:**
  - **Priorizar la atención** sobre tramos con ritmo anormal; no reemplaza al anestesiólogo ni diagnostica.
  - Usuarios: anestesiólogos, técnicos y enfermería de quirófano o recuperación.

### Notas del Orador
> "Delimitamos el alcance desde el inicio: el sistema no diagnostica ni reemplaza al anestesiólogo. Trabaja con una sola derivación, la II, a 500 Hz, y clasifica ventanas de 10 segundos. El ECG intraoperatorio tiene artefactos de electrobisturí, movimiento y deriva de línea base, por eso el modelo debe reconocer el ruido como una clase propia y no confundirlo con una arritmia."

---

## Diapositiva 03: VitalDB Arrhythmia Database y EDA (Eje 1)
**Título:** VitalDB: Caracterización del Dataset y Desbalance  
**Eje Temático:** 1. El Problema y Contexto Clínico (Sección 3.3)

### Contenido de la Diapositiva
- **Fuente y anotación:**
  - *VitalDB Arrhythmia Database* (PhysioNet, 2026): **482 cirugías** de **473 pacientes (`subjectid`)**, más de **660,000 latidos** anotados.
  - Anotación por un panel de **cinco anestesiólogos**; cada segmento lo etiquetaron al menos dos, hasta consenso. Concordancia: **kappa de Cohen 0.930 ± 0.130**.
  - **10 clases modeladas:** Normal, AFib, Ectopia ventricular, Ectopia auricular, Disfunción del nodo sinusal, SVTA, TV, Bloqueo AV, WAP/MAT y Ruido.
  - *Excluida:* Inclasificable (8 ventanas; zonas de transición sin ritmo definido).
- **Desbalance (ventanas de 10 s, 50,517 en total):**
  - Normal (31,441) y AFib (11,666) suman el **85 %**.
  - TV: 242 ventanas en 87 pacientes. SVTA: 481 en 106.
  - **Bloqueo AV: 409 ventanas de solo 9 pacientes. WAP/MAT: 867 de 23 pacientes.**
- **Duración de los episodios (`figures/eda_linea_tiempo.png`), medianas:**
  - Largos: AFib 123 s, WAP/MAT 84 s, Bloqueo AV 78 s, Disfunción del nodo 41 s, Normal 33 s.
  - Medios: ectopias, unos 14 s.
  - Muy cortos: **TV 2.7 s** (el 90 % dura menos de 6.2 s), SVTA 2.4 s, Ruido 5.4 s.
  - *Ventana:* 10 s con cobertura ≥80 % del ritmo, o ≥3 latidos para TV y SVTA, para no diluir las salvas.
- **Limpieza:**
  - Se eliminan las filas marcadoras (`Start`/`End` sin tipo de latido), que se contaban como latidos y creaban RR falsos.
  - Se excluye el **caso 1165**: anotación desalineada (desfase de −0.20 s y sensibilidad 0 con ambos detectores).

### Figura del Repositorio a Renderizar
- `figures/eda_linea_tiempo.png`

### Notas del Orador
> "El reto principal es el desbalance y la asimetría temporal. La fibrilación auricular dura minutos (mediana de 123 segundos), mientras una salva de taquicardia ventricular dura 2.7 segundos de mediana. Con una ventana de 60 segundos esa salva se diluiría; por eso usamos 10 segundos y una regla especial para salvas de 3 o más latidos. Además, Bloqueo AV viene de solo 9 pacientes y WAP/MAT de 23: esa escasez de pacientes marca el techo del modelo en esas clases."

---

## Diapositiva 04: Estructura No Supervisada (Eje 1)
**Título:** ¿Los ritmos forman grupos naturales? PCA y K-Means  
**Eje Temático:** 1. El Problema y Contexto Clínico (Sección 3.3)

### Contenido de la Diapositiva
- **Método (`figures/no_supervisado_pca.png`):** escalado robusto (`RobustScaler`, recorte a ±5), PCA y K-Means con k = 10.
- **Varianza explicada:** las dos primeras componentes, **43 %**; hacen falta 10 para el **81 %**.
- **Agrupación débil:** ARI 0.194 y silueta 0.127.
- **Hallazgos:**
  - **Ruido casi aislado:** el cluster 2 reúne **347 de las 419** ventanas de ruido.
  - **Latidos ventriculares juntos:** Ectopia ventricular y TV comparten clusters (3 y 9).
  - **AFib con SVTA** (cluster 4), y Bloqueo AV, WAP/MAT y Disfunción del nodo mezclados con Normal y AFib.
- **Conclusión:** las clases no son grupos naturales completos; se justifica el aprendizaje supervisado con características de morfología.

### Notas del Orador
> "Antes del modelo supervisado preguntamos si los datos se separan solos. El ruido queda casi aislado, y los latidos ventriculares se agrupan por su forma ancha. Pero Bloqueo AV, WAP/MAT y Disfunción del nodo se mezclan con Normal y AFib. Esas mismas confusiones reaparecen después en el modelo supervisado: el análisis no supervisado las anticipó."

---

## Diapositiva 05: Preprocesamiento y Detector de Picos R (Eje 2)
**Título:** Del ECG Crudo a los Picos R Detectados (Detector v2)  
**Eje Temático:** 2. Metodología y Preprocesamiento (Secciones 3.2 y 3.4)

### Contenido de la Diapositiva
- **Sin picos anotados en inferencia:** `FUENTE_PICOS = "detectados"`. Las anotaciones de los médicos solo se usan para etiquetar.
- **Filtrado:** Butterworth pasabanda 0.5–40 Hz aplicado hacia adelante y atrás (`sosfiltfilt`), sin desplazar los picos.
- **Detector v2 (tipo Pan-Tompkins):**
  1. Energía del QRS: banda 5–15 Hz, derivada al cuadrado y media móvil de unos 120 ms.
  2. Umbral local: 30 % del máximo de la energía en 2 s alrededor, con un **piso del 20 % de la mediana** de las alturas de los máximos, que no se deja arrastrar por artefactos gigantes.
  3. Periodo refractario de 200 ms y ajuste al máximo positivo cercano.
- **Validación contra los picos anotados (482 casos, 5 min limpios por caso):**
  - Mediana de sensibilidad **1.000** y de precisión **0.989**.
  - Donde fallaba v1: sensibilidad en **TV de 0.50 a 0.75** y en Ectopia ventricular de 0.71 a 0.87.
- **Ventanas:** **50,517** ventanas válidas de 10 s de 472 pacientes.

### Figura del Repositorio a Renderizar
- `figures/eda_ecg_clinico.png` (ECG sobre papel milimetrado, 25 mm/s y 10 mm/mV).

### Notas del Orador
> "Un error común es alimentar el modelo con los picos R que marcaron los médicos; en un monitor real esos picos no existen. Nuestro pipeline los detecta. El detector v2 usa un umbral local con un piso basado en la mediana, para que un artefacto enorme no lo deje ciego. Contra las anotaciones, la mediana de sensibilidad es 1.000 y la de precisión 0.989, y en taquicardia ventricular la sensibilidad subió de 0.50 a 0.75 frente a la primera versión."

---

## Diapositiva 06: Características y Manejo de Faltantes (Eje 2)
**Título:** 45 Características: Ritmo, Calidad de Señal, Morfología e Indicador de Faltantes  
**Eje Temático:** 2. Metodología y Preprocesamiento (Sección 3.4)

### Contenido de la Diapositiva
- **22 de ritmo (RR):** `fc_media`, `fc_pico`, `rr_mediana`, `sdnn`, `rmssd`, `pnn50`, `pnn20`, `cv_rr`, Poincaré (`sd1`, `sd2`, `sd1_sd2`), `tpr`, entre otras. Se eliminó la entropía muestral.
- **9 de calidad de señal:** `sig_sqi_qrs` (fracción de energía en la banda del QRS, 5–15 Hz), `sig_ruido_alto` (fracción en 40–100 Hz: electrobisturí, EMG), `sig_deriva`, `sig_frac_plana` (tramos planos ≥100 ms) y `sig_frac_repetidas`.
- **13 de morfología:** `qrs_ancho_med`, `qrs_ancho_max`, `corr_plantilla_med`, `corr_plantilla_min`, `frac_latidos_atipicos`, `p_presente_frac`, `pr_intervalo`, `p_corr_med`, `p_sin_qrs`, `t_amp_rel`, entre otras.
- **1 indicador:** `sin_rr_valido`, activo cuando la ventana tiene menos de 4 picos y no se puede calcular el ritmo.
- **Faltantes no aleatorios:** faltan latidos por ruido, desconexión o pausas. La mediana le inventaría a esas ventanas un ritmo "típico"; se imputa con **−1** (fuera de rango) junto con el indicador.
- **Escalado:** `StandardScaler` solo para regresión logística y SVM.

### Notas del Orador
> "Construimos 45 características con sentido clínico. Un punto clave fue el tratamiento de faltantes: cuando una ventana tiene menos de 4 latidos no se puede medir el ritmo, y eso no ocurre al azar. Imputar con la mediana haría parecer normal una ventana de ruido o una pausa. Por eso usamos un valor fijo fuera de rango y una variable indicadora. En la ablación, volver a la mediana baja el F1 de 0.539 a 0.531."

---

## Diapositiva 07: Selección de Modelos (Eje 3)
**Título:** Partición por Paciente y Comparación de Familias  
**Eje Temático:** 3. Selección del Modelo y de sus Parámetros (Sección 3.5)

### Contenido de la Diapositiva
- **Protocolo:**
  - `StratifiedGroupKFold` por **`subjectid`**, 5 pliegues fijos guardados en `data/pliegues_pacientes.csv`. El pliegue 0 es el test fijo (93 pacientes).
  - **Tope de 5,000 ventanas por clase solo al entrenar.** Validación y test conservan la distribución real.
- **Comparación con hiperparámetros por defecto (validación interna por paciente):**

| Modelo | Configuración | F1 macro val | F1 train | Exactitud bal. val | Exactitud val | Tiempo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Línea base trivial | Siempre Normal | 0.077 | | 0.100 | 0.62 | |
| Línea base aleatoria | `DummyClassifier(strategy="stratified")` | 0.091 ± 0.004 | | | | |
| Regresión logística | L2, `class_weight="balanced"`, StandardScaler | 0.426 ± 0.025 | 0.653 | 0.598 | 0.683 | 9 s |
| SVM RBF | `class_weight="balanced"`, StandardScaler | 0.460 ± 0.040 | 0.803 | 0.604 | 0.753 | 298 s |
| Random Forest | 300 árboles, `class_weight="balanced_subsample"` | 0.509 ± 0.033 | 1.000 | 0.509 | 0.879 | 16 s |
| HistGradientBoosting | `class_weight="balanced"` | **0.528 ± 0.017** | 0.983 | 0.585 | 0.866 | 41 s |

- **Búsqueda aleatoria por paciente:** Random Forest sube a 0.528 y HistGradientBoosting a **0.534**, que queda como modelo base (v2).
- XGBoost, LightGBM y SVM afinados se prueban después como experimento F9 (diapositiva 08).

### Notas del Orador
> "La regla fue que ningún paciente de entrenamiento aparezca en validación ni en test. Con hiperparámetros por defecto, HistGradientBoosting fue el mejor con 0.528, frente a 0.077 de predecir siempre Normal y 0.091 de predecir al azar. Random Forest memoriza el entrenamiento, con F1 de 1.000 en train y 0.509 en validación. Tras afinar, HistGradientBoosting llegó a 0.534 y quedó como base para los experimentos."

---

## Diapositiva 08: Ablación y Experimentos (Eje 3)
**Título:** Qué Aporta Cada Decisión y Qué se Probó para Mejorar  
**Eje Temático:** 3. Selección del Modelo y de sus Parámetros (Sección 3.5)

### Contenido de la Diapositiva
- **Ablación (validación interna, `figures/registro_correcciones.png`):**
  - Pipeline v2 completo: **0.539**.
  - **Sin morfología: 0.458 (−0.081).** Es la mayor mejora del proyecto.
  - Detector de v1: 0.523. Imputación con mediana: 0.531. Agrupando por cirugía: 0.532.
  - Picos anotados: 0.586, pero es un atajo: Ruido sube a 0.994 porque las ventanas de ruido no tienen latidos anotados.
- **Regla de adopción, fijada antes de ver resultados** (validación cruzada externa, 5 pliegues):
  1. ΔF1 macro ≥ 0.005;
  2. mejora en al menos 4 de 5 pliegues;
  3. no baja más de 0.02 el recall de TV, Bloqueo AV ni AFib.
- **18 variantes probadas (tabla G2, `models/registro_experimentos.csv`), resumen:**

| Variante | F1 macro | ΔF1 | Pliegues que mejoran | Recall TV | Recall Bloqueo AV | Decisión |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Base v2 (HistGradientBoosting) | 0.534 | | | 0.653 | 0.112 | Referencia |
| **F9 XGBoost afinado** | **0.545** | **+0.011** | **5** | **0.653** | **0.112** | **Adoptado (v3)** |
| F3b Aumento sobre la señal | 0.559 | +0.025 | 4 | 0.583 | 0.093 | Baja recall de TV |
| F11 XGBoost + aumento de señal | 0.554 | +0.020 | 5 | 0.607 | 0.093 | Baja recall de TV y Bloqueo AV |
| F3a SMOTE | 0.552 | +0.018 | 5 | 0.649 | 0.073 | Baja recall de Bloqueo AV |
| F9 LightGBM afinado | 0.552 | +0.018 | 4 | 0.599 | 0.071 | Baja recall de TV y Bloqueo AV |
| F8b Pesos por clase para F1 | 0.552 | +0.018 | 4 | 0.483 | 0.110 | Baja recall de TV |
| F8a Umbrales para recall | 0.525 | −0.009 | 1 | 0.640 | 0.267 | Bloqueo AV sube, pero su precisión cae y los umbrales son inestables |
| F10 Sin tope por clase | 0.537 | +0.003 | 4 | 0.640 | 0.108 | No alcanza el mínimo |
| F5 Sin redundantes | 0.535 | +0.001 | 2 | 0.657 | 0.103 | Sin mejora |
| F1 Ventanas deslizantes | 0.531 | −0.003 | 1 | 0.727 | 0.100 | Sin mejora; el paso de 2 s se usa en la demo |
| F4 Ventana de 20 s | 0.576 | +0.043 | 5 | 0.656 | 0.184 | Otro problema (más latencia); trabajo futuro |
| F9 SVM afinado | 0.487 | −0.047 | 0 | 0.554 | 0.105 | Peor en todos los pliegues |

- **XGBoost (v3):** 432 árboles, `max_depth=8`, `learning_rate≈0.038`, `subsample≈0.77`, `colsample_bytree≈0.86`, `min_child_weight=7`, `reg_lambda≈1.6`, con pesos balanceados por clase (`XGBBalanceado`).

### Figuras del Repositorio a Renderizar
- `figures/registro_correcciones.png` y `figures/exp_resumen.png`.

### Notas del Orador
> "La ablación muestra que lo que más aporta es el conocimiento del dominio: sin las 13 características de morfología el F1 cae de 0.539 a 0.458. Después probamos 18 variantes con una regla fijada de antemano. Varias subían el F1, como el aumento de datos o LightGBM, pero lo hacían perdiendo taquicardias ventriculares o bloqueos. Solo XGBoost cumplió la regla: +0.011, mejora en los 5 pliegues y ningún recall peligroso baja. Es una mejora real pero modesta. Como comparamos 18 variantes sobre los mismos pliegues, la mejor tiende a ser algo optimista."

---

## Diapositiva 09: Generalización (Eje 4)
**Título:** Validación Cruzada Externa y Test Fijo por Paciente  
**Eje Temático:** 4. Generalización del Modelo (Sección 3.6)

### Contenido de la Diapositiva
- **Validación cruzada externa** (cada pliegue hace de test una vez, 50,517 ventanas):

| | Pliegue 0 | Pliegue 1 | Pliegue 2 | Pliegue 3 | Pliegue 4 | Media ± desv. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| v2 (HistGradientBoosting) | 0.526 | 0.519 | 0.544 | 0.562 | 0.519 | 0.534 ± 0.019 |
| **v3 (XGBoost)** | **0.541** | **0.531** | **0.555** | **0.564** | **0.535** | **0.545 ± 0.014** |

- **Test fijo (pliegue 0): 93 pacientes nuevos, 9,950 ventanas, distribución real:**
  - **F1 macro 0.541** (v2: 0.526).
  - **IC95 % por bootstrap de pacientes (300 remuestreos): [0.474, 0.573].**
  - Exactitud balanceada 0.611, MCC 0.734, exactitud 0.842.
- **Advertencia:** el test fijo ya se usó para evaluar v2 y forma parte de la validación externa con la que se eligió v3; para v3 no es un test virgen y la estimación puede ser levemente optimista.

### Figura del Repositorio a Renderizar
- `figures/eval_final_confusion.png` (v3, validación externa, recall por clase).

### Notas del Orador
> "Evaluamos siempre con pacientes que el modelo no vio. En la validación cruzada externa, el modelo final logra 0.545 con una desviación de 0.014, y mejora a la base en los 5 pliegues. En el test fijo de 93 pacientes nuevos obtiene 0.541, con un intervalo de confianza del 95 % entre 0.474 y 0.573. Somos explícitos con una limitación: ese test ya se había usado, así que para la versión final no es completamente virgen."

---

## Diapositiva 10: Fuga de Datos y Curva de Aprendizaje (Eje 4)
**Título:** Por qué la Partición por Paciente es Indispensable  
**Eje Temático:** 4. Generalización del Modelo (Sección 3.6)

### Contenido de la Diapositiva
- **Validación aleatoria por ventana vs. agrupada por paciente (modelo v3):**

| Partición | F1 macro | Diferencia |
| :--- | :---: | :---: |
| Aleatoria por ventana (incorrecta) | 0.792 ± 0.010 | **+0.258 inflado** |
| Agrupada por `subjectid` (correcta) | 0.534 ± 0.026 | Referencia |

- **Cirugías del mismo paciente:** 8 pacientes tienen varias cirugías; por eso se agrupa por `subjectid` y no por `case_id`.
- **Etiquetas barajadas:** el F1 cae a **0.096 ± 0.001**, el nivel del azar para 10 clases. No hay fuga ni estructura espuria.
- **Brecha:** F1 en train 0.981 frente a 0.535 en validación interna y 0.541 en test. El modelo memoriza el entrenamiento, pero validación y test coinciden.
- **Curva de aprendizaje (`figures/gen_curva_aprendizaje.png`):** la validación sube de 0.50 (unas 3,000 ventanas) a 0.53 (14,800). Más ventanas de los mismos pacientes ayudan poco; el límite son los pacientes de las clases raras.

### Notas del Orador
> "Si dividiéramos las ventanas al azar, el F1 sería 0.792: el modelo reconocería al paciente, no la arritmia, porque vería ventanas del mismo paciente en entrenamiento y en prueba. Separando por paciente, el valor real es 0.534. Con etiquetas barajadas cae a 0.096, el azar, lo que confirma que no hay fuga. Y la curva de aprendizaje muestra que más ventanas de los mismos pacientes casi no ayudan: lo que falta son más pacientes de las clases raras."

---

## Diapositiva 11: Desempeño por Clase (Eje 5)
**Título:** Desempeño por Clase y Comparación con los Autores  
**Eje Temático:** 5. Métricas de Desempeño e Interpretabilidad (Sección 3.7)

### Contenido de la Diapositiva
- **Por qué no la exactitud:** predecir siempre Normal acertaría el 62 % de las ventanas sin alertar nunca. Se priorizan F1 macro, exactitud balanceada, MCC, PR-AUC y el recall de las arritmias peligrosas.
- **Modelo v3 por clase (validación externa, 50,517 ventanas; y test fijo):**

| Clase | F1 CV | Recall CV | Precisión CV | F1 test |
| :--- | :---: | :---: | :---: | :---: |
| AFib | 0.951 | 0.951 | 0.952 | 0.941 |
| Normal | 0.922 | 0.896 | 0.949 | 0.921 |
| Ectopia ventricular | 0.746 | 0.858 | 0.659 | 0.847 |
| Ruido | 0.666 | 0.902 | 0.528 | 0.722 |
| Ectopia auricular | 0.654 | 0.743 | 0.584 | 0.617 |
| SVTA | 0.468 | 0.590 | 0.388 | 0.632 |
| **TV** | **0.404** | **0.653** | **0.292** | **0.440** |
| Disfunción del nodo | 0.381 | 0.411 | 0.356 | 0.269 |
| Bloqueo AV | 0.121 | 0.112 | 0.131 | 0.000 |
| WAP/MAT | 0.050 | 0.045 | 0.057 | 0.025 |

- **TV:** detecta dos de cada tres episodios, pero con más de 2 falsas alarmas por acierto (precisión 0.29).
- **Bloqueo AV en test = 0:** el test tiene un solo paciente con Bloqueo AV (117 ventanas). Entre pliegues, el F1 de v2 en esta clase va de 0.00 a 0.34.
- **Comparación con los autores (G9).** Se agrupan las predicciones en normal y anormal (sin Ruido, 50,098 ventanas). No es directamente comparable: ellos resuelven un problema binario con una CNN 1D sobre 5 s.

| | Autores (CNN 1D, 5 s, binario) | Este proyecto (v3, 10 s, 10 clases agrupadas) |
| :--- | :---: | :---: |
| F1 | 0.910 | 0.878 |
| Sensibilidad | 0.848 | 0.919 |
| Especificidad | 0.988 | 0.896 |
| AUROC | 0.990 | 0.968 |

### Figuras del Repositorio a Renderizar
- `figures/eval_final_confusion.png` y `figures/eval_pr.png`.

### Notas del Orador
> "AFib y Normal superan 0.92 de F1, y Ectopia ventricular llega a 0.75. El ruido se reconoce con recall de 0.90. En taquicardia ventricular el recall es 0.65 con precisión 0.29: detecta dos de cada tres episodios, pero con falsas alarmas que habría que ajustar con el equipo clínico. Bloqueo AV y WAP/MAT no se resuelven: vienen de 9 y 23 pacientes. Al agrupar en normal y anormal, el modelo logra sensibilidad 0.919 y AUROC 0.968; la red de los autores es mejor en conjunto, con menos falsas alarmas. Detectar que algo anda mal es mucho más fácil que decir exactamente qué es."

---

## Diapositiva 12: Explicabilidad (Eje 5)
**Título:** Importancia por Permutación y SHAP por Clase  
**Eje Temático:** 5. Métricas de Desempeño e Interpretabilidad (Sección 3.7)

### Contenido de la Diapositiva
- **Permutación (`figures/explic_permutacion.png`):** la más importante es **`p_corr_med`** (parecido de las ondas P entre sí): barajarla baja el F1 en 0.13. Siguen `fc_pico` (0.05), `pnn50` (0.03), `pnn20`, `sig_deriva`, `sd1_sd2`, `qrs_ancho_max`, `rr_skew` y `pr_intervalo`. `n_latidos`, primera en v0 por el atajo del ruido, ya no aporta.
- **SHAP por clase (G7, `figures/explic_shap_<clase>.png`):**
  - **AFib:** `p_corr_med` bajo, `p_presente_frac` bajo, `pnn20` y `rr_mad` altos: sin onda P y con RR irregular.
  - **Normal:** RR regular (`rr_iqr`, `sdnn` bajos) con ondas P presentes y parecidas.
  - **Ruido:** `corr_plantilla_med` baja, `sig_deriva` y `sig_ruido_alto` altos, `sig_sqi_qrs` bajo.
  - **TV:** `corr_plantilla_min` baja, `frac_latidos_atipicos` y `qrs_ancho_max` altos, `fc_pico` alto: latidos anchos, distintos y rápidos.
  - **SVTA:** casi solo `fc_pico` alto, con QRS parecidos entre sí (`frac_latidos_atipicos` baja), lo que la separa de la TV.
  - **Ectopia ventricular:** `qrs_ancho_max` alto y patrón corto-largo, sin frecuencia alta sostenida.
  - **Ectopia auricular:** latidos prematuros con onda P conservada.
  - **Disfunción del nodo:** RR largos; también describe una bradicardia normal, por eso se confunde con Normal.
  - **Bloqueo AV:** `pr_intervalo` alto (correcto), pero con efecto disperso e influencia de variables de señal: con 9 pacientes puede estar aprendiendo rasgos de esos pacientes.
  - **WAP/MAT:** la decide sobre todo el QRS estrecho, que no define a esta arritmia; `p_corr_med` pesa poco. Las características no ven lo que la distingue.

### Figuras del Repositorio a Renderizar
- `figures/explic_permutacion.png`, `figures/explic_shap_global.png` y 2 o 3 de `figures/explic_shap_<clase>.png` (sugerido: AFib, TV y WAP/MAT).

### Notas del Orador
> "Revisamos qué usa el modelo para decidir. En las clases bien resueltas la lógica es fisiológica: AFib por ausencia de onda P y RR irregular; TV por latidos anchos, distintos y rápidos; Ruido por baja calidad de señal. Las clases que fallan también se explican: en WAP/MAT el modelo se apoya en el QRS estrecho, que no es lo que define esa arritmia, y en Bloqueo AV, con solo 9 pacientes, puede estar aprendiendo rasgos de los pacientes. El límite está en la onda P con una sola derivación."

---

## Diapositiva 13: Despliegue e Interfaz (Eje 6)
**Título:** App en Streamlit y Monitor de Demostración  
**Eje Temático:** 6. Despliegue e Interfaz para el Usuario (Sección 3.8)

### Contenido de la Diapositiva
- **Modelos guardados:**
  - `models/modelo_arritmias.joblib`: XGBoost reajustado con 17,410 ventanas de 472 pacientes; lo usa la app con archivos propios.
  - `models/modelo_evaluacion.joblib`: copia entrenada **sin** los pacientes del test; analiza los ejemplos y es la referencia de `reentrenar.py`.
  - Mismo preprocesamiento en entrenamiento y producción: `src/ecg_utils.py`.
- **App (`app/app.py`):**
  - Elegir un ejemplo incluido o subir un CSV/TXT (una columna en mV); frecuencia de muestreo configurable, con conversión a 500 Hz; control de confianza mínima (0.60 por defecto).
  - **Primero la conclusión:** tramos analizados, con alerta, para revisar y con ruido, y un mensaje claro ("Alerta: ... a partir del segundo ...").
  - Señal con fondo por estado: **rojo alerta** (AFib, SVTA, TV, Bloqueo AV), **ámbar revisar** (otra clase no normal o confianza baja), **verde sin alerta**, **gris ruido**.
  - Tabla por tramo de 10 s (inicio, estado, ritmo en lenguaje claro, frecuencia, confianza), descargable en CSV.
  - Avisos de formato y aclaración de que es una herramienta de apoyo.
- **Ejemplos (`app/ejemplos/`):** cinco tramos de 1 minuto de pacientes del test (31 predicciones cada uno, una cada 2 s en el monitor):
  - Cuatro **ilustrativos**, elegidos como el minuto que el modelo clasifica mejor; no representan el desempeño promedio:
    - AFib (caso 1738): 31 de 31 AFib.
    - Latidos ventriculares prematuros (caso 884): 31 de 31 aciertos.
    - TV (caso 2016, salvas en el 30 % del minuto): 17 TV, 10 Normal y 4 Ectopia ventricular.
    - SVTA (caso 4158, 34 % del minuto): 28 SVTA, 2 AFib y 1 Normal.
  - Uno de **clase difícil**, elegido solo por su etiqueta: Bloqueo AV (caso 4364, todo el minuto): nunca lo reconoce; 20 Normal y 11 Disfunción del nodo.
- **Monitor (`demo/monitor.html`):** página autónoma con dos monitores sincronizados: el convencional, con un pitido en cada latido detectado, y el que colorea según la predicción (un color por clase), junto a la etiqueta de los médicos. Trae 10 s de señal previa, así que predice desde el segundo 0. Reproduce predicciones ya calculadas con el modelo real.
- **Integración real (no implementada):** el monitor transmitiría por protocolo del fabricante o HL7 (VitalDB grabó con Vital Recorder); se guardarían los últimos 10 s y se repetiría el mismo análisis.

### Notas del Orador
> "La app está pensada para personal no técnico. Primero muestra la conclusión, cuántos tramos tienen alerta o requieren revisión, y luego la señal coloreada y una tabla descargable. Los ejemplos son de pacientes del test y los analiza una copia del modelo que nunca vio a esos pacientes. Cuatro los elegimos para ilustrar cómo se ve el sistema cuando acierta, y lo decimos así; el quinto, Bloqueo AV, lo elegimos sin mirar al modelo, y ahí se ve su límite: nunca reconoce el bloqueo. El monitor muestra además el retardo real de una ventana de 10 segundos: la alerta de TV se mantiene mientras la salva sigue dentro de la ventana."

---

## Diapositiva 14: Monitoreo de Deriva y Reentrenamiento (Eje 7)
**Título:** Deriva de Datos (PSI / KS) y Reentrenamiento Seguro  
**Eje Temático:** 7. Estrategia de Monitoreo y Reajuste del Modelo (Sección 3.9)

### Contenido de la Diapositiva
- **Qué se vigila:**
  - Entradas (sin etiquetas): PSI de cada característica frente a `models/referencia_muestra.csv`.
  - Predicciones: fracción por clase, alertas por cirugía y confianza media; más ruido suele indicar problemas de equipo.
  - Desempeño real: un anestesiólogo revisa una muestra mensual; alarma si el F1 cae > 0.03 o el recall de TV, Bloqueo AV o AFib > 0.02.
- **PSI vs. KS:** con miles de ventanas el p-valor del KS sale ínfimo aunque el cambio sea pequeño; se decide con el PSI (< 0.1 estable, 0.1–0.2 moderado, > 0.2 alto).
- **Dos escenarios (sección 16.1):**
  1. Dos mitades de pacientes del mismo hospital: ninguna variable en ALTO; solo `sig_std` moderada (PSI 0.103).
  2. Equipo simulado más ruidoso y de menor ganancia (`sig_ruido_alto` ×3, `sig_std` y `sig_p2p` ×0.6): exactamente esas 3 pasan a ALTO (PSI 2.484, 1.944 y 1.646) y el resto no cambia.
- **Reentrenamiento (`reentrenar.py`, sección 16.2):**
  - Construye las mismas 50,517 ventanas que el notebook (etiquetas y características idénticas).
  - Entrena un candidato y lo compara con el actual en la validación fija (pliegue 0); lo promueve solo si el F1 y el recall de TV, Bloqueo AV y AFib no bajan más de 0.01. Guarda versiones en `models/versiones/`.
  - Ensayo con 60 pacientes buenos: F1 0.541 → 0.545, recall TV 0.60 → 0.68 → **PROMOVER**.
  - Ensayo con esos pacientes mal etiquetados: F1 0.541 → 0.531, recall AFib 0.942 → 0.929 → **RECHAZAR**.
- **Puesta en producción:** modo sombra durante algunas cirugías antes de reemplazar al modelo; volver atrás es recuperar la versión anterior.

### Notas del Orador
> "Un modelo en producción se degrada cuando cambian monitores, electrodos o pacientes. Vigilamos las entradas con el PSI, porque el KS da falsas alarmas con tantos datos. Entre dos grupos de pacientes del mismo hospital todo queda estable; al simular un equipo más ruidoso, las tres variables alteradas se disparan y el resto no se mueve. Para reentrenar, el script compara al candidato con el modelo actual sobre pacientes fijos que ninguno vio: con datos buenos lo promueve y con datos mal etiquetados lo rechaza."

---

## Diapositiva 15: Conclusiones y Trabajo Futuro (Cierre)
**Título:** Síntesis: Rigor Metodológico y Próximos Pasos  
**Eje Temático:** Cierre

### Contenido de la Diapositiva
- **Aportes:**
  1. **Del ECG crudo a la predicción**, con picos detectados por un algoritmo propio; las anotaciones solo etiquetan.
  2. **Medición honesta:** partición por paciente, pliegues fijos y distribución real. F1 macro **0.545 ± 0.014** (validación externa) y **0.541** en test (IC95 % [0.474, 0.573]), frente a 0.077 de la línea base. La validación aleatoria lo habría inflado a 0.792.
  3. **El dominio pesa más que el algoritmo:** la morfología aporta +0.08; XGBoost, +0.011.
  4. **Una regla que protege las arritmias peligrosas** evitó adoptar variantes con más F1 pero menos TV detectadas.
  5. **Ruta a producción:** app, monitor de demostración, monitoreo de deriva y reentrenamiento con salvaguarda.
- **Limitaciones:**
  - Segmentos preseleccionados por un modelo de IA: las métricas no estiman una cirugía completa.
  - Un solo hospital; Bloqueo AV y WAP/MAT sin resolver; precisión de TV 0.29; retardo de la ventana de 10 s; v3 elegida sobre los mismos pliegues; sin integración en vivo; no es un dispositivo médico validado.
- **Trabajo futuro:** más pacientes de las clases raras, CNN 1D sobre la señal cruda, combinar ventanas corta y larga (la de 20 s dio 0.576), mejores características de onda P, umbrales ajustados con el equipo clínico y validación en otra base.

### Notas del Orador
> "En resumen: el número final es más bajo que el de nuestra primera versión, pero es honesto. Cada corrección quitó un atajo, y en las mismas condiciones cada versión mejoró a la anterior. La mayor ganancia vino de entender el ECG, no de cambiar de algoritmo, y la regla de adopción evitó elegir modelos que detectaban menos taquicardias ventriculares. El siguiente paso es conseguir más pacientes de las clases raras y probar redes sobre la señal cruda. Muchas gracias."
