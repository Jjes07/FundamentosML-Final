# Pendientes del notebook (`model_development.ipynb`)

Lista de cambios acordados por el equipo. Cada punto dice **qué** hacer, **por qué** y **cómo** se planeó.

## Reglas generales (aplican a todo)

- **Antes y después.** Cada mejora se presenta con el rendimiento antes y después del cambio, con la misma validación por paciente. La versión anterior no se borra: queda como línea base o como experimento descartado, con la razón de la decisión (normalmente, que el modelo generaliza mejor o peor en pacientes nuevos).
- **Registro de experimentos.** Todo lo probado va a una tabla resumen (ver G2) para que nadie repita pruebas ya hechas.
- **Toda interpretación se apoya en una salida visible del notebook.** Si un texto afirma un número o un hecho, debe haber una celda que lo muestre. Nada de afirmaciones que solo se verificaron por fuera.
- **Comentarios en el código.** Los bloques no evidentes llevan comentarios cortos, en infinitivo o tercera persona.
- Las salidas cambiarán al aplicar estos cambios. Al final hay que **releer y reescribir todas las interpretaciones** con las salidas nuevas.

---

## A. Correcciones de datos y metodología (hacer primero; afectan todos los resultados)

**A1. Agrupar por paciente real (`subjectid`), no por cirugía (`case_id`).**
- Por qué: hay 482 casos (cirugías) de 473 pacientes; 8 pacientes tienen 2 o 3 cirugías (subjectid 264, 485, 2242, 2369, 4833, 5105, 5666, 5820). Si un paciente queda en train con una cirugía y en test con otra, hay fuga de datos. En una reproducción aproximada de la partición, 2 pacientes quedaron en ambos lados (5820 con Bloqueo AV y 485 con Ectopia auricular). Además, los 10 casos de Bloqueo AV son de solo 9 pacientes.
- Cómo: usar `subjectid` como `groups` en `StratifiedGroupKFold`, en la búsqueda de hiperparámetros, en el bootstrap y en la curva de aprendizaje. Cambiar en todos los textos "482 pacientes" por "482 casos (473 pacientes)". Reportar el resultado antes y después para medir cuánto inflaba la fuga.

**A2. Eliminar las filas marcadoras que siguen en la tabla de anotaciones.**
- Por qué: quedan unas 3,808 filas sin `beat_type` dentro de ritmos que no son Ruido (casi todas con `Start`/`End` en `bad_signal_quality_label`). No son latidos, pero hoy se cuentan como tales. Caen a 0.14 a 0.5 s de un latido real, así que crean intervalos RR falsos y cortos que inflan la irregularidad, `frac_rr_corto` y `fc_pico`.
- Cómo: en la sección 5, quitar toda fila con `beat_type` vacío excepto las de Ruido (o directamente las que tengan marca en `bad_signal_quality_label`). Antes de usarlas para delimitar los tramos de mala calidad, guardar sus tiempos en una tabla aparte. Mostrar con una salida cuántas se quitan y de qué ritmos.

**A3. Calcular las características con picos detectados, no con los anotados.**
- Por qué: hoy los RR salen de los picos marcados por los médicos. En las ventanas de ruido no hay picos anotados, así que `n_latidos` ≈ 0 y el modelo aprende "sin latidos = ruido" (el SHAP de Ruido depende casi solo de `n_latidos`). En un monitor real no hay anotaciones: los picos los da el detector, que sí encontrará picos falsos en el ruido. Además, solo así se puede afirmar "del ECG crudo a la predicción".
- Cómo: `FUENTE_PICOS = "detectados"`. Las anotaciones siguen dando solo la etiqueta. El detector tiene una mediana de sensibilidad de 0.996 y de precisión de 0.981 (sección 7). Hipótesis a comprobar: Ruido seguirá alto, pero ahora reconocido por las características de calidad de la señal.

**A4. Excluir casos con anotación desalineada.**
- Por qué: en el caso 1165, el detector encuentra los latidos con un corrimiento constante respecto a las anotaciones, así que las etiquetas no coinciden con la señal.
- Cómo: con la tabla de la sección 7, excluir los casos con |desfase mediano| > 0.1 s y documentarlo. Ampliar la revisión a los 482 casos (hoy solo se revisan 60).

**A5. Volver a ejecutar la curva de aprendizaje.**
- Por qué: `datos` está ordenado por clase y `learning_curve` tomaba subconjuntos sin barajar; los primeros tamaños tenían pocas clases. Ya se agregó `shuffle=True` al código, pero falta ejecutarlo.

**A6. Test con la distribución real de clases.**
- Por qué: el tope `MAX_POR_CLASE` (5,000 ventanas para Normal y AFib) se aplicó antes de partir train y test. El test tiene entonces una proporción artificial, y la precisión de las arritmias sale optimista.
- Cómo: partir primero en train y test por paciente. Aplicar el tope (si se usa) solo al train. Probar también sin tope (ver F10).

**A7. Manejo de nulos.**
- Por qué: `sampen` falta en el 62 % de las ventanas (en 10 s no hay latidos suficientes para calcularla). Las variables de RR son nulas cuando la ventana tiene menos de 4 latidos válidos, casi siempre ventanas de ruido. Hoy se rellenan con la **mediana**, y eso es engañoso: a una ventana de ruido se le asigna la frecuencia "típica" (unos 75 lpm) y parece ritmo normal. El nulo es información (falta *porque* no hay latidos), no un dato perdido al azar.
- Cómo: eliminar `sampen`. Para las variables de RR: agregar una variable indicadora `sin_rr_valido` (1 si no se pudo calcular) e imputar con un valor fijo fuera de rango (p. ej. −1) o dejar NaN para HistGradientBoosting, que los maneja de forma nativa. No usar interpolación: las ventanas son independientes y no hay una serie que interpolar. Comparar mediana frente a indicador.

**A8. Revisar el mínimo de latidos por ventana (`MIN_LATIDOS = 6`).**
- Por qué: descarta 60 ventanas; 20 son de Bloqueo AV y 19 de Disfunción del nodo, justamente las bradiarritmias con pausas largas, que son de las peligrosas. 6 latidos en 10 s equivalen a 36 lpm.
- Cómo: probar con un mínimo de 3 (mínimo para calcular 2 intervalos) o conservar esas ventanas, dejando que el número de latidos sea una pista. Comparar antes y después.

---

## B. Sección 5: limpieza de etiquetas

**B1. Conservar todas las clases menos Inclasificable.** Bloqueo AV se queda y se reforzará con aumento de datos (F3). Quitar los textos que proponen fusionar o descartar clases raras. Inclasificable se elimina porque son zonas de transición sin ritmo definido (8 ventanas en 2 pacientes).

**B2. Explicar por qué se limpian las anotaciones aunque al modelo entre el ECG crudo.** Las anotaciones dan la etiqueta de cada ventana, los tiempos de los picos (para validar el detector) y los tramos de mala calidad. Una fila marcadora tratada como latido distorsiona las características (ver A2).

**B3. Celdas de evidencia** que muestren con salidas:
- que las 4,258 filas sin ritmo son 3,431 filas marcadoras sin tipo de latido (3,429 con `Start`/`End`) y 827 latidos con tipo pero sin ritmo (463 N, 283 U, 41 S, 39 V, 1 P);
- que de los 17,364 NaN en `beat_type`, 10,098 son de ruido y unos 7,266 no (casi todos marcadores);
- cuántas filas de ruido quedan fuera de un tramo `Start`/`End` de mala calidad (como la fila 1230 del caso 1002);
- que solo 4 casos no tienen ninguna arritmia (3 solo Normal y 1 Normal + Ruido).

**B4. Explicar las marcas `StartK`/`EndK`.** Indican el inicio y el fin del k-ésimo tramo de mala calidad dentro de un caso. La salida solo muestra los primeros 8 valores; puede haber más tramos.

**B5. Control de calidad de la carga: tabla de descomposición por clase.** Mostrar que cada diferencia con lo publicado se explica exactamente: diferencia = latidos rellenados − duplicados eliminados − marcadores con ritmo eliminados. Ejemplos: Normal +449 = +486 − 25 − 12; AFib −1 = +31 − 32; Bloqueo AV −3 = 0 − 3. Suma total: +645. Aclarar que los "latidos publicados" cuentan filas, incluidas las marcadoras.

**B6. Documentar la discrepancia de duración:** el metadata suma 534,398 s analizados frente a los 734,528 s publicados. No se sabe de dónde sale la diferencia.

---

## C. EDA (secciones 1, 4 y 6)

**C1. Sección 1.4:** hecho (porcentajes y definición de caso).

**C2. Ritmos por caso.** Interpretar la asimetría a la derecha (media 2.55 > mediana 2, por los casos con 4 a 6 ritmos). Aclarar que Ruido cuenta como "ritmo" y añadir el mismo conteo sin Ruido.

**C3. Dato de los casos sin ritmo normal.** 112 casos no tienen ningún tramo normal: 94 son solo AFib o AFib + Ruido, y 100 de los 112 tenían ECG preoperatorio normal (la AFib apareció o se detectó en la cirugía). Recordar que el segmento se eligió por parecer arrítmico. Añadirlo con su salida.

**C4. Rehacer la línea de tiempo de ritmos.** Usar tiempo relativo de 0 a 20 minutos por caso (no la posición en la grabación original), incluir los tramos de Ruido como bandas de color y elegir casos que en conjunto muestren todas las clases. **Recordar a Juan revisar la nueva gráfica y su interpretación.**

**C5. Gráfica de tipos de latido.** Explicar las siglas:
- N: normal (incluye bloqueos de rama)
- S: supraventricular (incluye todos los latidos durante AFib)
- V: ventricular
- U: inclasificable por ruido
- P: no documentado en el artículo; 7 filas, probablemente de marcapasos (suposición)

Advertir que la barra de Ruido solo refleja unas 19 filas, porque el ruido no tiene tipo de latido.

**C6. Boxplots de RR.** En los títulos o ejes, indicar que la izquierda muestra todos los intervalos RR en segundos (cada caja es un ritmo) y la derecha el coeficiente de variación del RR calculado una vez por caso.

**C7. Mann-Whitney.** Explicar la prueba (compara si los valores de un grupo tienden a ser mayores que los de otro, sin suponer normalidad) y aplicar corrección por comparaciones múltiples (Bonferroni: p < 0.05 / 10 = 0.005).

**C8. Ejemplos de ECG.**
- Comentar el código de `ventana_ecg` y `ejemplo_por_ritmo`.
- Etiquetar el eje x ("segundos desde el inicio de la grabación de VitalDB") y quitar el desplazamiento "+1.907e4".
- Escribir la interpretación de cada ejemplo: dónde está la arritmia, cómo se reconoce y cuánto dura (p. ej. TV: salva de unos 10 latidos anchos a unos 200 lpm durante 2.7 s; bigeminismo en ectopia ventricular; P sin QRS en Bloqueo AV).
- Revisar la espiga no anotada cerca del segundo 1609.9 en el ejemplo de AFib.

**C9. Renderizado clínico.** ECG con cuadrícula de papel (25 mm/s, 10 mm/mV) y una gráfica interactiva (plotly) para explorar un caso completo.

---

## D. Preprocesamiento y características (secciones 7 y 8)

**D1. Comentar el bloque de validación del detector** y explicar el algoritmo con un ejemplo numérico. Es un algoritmo fijo, sin entrenamiento: filtro de 5 a 15 Hz, derivada al cuadrado, media móvil de 120 ms, umbral adaptativo (0.25 × percentil 99.5 por bloques de 10 s), separación mínima de 250 ms y ajuste al máximo en ±50 ms.

**D2. Agregar desde el inicio (no como experimento) las características de forma de onda.**
- Por qué: las clases que fallan (Bloqueo AV, WAP/MAT, Disfunción del nodo) y la separación TV/SVTA se distinguen por la onda P y el ancho del QRS, que hoy no se miden.
- Cómo: delinear las ondas (p. ej. `neurokit2.ecg_delineate`) y calcular por ventana:
  - ancho medio del QRS;
  - fracción de latidos con onda P detectable;
  - intervalo PR medio y su variabilidad;
  - número de formas distintas de P (para WAP/MAT);
  - relación P:QRS (P sin QRS, para Bloqueo AV);
  - amplitud y forma de T;
  - correlación de cada latido con la plantilla promedio del paciente (latidos ectópicos).
- Requiere un preprocesamiento extra: corrección de línea base, promediado de plantilla y normalización de amplitud.

**D3. Tabla de características detallada:** una fila por característica (incluidas las nuevas) con qué mide, cómo se calcula y qué clases ayuda a separar. La tabla actual agrupa demasiado (p. ej. "Señal: calidad").

**D4. Comentar el código del ventaneo** y agregar una celda que explique el número de ventanas: 53,110 posibles = 50,573 válidas + 2,386 mixtas + 60 con pocos latidos + 91 sin filas. Indicar que los 46 s son el tiempo de construir ventanas y calcular características.

**D5. Mostrar el dataset de características** que entra al modelo: `head()`, `describe()`, tipos y nulos, después de la extracción.

**D6. Explicar la salida del conteo por clase tras el tope** y por qué se aplica (tiempo de cómputo, sobre todo del SVM, y reducción del desbalance), junto con su costo (ver A6).

**D7. Correlación y redundancia.**
- Justificar Spearman: mide relaciones monótonas aunque no sean lineales (p. ej. frecuencia = 60 / RR) y es robusto a colas largas y valores extremos.
- Listar todos los pares con |ρ| > 0.85 (no solo > 0.95).
- Proponer la eliminación: agrupar las variables muy correlacionadas y conservar una por grupo (la más interpretable o la de mayor importancia). Ejemplos ya visibles: `fc_media`/`rr_media` (ρ = −1.00), `rmssd`/`sd1` (0.999), `sdnn`/`rmssd` (0.97). La ejecución es el experimento F5.

**D8. Escalado.** Explicar por qué `StandardScaler`: los modelos que lo necesitan (regresión logística, SVM) funcionan mejor con media 0 y desviación 1, y los árboles no lo necesitan. Experimento opcional: `RobustScaler` (usa mediana y rango intercuartílico) por las colas largas de las variables de RR.

---

## E. Exploración no supervisada (sección 9)

**E1. Aclarar su propósito y su límite.** Sirve para ver si hay estructura natural, encontrar atípicos y entender qué clases se parecen. No permite concluir sobre el modelo supervisado: con 2 componentes solo se conserva un 49 % de la varianza (29 % + 20 %), y con 10 clases y 31 variables la proyección en 2D pierde mucho.

**E2. Explicar la gráfica de varianza explicada acumulada:** cada punto es la fracción de la varianza total que se conserva con k componentes (10 componentes = 84 %).

**E3. Gráficas más grandes y leyenda legible:** marcadores grandes o cuadrados de color en la leyenda.

**E4. Interpretación por agrupaciones de alto nivel**, con números de la tabla cruzada de K-Means:
- regulares: grupos 1 y 6, Normal y Disfunción del nodo;
- taquiarritmias: grupo 5, AFib, SVTA y TV;
- irregulares con patrón corto-largo: grupos 2 y 3, ectopias;
- irregulares lentos: grupo 0, AFib, Disfunción del nodo, WAP/MAT y Bloqueo AV;
- ruido: grupo 9, 394 de 419 ventanas.

Conclusión: sin etiquetas, el algoritmo separa por frecuencia, regularidad y calidad, pero no por el origen del impulso (onda P, ancho del QRS), lo que anticipa las confusiones.

---

## F. Experimentos para mejorar el modelo (cada uno con antes y después)

**F1. Ventanas deslizantes con solapamiento.** Ventana de 10 s con paso de 1 o 2 s, frente a las ventanas fijas sin solapamiento actuales.
- Ventajas: un episodio breve queda completo en alguna ventana (hoy 2,386 ventanas se pierden por mixtas), hay más ejemplos de clases raras y la latencia baja en la demo (una predicción cada 1 o 2 s).
- Cuidado: las ventanas solapadas son casi copias, así que la separación por paciente (A1) es indispensable. Evaluar en test con un esquema comparable.

**F2. Etiquetado con prioridad a lo anormal.** Extender la regla de "contiene" (hoy solo TV y SVTA con 3 o más latidos) a las demás arritmias, con un umbral por clase (p. ej. cobertura ≥ 30 %) y un orden de prioridad clínica (TV > Bloqueo AV > AFib > SVTA > ...). Comparar con la regla actual (un ritmo debe cubrir el 80 % de la ventana). Riesgo: ventanas casi normales etiquetadas como arritmia, lo que baja la precisión.

**F3. Aumento de datos para las clases minoritarias** (pedido explícito del profesor). Comparar:
- `class_weight` (actual);
- SMOTE sobre las características;
- aumento sobre la señal cruda: ruido gaussiano, cambio de amplitud, deriva de la línea base y estiramiento leve del tiempo, recalculando las características.

Siempre solo dentro del entrenamiento de cada pliegue. Analizar la matriz de confusión antes y después, sobre todo en Bloqueo AV, WAP/MAT, TV y SVTA: si mejora su recall o la confusión solo se mueve a otra clase.

**F4. Tamaño de ventana.** Probar 5, 10 y 20 s, o combinar una ventana corta (salvas) con una larga (irregularidad).

**F5. Eliminar características redundantes** con |ρ| > 0.85 (según D7) como reducción de dimensionalidad. Comparar F1 macro, recall por clase y tiempos.

**F6. Ruido como filtro previo de calidad.** Un primer modelo decide si la ventana es interpretable y solo las interpretables pasan al clasificador de ritmo. Comparar con Ruido como una clase más.

**F7. Esquema en dos etapas o fusión clínica**, solo como comparación: normal frente a anormal y luego el tipo de arritmia; o agrupar bradiarritmias (Disfunción del nodo y Bloqueo AV).

**F8. Priorizar las arritmias peligrosas.** Ajustar los umbrales de decisión por clase (p. ej. alertar TV con probabilidad ≥ 0.3) usando las curvas precisión-recall. Reportar el recall (o F2) de TV, Bloqueo AV y AFib además del F1 macro.

**F9. Búsqueda de hiperparámetros ampliada.** Incluir el SVM, que fue el mejor sin afinar (0.556) pero no se afinó por su costo, y probar XGBoost o LightGBM.

**F10. Entrenar sin tope por clase** (todas las ventanas, con `class_weight`) y comparar con el tope de 5,000.

---

## G. Evaluación, explicabilidad y registro (secciones 10 a 14)

**G1. Líneas base.** Explicar que la "clase mayoritaria" salió AFib por un empate tras el tope (Normal y AFib con 5,000 cada una; `idxmax` toma la primera). Agregar también una línea base aleatoria estratificada (`DummyClassifier(strategy="stratified")`).

**G2. Tabla resumen de experimentos:** qué se probó, F1 macro en validación por paciente, recall de las clases peligrosas, tiempo, decisión y razón.

**G3. Definir las métricas en el texto.**
- Exactitud: aciertos sobre el total; engañosa con desbalance.
- Exactitud balanceada: promedio del recall de cada clase.
- F1 macro: promedio del F1 por clase.
- MCC.
- PR-AUC: área bajo la curva precisión-recall; mejor que ROC con desbalance.

Explicar que la tabla de comparación inicial (sección 11.1) usa hiperparámetros por defecto, solo para elegir qué familias afinar, y comentar el sobreajuste visible (Random Forest con 1.00 en train frente a 0.53 en validación).

**G4. Explicar por qué HistGradientBoosting.** Agrupa cada variable en 255 intervalos (histogramas) antes de buscar cortes. Es mucho más rápido que el `GradientBoostingClassifier` clásico con decenas de miles de filas, maneja NaN de forma nativa y admite `class_weight`. XGBoost y LightGBM usan la misma idea; probarlos en F9.

**G5. Matrices de confusión legibles.** Quitar la cuadrícula que atraviesa los números (`ax.grid(False)`), alinear y rotar las etiquetas del eje x.

**G6. Explicar la importancia por permutación en el texto.** La barra es la caída media del F1 macro al barajar esa variable en test. La línea negra es la desviación estándar entre 10 repeticiones. Una barra cercana a 0 o negativa significa que la variable no aporta, o que barajarla incluso mejoró un poco por azar (son candidatas a eliminar).

**G7. SHAP para las 10 clases.** Mostrar el gráfico de puntos (beeswarm) para cada clase, con su interpretación, y explicar cómo se lee: cada punto es una ventana; el eje x es cuánto empuja esa variable hacia la clase (derecha = a favor); el color es el valor de la variable (rojo alto, azul bajo); gris = valor faltante. Después de A3 y A7, revisar que Ruido ya no dependa de `n_latidos`.

**G8. Explicar la prueba de etiquetas barajadas (14.3).** Se mezclan al azar las etiquetas de train, rompiendo la relación entre señal y ritmo, y se repite la validación. Si el F1 sigue alto, hay fuga o estructura espuria; si cae al azar (como pasó: 0.08), el modelo aprende la relación real.

**G9. Comparación con los autores (sin red neuronal propia por tiempo).** Agrupar nuestras predicciones en normal y anormal y calcular las mismas métricas que reportan (CNN 1D, segmentos de 5 s, partición por paciente: F1 0.910, sensibilidad 0.848, especificidad 0.988, AUROC 0.990). Aclarar que no es directamente comparable: ellos resuelven un problema binario con deep learning; nosotros, 10 clases con ML tradicional. En trabajo futuro: entrenar una red de series de tiempo (CNN 1D) sobre la señal cruda, como sugirió el profesor.

---

## H. Despliegue, monitoreo y demo (secciones 15 y 16)

**H1. Probar la app de Streamlit.** El notebook genera `app/app.py`, pero nunca se ejecutó con `streamlit run app/app.py`. Probarla y corregir lo que falle.

**H2. Ejemplos para la demo:** tres casos del conjunto de **test**, de 1 minuto cada uno, que en conjunto muestren varios ritmos (p. ej. normal con ectopias; AFib; salva de TV con ruido). Reemplazan al ejemplo actual de 5 minutos del caso 1001.

**H3. Documentar el formato de entrada.** Los `.npz` de `data/ecg_cache` son la caché interna del proyecto: un archivo comprimido de NumPy con el vector de voltajes (`ecg`) y su desfase de tiempo (`offset_s`). No es el formato de un monitor. Un monitor real transmite la señal en vivo (protocolos del fabricante o HL7; VitalDB la grabó con Vital Recorder). La integración sería: recibir el flujo, guardar los últimos 10 s en memoria, extraer las características y predecir. La app acepta CSV para la demostración.

**H4. Demo tipo monitor.**
- Monitor izquierdo: traza animada con la frecuencia cardíaca.
- Monitor derecho: la misma traza sincronizada, con el fondo coloreado según la predicción y alerta en las arritmias peligrosas.
- Opcional: franja con la etiqueta real de los médicos.
- Página HTML que reproduce predicciones calculadas antes con el modelo real (más fluida que Streamlit), con ventanas deslizantes para que la predicción se actualice cada 1 o 2 s.

**H5. Explicar la tabla de deriva de datos.**
- PSI (Population Stability Index): mide cuánto cambió la distribución de una variable entre la referencia y los datos nuevos. Por convención, < 0.1 es estable, 0.1 a 0.2 moderado y > 0.2 alto.
- KS (Kolmogorov-Smirnov): distancia máxima entre las dos distribuciones acumuladas, con su p-valor. Con miles de datos, el p-valor sale casi siempre ínfimo aunque el cambio sea pequeño, por eso se decide con el PSI.
- Explicar la simulación de "equipo nuevo": se multiplicó `sig_ruido_alto` por 3 y `sig_std`/`sig_p2p` por 0.6 en la mitad de los datos, imitando un monitor más ruidoso y con menor ganancia.

**H6. Pipeline de reentrenamiento ejecutable.** Hoy la sección 16 solo lo describe. Crear una función o script (p. ej. `reentrenar.py`) que, con datos nuevos etiquetados:
1. construya ventanas y características con el mismo `ecg_utils.py`;
2. valide con partición por paciente;
3. entrene el candidato con los hiperparámetros vigentes (o una búsqueda corta);
4. compare candidato y modelo actual sobre un conjunto de validación fijo;
5. lo promueva solo si no empeora el recall de las clases peligrosas;
6. guarde la nueva versión con fecha y métricas.

Debe ser mucho más corto que el notebook de desarrollo.

---

## I. Conclusiones

**I1.** Releer y reescribir conclusiones, limitaciones y trabajo futuro con los resultados finales. Mencionar la comparación con los autores (G9) y la red neuronal de series de tiempo como trabajo futuro.
