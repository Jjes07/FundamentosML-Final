# Guion del Video Demostrativo (2 minutos)
## Sistema de Apoyo a la Detección de Arritmias en Cirugía
### Modelo desplegado (`app/app.py`) y monitor de demostración (`demo/monitor.html`)

---

> **Especificaciones:**
> - **Duración:** 120 s como máximo (Sección 4 de `docs/Requeriments_FinalP.pdf`: "video de máximo 2 minutos mostrando el funcionamiento del modelo desplegado y su interfaz").
> - **Narración:** unas 260 palabras, a unos 130 palabras por minuto.
> - **Modelo:** XGBoost v3. La app lo muestra al pie: "Modelo XGBoost v2026.11.02". Si se vuelve a ejecutar el notebook, la fecha cambia.
> - **Antes de grabar:** ejecutar `streamlit run app/app.py` y abrir `demo/monitor.html` en el navegador. Revisar en pantalla los resultados reales de cada ejemplo: este guion no inventa cifras por tramo, se leen de la app.

---

## Estructura

| Bloque | Tiempo | Qué se muestra |
| :---: | :---: | :--- |
| **B1** | 00:00 – 00:15 | Título de la app y aviso de herramienta de apoyo |
| **B2** | 00:15 – 00:45 | Ejemplo "Fibrilación auricular": conclusión, señal coloreada, tabla |
| **B3** | 00:45 – 01:15 | Monitor de demostración con la salva de TV |
| **B4** | 01:15 – 01:40 | Ejemplo "Normal con ectopias", control de confianza y descarga |
| **B5** | 01:40 – 02:00 | Archivo propio, respaldo del modelo y cierre |

---

## Guion Detallado

### B1. Presentación (00:00 – 00:15)

**Pantalla**
- App abierta: título "Apoyo a la detección de arritmias en cirugía" y el aviso "Herramienta de apoyo. No reemplaza el criterio clínico ni diagnostica."

**Narración** (unas 35 palabras)
> "Durante una cirugía, el anestesiólogo vigila el electrocardiograma mientras atiende muchas otras tareas. Esta herramienta analiza el ECG en tramos de diez segundos y señala los que tienen un ritmo anormal. Es un apoyo: no diagnostica."

---

### B2. Ejemplo de fibrilación auricular (00:15 – 00:45)

**Pantalla**
- Opción "Un ejemplo incluido" y, en el selector, "Fibrilación auricular". Se lee la nota: el análisis lo hace una copia del modelo que nunca vio a este paciente.
- Las cuatro cifras de arriba: tramos analizados, con alerta, para revisar y con ruido.
- El mensaje rojo "Alerta: Fibrilación auricular a partir del segundo ...".
- La señal con el fondo rojo en los tramos con alerta y la tabla "Detalle por tramo de 10 segundos" (inicio, estado, ritmo, frecuencia, confianza).

**Narración** (unas 65 palabras)
> "Elegimos un ejemplo real de un paciente que el modelo nunca vio, en fibrilación auricular. La app muestra primero la conclusión: cuántos tramos tienen alerta y desde qué segundo. Debajo, la señal se colorea: rojo para alerta, ámbar para revisar, verde sin alerta y gris para ruido. La tabla explica cada tramo en lenguaje claro, con la frecuencia cardíaca y la confianza del modelo."

---

### B3. Monitor con la salva de taquicardia ventricular (00:45 – 01:15)

**Pantalla**
- `demo/monitor.html`, ejemplo "Salva de TV" (caso 1484). Dos monitores sincronizados: el convencional y el que colorea según el modelo, con la etiqueta de los médicos al lado.
- Dejar correr hasta que entra la salva: la alerta de TV se enciende y se mantiene unos 10 s.

**Narración** (unas 65 palabras)
> "Así se vería en un quirófano. A la izquierda, el monitor convencional; a la derecha, el mismo trazo con la predicción del modelo, actualizada cada dos segundos. Cuando aparece una salva breve de taquicardia ventricular, el monitor se pone en rojo. La alerta llega cuando la salva entra en los últimos diez segundos analizados: ese es el retardo real de este diseño."

---

### B4. Ectopias, confianza y descarga (01:15 – 01:40)

**Pantalla**
- En la app, ejemplo "Normal con ectopias" (caso 1110).
- Mover el control "Confianza mínima" de 0.60 a un valor más alto y mostrar cómo cambian los tramos "Para revisar".
- Clic en "Descargar resultados (CSV)".

**Narración** (unas 55 palabras)
> "En este paciente con latidos prematuros, la mayoría de los tramos son normales. Si el modelo duda, el tramo se marca para revisión, y el usuario decide qué tan exigente ser con este control. Los resultados se descargan para el registro. El modelo también se equivoca: por eso siempre pide confirmar en el monitor antes de actuar."

**Nota para la grabación:** en las predicciones cada 2 s del monitor, este ejemplo tuvo una falsa alarma aislada de Bloqueo AV. Si aparece en la app, mencionarla: muestra con honestidad la baja precisión de esa clase.

---

### B5. Archivo propio y cierre (01:40 – 02:00)

**Pantalla**
- Opción "Mi propio archivo" y el desplegable "Formato del archivo y cómo funcionaría en un monitor".
- Pie de la app con el modelo y su advertencia. Cierre con el enlace al repositorio.

**Narración** (unas 45 palabras)
> "También acepta un archivo propio, con la frecuencia de muestreo del equipo. El modelo se validó con pacientes que nunca vio, en cinco grupos distintos, y detecta casi el 92 por ciento de los tramos anormales. Bloqueo AV y ritmo multifocal aún se reconocen poco."

---

## Cifras citadas y su origen en el notebook

| Cifra | Origen |
| :--- | :--- |
| Tramos de 10 s, predicciones cada 2 s en el monitor | Secciones 8 y 15.2 |
| "Casi el 92 %" de los tramos anormales | G9: sensibilidad 0.919 al agrupar en normal y anormal (validación externa, sin Ruido) |
| Alerta de TV que dura unos 10 s | Interpretación de la sección 15 (salida de `generar_demo.py`) |
| Falsa alarma de Bloqueo AV en el ejemplo de ectopias | Salida de `generar_demo.py`: 22 Normal, 3 Ectopia ventricular, 1 Bloqueo AV |

## Pendiente para la grabación

- Grabar con la app local o con la app ya publicada (publicarla requiere una cuenta del equipo en Streamlit Community Cloud o Hugging Face Spaces; ver `PENDIENTES.md`).
- Cronometrar la narración: si pasa de 2 minutos, recortar B4.
