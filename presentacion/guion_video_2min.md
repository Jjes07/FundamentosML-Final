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
| **B3** | 00:45 – 01:15 | Monitor de demostración con taquicardia ventricular y pitido por latido |
| **B4** | 01:15 – 01:40 | Monitor con bloqueo AV: una clase que el modelo no reconoce |
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

### B3. Monitor con taquicardia ventricular (00:45 – 01:15)

**Pantalla**
- `demo/monitor.html`, botón "Taquicardia ventricular" (caso 2016), con "Sonido: sí". Dos monitores sincronizados: el convencional, que pita en cada latido, y el que colorea según el modelo, con la etiqueta de los médicos debajo.
- Dejar correr hasta que entra una salva: el fondo pasa al color de TV y aparece "⚠ Arritmia peligrosa: Taquicardia ventricular".

**Narración** (unas 65 palabras)
> "Así se vería en un quirófano. A la izquierda, el monitor convencional; a la derecha, el mismo trazo con la predicción del modelo, actualizada cada dos segundos. Cuando llegan las salvas de taquicardia ventricular, latidos anchos y rápidos, el monitor se pone en alerta. La alerta se mantiene mientras la salva siga dentro de los últimos diez segundos analizados: ese es el retardo real de este diseño."

---

### B4. Una clase difícil: bloqueo AV (01:15 – 01:40)

**Pantalla**
- En el monitor, botón "Bloqueo AV (clase difícil)" (caso 4364). La franja de los médicos marca bloqueo AV todo el minuto; la del modelo muestra ritmo normal y disfunción del nodo.
- Leer la nota: este tramo se eligió sin mirar las predicciones.

**Narración** (unas 55 palabras)
> "No todo sale bien. Aquí los médicos marcaron un bloqueo auriculoventricular durante todo el minuto, y el modelo nunca lo reconoce: lo confunde con ritmos lentos. Solo hubo nueve pacientes con bloqueo en toda la base. Por eso el sistema es un apoyo: cada alerta, y cada silencio, se confirma en el monitor."

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
| TV: 17 de 31 predicciones; Bloqueo AV nunca reconocido (20 Normal, 11 Disfunción del nodo) | Salida de `generar_demo.py` (sección 15) |
| Nueve pacientes con bloqueo AV | Sección 8 (409 ventanas de 9 pacientes) |

## Pendiente para la grabación

- Grabar con la app local o con la app ya publicada (publicarla requiere una cuenta del equipo en Streamlit Community Cloud o Hugging Face Spaces; ver `PENDIENTES.md`).
- Cronometrar la narración: si pasa de 2 minutos, recortar B4.
- En el monitor, la demo arranca en pausa: el sonido solo funciona después de un clic (Reproducir o elegir un caso).
- Los cuatro primeros casos se eligieron por ser los que el modelo clasifica mejor. Si se muestran, decirlo: "son ejemplos ilustrativos".
