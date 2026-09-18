# Demostración del modelo

Aplicación Streamlit que expone el modelo de predicción de cancelación de clientes en dos
modos de uso.

## Aplicación desplegada

**https://telco-churn-mlops-upb-sva.streamlit.app**

Desplegada en Streamlit Community Cloud desde la rama `main`. Cada cambio en el
repositorio se refleja automáticamente en la aplicación.

## Modos de uso

### Predicción individual

Formulario con los diecinueve atributos que requiere el modelo. Devuelve la probabilidad
estimada de cancelación y la decisión correspondiente al umbral configurado.

Para probarlo: con los valores por defecto (contrato mes a mes, antigüedad de 12 meses,
cargo mensual de 70) la probabilidad ronda el 73% y el cliente se marca en riesgo. Al
cambiar el contrato a `Two year` y subir la antigüedad a 60 meses, la probabilidad cae
por debajo del umbral.

### Procesamiento por lote

Carga de un archivo CSV con varios clientes. La aplicación valida que estén todas las
columnas requeridas, calcula las predicciones y permite descargar el resultado ordenado
por probabilidad descendente, que es el orden en que conviene contactar a los clientes.

Archivos de referencia:

| Archivo | Contenido |
|---|---|
| `app/ejemplos/clientes_ejemplo.csv` | Entrada: 24 clientes, ocho por cada tipo de contrato |
| `app/ejemplos/predicciones_ejemplo.csv` | Salida esperada para ese archivo |

Sobre el archivo de ejemplo, el resultado es de 24 clientes evaluados, 5 en riesgo, 20.8%.
Ese mismo resultado se obtiene ejecutando el pipeline de inferencia por línea de comandos,
lo que confirma que ambas vías comparten modelo, transformaciones y umbral.

## Ejecución local

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Disponible en `http://localhost:8501`.

## Umbral de decisión

La barra lateral permite ajustar el umbral. El valor por defecto, 0.562, es el que maximiza
el F1 sobre validación cruzada, pero no es una constante del modelo: es una decisión de
negocio que depende de cuánto cueste perder un cliente frente al costo de una oferta de
retención innecesaria.

Según el análisis del notebook de interpretación, sobre el conjunto de prueba:

| Relación de costos | Umbral | Clientes marcados | Cancelaciones detectadas |
|---|---|---|---|
| 2 a 1 | 0.66 | 386 | 236 de 374 |
| 3 a 1 | 0.41 | 682 | 323 de 374 |
| 5 a 1 | 0.31 | 809 | 347 de 374 |
| 10 a 1 | 0.28 | 853 | 354 de 374 |

## Arquitectura

La aplicación no implementa lógica de transformación propia. Importa `construir_atributos`
del pipeline de atributos, de modo que aplica exactamente las mismas transformaciones que
se usaron durante el entrenamiento.

Del modelo carga dos cosas: el pipeline serializado, que incluye el preprocesamiento
ajustado, y los metadatos, de donde obtiene el umbral de decisión y la lista de columnas
esperadas. La interfaz no tiene conocimiento propio del modelo: si se reentrena con otra
configuración, la aplicación se adapta sin cambios de código.

## Despliegue

Streamlit Community Cloud instala dependencias desde `requirements.txt`, no desde
`pyproject.toml`. Ese archivo declara las dependencias directas y deja que la plataforma
resuelva las transitivas.

`scikit-learn` va fijada a la versión exacta con la que se serializó el modelo.
Deserializar un pipeline con una versión distinta produce avisos de incompatibilidad y,
ante cambios internos de la librería, puede fallar.

La versión de Python se fija en 3.12 en la configuración avanzada del despliegue, para que
coincida con la del proyecto.

## Evidencia

Las capturas de `app/capturas/` documentan el funcionamiento:

| Captura | Contenido |
|---|---|
| 01 a 06 | Ejecución local, ambas modalidades |
| 07 y 08 | Ejecución local tras reentrenar el modelo con el atributo derivado |
| 09 | Predicción individual en la aplicación desplegada |
| 10 | Procesamiento por lote en la aplicación desplegada |
