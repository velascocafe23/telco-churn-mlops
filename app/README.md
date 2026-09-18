# Demostración del modelo

Aplicación Streamlit que expone el modelo de predicción de cancelación de clientes en dos
modos de uso.

## Requisitos

El modelo entrenado debe existir en el repositorio:

- `models/modelo_churn.joblib`
- `models/modelo_churn_metadatos.json`

Ambos se generan al ejecutar el notebook `notebooks/5-models/6-seleccion-modelo.ipynb`, y
están versionados, así que no hace falta reentrenar para ejecutar la demo.

## Ejecución local

```bash
uv sync
uv run streamlit run streamlit_app.py
```

La aplicación queda disponible en `http://localhost:8501`.

## Modos de uso

### Predicción individual

Formulario con los diecinueve atributos que requiere el modelo. Devuelve la probabilidad
estimada de cancelación y la decisión correspondiente al umbral configurado.

### Procesamiento por lote

Carga de un archivo CSV con varios clientes. La aplicación valida que estén todas las
columnas requeridas, calcula las predicciones y permite descargar el resultado ordenado por
probabilidad descendente, que es el orden en que conviene contactar a los clientes.

El archivo `app/ejemplos/clientes_ejemplo.csv` sirve como plantilla y se puede descargar
desde la propia aplicación.

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

## Normalización de los datos de entrada

La aplicación normaliza los archivos cargados antes de predecir: unifica la representación
de los valores ausentes, convierte los cargos a numérico e interpreta la condición de
adulto mayor cuando llega codificada como cero y uno.

Esto es necesario porque el modelo fue entrenado sobre datos ya normalizados. Un archivo en
el formato original de la fuente, sin esa preparación, produciría indicadores en cero para
esa columna y el modelo degradaría su predicción sin emitir ningún aviso.
