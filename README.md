# Predicción de cancelación de clientes

Sistema de predicción de fuga de clientes en una empresa de telecomunicaciones,
desarrollado como trabajo final del curso **Ciencia de Datos en Producción**
(Universidad Pontificia Bolivariana, prof. Jose R. Zapata).

**Aplicación desplegada:** https://telco-churn-mlops-upb-sva.streamlit.app

---

## El problema

Anticipar qué clientes van a cancelar su servicio en el próximo ciclo de facturación, con
antelación suficiente para que el equipo de retención pueda intervenir. Adquirir un cliente
nuevo cuesta varias veces más que retener uno existente, de modo que el valor del sistema
está en priorizar correctamente a quién contactar.

**Datos:** Telco Customer Churn (IBM), 7.043 clientes, 21 atributos, clasificación binaria
con desbalance moderado (26,5% de cancelación).

**Encuadre:** aprendizaje supervisado, entrenamiento offline, inferencia por lote mensual
con consulta individual como complemento.

---

## Resultados

| Métrica | Valor |
|---|---|
| F1 sobre la clase de fuga | 0,6205 |
| Precisión | 0,5395 |
| Exhaustividad | 0,7299 |
| ROC-AUC | 0,8411 |
| Umbral de decisión | 0,562 |

Sobre el conjunto de prueba, de 374 cancelaciones reales el modelo detecta 273, deja
escapar 101 y genera 233 contactos innecesarios.

### El hallazgo principal: el modelo no supera al modelo base

Se evaluaron cinco familias de modelos, se optimizaron hiperparámetros, se ajustó el umbral
de decisión y se incorporó un atributo derivado. El resultado final, F1 de 0,6205, es
equivalente al del modelo base establecido en la fase de prueba de concepto, 0,620.

**Esto no es una limitación del procedimiento sino su conclusión mejor sustentada.** Cuatro
evidencias independientes convergen en el mismo diagnóstico:

1. **La curva de aprendizaje es plana**, con una brecha de 0,0053 entre entrenamiento y
   validación. Más datos no mejorarían el desempeño.
2. **El 6,1% de los registros pertenece a perfiles de servicio con desenlace ambiguo**:
   clientes con atributos idénticos y resultado opuesto. Ningún algoritmo puede resolverlo.
3. **La búsqueda de hiperparámetros eligió regularización fuerte** (`C=0,01`). El modelo
   rinde mejor cuando se le impide ajustarse, lo que ocurre cuando hay poca señal.
4. **El 50,9% de los errores se concentra a menos de 0,15 del umbral**, frente al 18,0% de
   los aciertos. El modelo no se equivoca con seguridad: se equivoca donde reconoce su
   propia incertidumbre.

El límite está en la información contenida en los atributos disponibles, no en la elección
del algoritmo. La vía de mejora es incorporar atributos nuevos —reclamos, calidad de
servicio percibida, interacciones con soporte— y no probar más modelos.

### Otros hallazgos

- **`Contract` en solitario alcanza el 91,6% del F1 del modelo completo.** Los otros
  dieciocho atributos aportan en conjunto menos del 9% restante.
- **La ponderación de clases no mejora el modelo, desplaza el umbral.** Ambas variantes del
  modelo lineal obtienen el mismo ROC-AUC (0,8463): cambia dónde se corta el ordenamiento,
  no la capacidad de ordenar.
- **La familia más simple resultó la mejor.** Los modelos de árboles no aprovecharon la
  interacción entre contrato y antigüedad, porque el pipeline ya la expone al modelo lineal
  mediante la antigüedad discretizada en tramos.
- **El atributo derivado `servicios_contratados` no aporta.** Es la suma de indicadores que
  el modelo ya recibe por separado, y en un modelo lineal cualquier combinación lineal de
  columnas existentes es redundante por construcción.

---

## Arquitectura

```
Datos crudos → Feature Pipeline → Training Pipeline → Modelo serializado
                      ↓                                       ↓
                 Validación                      Inference Pipeline / Aplicación
```

### Feature Pipeline

`src/pipelines/feature_pipeline/`

Convierte datos crudos en atributos. **Es la única fuente de verdad sobre la construcción de
atributos**: el entrenamiento, la inferencia y la aplicación web lo importan, de modo que
las transformaciones aplicadas en producción son necesariamente las del entrenamiento. No
hay dos implementaciones que puedan divergir.

Incluye validación en tres niveles: esquema (Pandera), integridad entre campos y
distribución. Un fallo de esquema o integridad detiene el pipeline y **no persiste el
archivo de salida**.

### Training Pipeline

`src/pipelines/training_pipeline/`

Entrena, optimiza el umbral sobre predicciones de validación cruzada y persiste el modelo
con sus metadatos. Antes de entrenar verifica la partición (fuga de información,
estratificación, distribuciones) y después diagnostica el ajuste con umbrales declarados.

### Inference Pipeline

`src/pipelines/inference_pipeline/`

Genera predicciones sobre datos nuevos, ordenadas por probabilidad descendente. Lee el
umbral de los metadatos del modelo, no de una constante.

### Aplicación

`streamlit_app.py`

Dos modalidades: evaluación individual mediante formulario y procesamiento por lote desde
archivo. El umbral de decisión es configurable desde la interfaz.

---

## Ejecución

```bash
# Entorno
uv sync

# Pipeline completo
uv run python src/pipelines/feature_pipeline/feature_pipeline.py
uv run python src/pipelines/training_pipeline/train_pipeline.py
uv run python src/pipelines/inference_pipeline/inference_pipeline.py

# Aplicación
uv run streamlit run streamlit_app.py

# Pruebas
uv run pytest
```

---

## Estructura

```
├── notebooks/            Prueba de concepto, ocho notebooks
│   ├── 1-data/           Descarga y encuadre del problema
│   ├── 2-exploration/    Tipos de datos y valores ausentes
│   ├── 3-analysis/       Análisis exploratorio
│   ├── 4-feat_eng/       Pipeline de atributos
│   ├── 5-models/         Modelo base y selección
│   └── 6-interpretation/ Interpretación y análisis de errores
├── src/pipelines/        Arquitectura FTI en scripts
├── tests/                74 pruebas unitarias
├── models/               Modelo serializado y metadatos
├── reports/              Evidencia de validación
├── app/                  Documentación, ejemplos y capturas
└── streamlit_app.py      Aplicación de despliegue
```

---

## Decisiones de diseño

**El umbral de decisión es un parámetro de negocio, no del modelo.** El valor que maximiza
el F1 (0,562) equivale a asumir una relación de costos de entre 2 y 3 a 1 entre no detectar
una cancelación y contactar innecesariamente a un cliente. Optimizar F1 no es neutral:
adopta ese supuesto sin declararlo. Por eso el umbral se almacena en los metadatos y se
expone como control en la aplicación.

**Se persiste el preprocesamiento sin ajustar.** Lo que se comparte entre etapas es la
definición de las transformaciones, no los parámetros aprendidos, que deben ajustarse sobre
cada conjunto de entrenamiento.

**La comparación entre modelos usa una prueba estadística, no diferencias de medias.** Con
desviaciones entre pliegues del orden de 0,02, una diferencia de 0,005 entre medias es
ruido. La prueba t pareada con corrección de Nadeau y Bengio ajusta la varianza por el
solapamiento entre pliegues de validación cruzada, que la prueba t ordinaria subestima.

**El criterio de desempate se fijó antes de ejecutar:** a igualdad estadística, el modelo
más simple y rápido. Elegir el criterio después de ver los resultados sería seleccionar la
regla que favorece al modelo preferido.

**La distinción entre error y advertencia es deliberada.** Un valor de categoría desconocido
siempre es un problema y detiene el proceso. Una tasa de cancelación fuera del rango
habitual puede reflejar un cambio real de la población, así que se registra sin detener la
producción.

---

## Calidad del código

- 74 pruebas unitarias
- Linter y formateador (ruff) con reglas de bugbear, bandit, pylint y complejidad ciclomática
- Verificación de tipos (mypy) con anotaciones obligatorias
- Ganchos de pre-commit y validación de mensajes de commit
- Integración continua en cada pull request
- 16 issues, 32 pull requests, cada tarea con rama, commits convencionales y merge

---

## Trabajo futuro

1. Incorporar atributos de comportamiento: reclamos, incidencias de servicio, interacciones
   con soporte. Es la única vía que puede reducir la zona ambigua donde se concentran los
   errores.
2. Investigar el segmento de cancelaciones no detectadas: clientes con 28 meses de
   antigüedad mediana que se van sin señales previas.
3. Establecer con el área comercial la relación de costos entre los dos tipos de error, que
   es el dato que falta para fijar el umbral con fundamento.
4. Eliminar `gender` del conjunto de atributos: no tiene aporte medible y su uso en
   decisiones comerciales es cuestionable.

---

## Autor

Sebastián Velasco Ardila
Maestría en Ciencia de Datos, Universidad Pontificia Bolivariana
