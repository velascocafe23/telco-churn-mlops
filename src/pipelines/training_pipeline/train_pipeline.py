"""Pipeline de entrenamiento del modelo de cancelacion.

Lee los atributos producidos por el pipeline de atributos, entrena el modelo
seleccionado en la fase de prueba de concepto, optimiza el umbral de decision y
persiste el pipeline completo junto con sus metadatos.

La configuracion del modelo no es arbitraria: proviene de la comparacion de cinco
familias del notebook 6, donde la regresion logistica con regularizacion fuerte
resulto estadisticamente indistinguible de las alternativas y notablemente mas rapida
de ajustar.

Ejecucion:

    uv run python src/pipelines/training_pipeline/train_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

DIRECTORIO_FUENTES = Path(__file__).resolve().parents[2]
if str(DIRECTORIO_FUENTES) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_FUENTES))

from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    ATRIBUTO_DERIVADO,
    CLASE_POSITIVA,
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    IDENTIFICADOR,
    OBJETIVO,
)
from pipelines.training_pipeline.verificacion_particion import (  # noqa: E402
    ResultadoParticion,
    verificar_particion,
)

RAIZ = Path(__file__).resolve().parents[3]
ATRIBUTOS_POR_DEFECTO = RAIZ / "data" / "04_feature" / "telco_features.parquet"
MODELO_POR_DEFECTO = RAIZ / "models" / "modelo_churn.joblib"
METADATOS_POR_DEFECTO = RAIZ / "models" / "modelo_churn_metadatos.json"

PROPORCION_PRUEBA = 0.2
SEMILLA = 42
N_PLIEGUES = 5
N_TRAMOS_ANTIGUEDAD = 5

REGULARIZACION = 0.01
MAX_ITERACIONES = 1000
PONDERACION = "balanced"

COLUMNAS_MODELO_NUMERICAS = [*COLUMNAS_NUMERICAS, ATRIBUTO_DERIVADO]
COLUMNA_ANTIGUEDAD = ["tenure"]

registro = logging.getLogger(__name__)


def cargar_atributos(origen: Path) -> pd.DataFrame:
    """Lee el archivo de atributos producido por el pipeline de atributos."""
    if not origen.exists():
        mensaje = (
            f"No existe el archivo de atributos {origen}. Ejecute antes el pipeline de atributos."
        )
        raise FileNotFoundError(mensaje)

    datos = pd.read_parquet(origen)
    registro.info("Atributos cargados: %d registros, %d columnas", len(datos), datos.shape[1])
    return datos


def separar_objetivo(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de atributos del vector objetivo binario."""
    if OBJETIVO not in datos.columns:
        mensaje = f"El conjunto no contiene la columna objetivo '{OBJETIVO}'"
        raise ValueError(mensaje)

    columnas_a_descartar = [OBJETIVO]
    if IDENTIFICADOR in datos.columns:
        columnas_a_descartar.append(IDENTIFICADOR)

    atributos = datos.drop(columns=columnas_a_descartar)
    objetivo = (datos[OBJETIVO].astype("string") == CLASE_POSITIVA).astype(int)
    return atributos, objetivo


def construir_preprocesador() -> ColumnTransformer:
    """Arma el preprocesamiento con tres ramas sobre las columnas de entrada.

    La antiguedad alimenta dos ramas a proposito: como magnitud continua y como
    tramo discretizado. Su distribucion bimodal y su interaccion con el tipo de
    contrato justifican ambas representaciones.
    """
    rama_numerica = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="constant", fill_value=0)),
            ("escalado", StandardScaler()),
        ]
    )

    rama_antiguedad = Pipeline(
        steps=[
            (
                "tramos",
                KBinsDiscretizer(
                    n_bins=N_TRAMOS_ANTIGUEDAD,
                    encode="onehot-dense",
                    strategy="uniform",
                ),
            ),
        ]
    )

    rama_categorica = Pipeline(
        steps=[
            ("indicadores", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numericas", rama_numerica, COLUMNAS_MODELO_NUMERICAS),
            ("antiguedad", rama_antiguedad, COLUMNA_ANTIGUEDAD),
            ("categoricas", rama_categorica, COLUMNAS_CATEGORICAS),
        ],
        remainder="drop",
    )


def construir_pipeline() -> Pipeline:
    """Une preprocesamiento y estimador en un unico objeto serializable."""
    return Pipeline(
        steps=[
            ("preprocesamiento", construir_preprocesador()),
            (
                "modelo",
                LogisticRegression(
                    C=REGULARIZACION,
                    class_weight=PONDERACION,
                    max_iter=MAX_ITERACIONES,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )


def dividir_datos(
    atributos: pd.DataFrame, objetivo: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Parte los datos en entrenamiento y prueba conservando la proporcion de clases."""
    partes = train_test_split(
        atributos,
        objetivo,
        test_size=PROPORCION_PRUEBA,
        random_state=SEMILLA,
        stratify=objetivo,
    )
    registro.info(
        "Particion: %d registros de entrenamiento, %d de prueba",
        len(partes[0]),
        len(partes[1]),
    )
    return partes[0], partes[1], partes[2], partes[3]


class ErrorDeParticion(Exception):
    """Se levanta cuando la particion no es apta para entrenar y evaluar."""


def comprobar_particion(
    entrenamiento_x: pd.DataFrame,
    prueba_x: pd.DataFrame,
    entrenamiento_y: pd.Series,
    prueba_y: pd.Series,
) -> ResultadoParticion:
    """Verifica la particion antes de entrenar y detiene el proceso si es invalida.

    Un solo registro compartido entre conjuntos invalida la evaluacion posterior, de
    modo que conviene descubrirlo aqui y no despues de haber reportado metricas.
    """
    resultado = verificar_particion(entrenamiento_x, prueba_x, entrenamiento_y, prueba_y)

    for clave, valor in resultado.reporte.items():
        registro.debug("Particion | %s: %s", clave, valor)

    for advertencia in resultado.advertencias:
        registro.warning(advertencia)

    if not resultado.valida:
        for error in resultado.errores:
            registro.error(error)
        mensaje = f"La particion no supero las verificaciones ({len(resultado.errores)} errores)"
        raise ErrorDeParticion(mensaje)

    registro.info("Particion verificada: indices disjuntos, estratificacion y distribuciones")
    return resultado


def optimizar_umbral(pipeline: Pipeline, atributos: pd.DataFrame, objetivo: pd.Series) -> float:
    """Elige el umbral que maximiza el F1 sobre predicciones de validacion cruzada.

    Nunca se ajusta contra el conjunto de prueba: hacerlo convertiria la evaluacion
    final en optimista, porque el umbral habria visto esos datos.
    """
    validacion = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)
    probabilidades = cross_val_predict(
        pipeline, atributos, objetivo, cv=validacion, method="predict_proba"
    )[:, 1]

    precisiones, exhaustividades, umbrales = precision_recall_curve(objetivo, probabilidades)
    denominador = precisiones + exhaustividades
    f1_por_umbral = np.divide(
        2 * precisiones * exhaustividades,
        denominador,
        out=np.zeros_like(precisiones),
        where=denominador > 0,
    )

    mejor = int(np.argmax(f1_por_umbral[:-1]))
    umbral = float(umbrales[mejor])
    registro.info("Umbral optimo: %.4f (F1 en validacion %.4f)", umbral, f1_por_umbral[mejor])
    return umbral


def evaluar(
    pipeline: Pipeline, atributos: pd.DataFrame, objetivo: pd.Series, umbral: float
) -> dict[str, float]:
    """Calcula las metricas del modelo sobre un conjunto dado."""
    probabilidades = pipeline.predict_proba(atributos)[:, 1]
    prediccion = (probabilidades >= umbral).astype(int)

    return {
        "f1": round(float(f1_score(objetivo, prediccion)), 4),
        "precision": round(float(precision_score(objetivo, prediccion, zero_division=0)), 4),
        "exhaustividad": round(float(recall_score(objetivo, prediccion)), 4),
        "exactitud_balanceada": round(float(balanced_accuracy_score(objetivo, prediccion)), 4),
        "roc_auc": round(float(roc_auc_score(objetivo, probabilidades)), 4),
    }


def medir_validacion_cruzada(
    pipeline: Pipeline, atributos: pd.DataFrame, objetivo: pd.Series
) -> dict[str, float]:
    """Reporta media y desviacion del F1 entre pliegues de validacion cruzada."""
    validacion = StratifiedKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=SEMILLA)
    puntajes = cross_val_score(pipeline, atributos, objetivo, cv=validacion, scoring="f1")

    return {
        "f1_media": round(float(puntajes.mean()), 4),
        "f1_desviacion": round(float(puntajes.std()), 4),
    }


def guardar_modelo(
    pipeline: Pipeline,
    metadatos: dict[str, Any],
    destino_modelo: Path,
    destino_metadatos: Path,
) -> None:
    """Persiste el pipeline ajustado y sus metadatos.

    El umbral forma parte de los metadatos porque sin el, quien cargue el modelo
    usaria el corte por defecto y obtendria un comportamiento distinto al evaluado.
    """
    destino_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destino_modelo)
    destino_metadatos.write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    registro.info("Modelo guardado en %s", destino_modelo)
    registro.info("Metadatos guardados en %s", destino_metadatos)


def entrenar(origen: Path, destino_modelo: Path, destino_metadatos: Path) -> dict[str, Any]:
    """Ejecuta el entrenamiento completo y devuelve los metadatos resultantes."""
    datos = cargar_atributos(origen)
    atributos, objetivo = separar_objetivo(datos)

    entrenamiento_x, prueba_x, entrenamiento_y, prueba_y = dividir_datos(atributos, objetivo)

    verificacion = comprobar_particion(entrenamiento_x, prueba_x, entrenamiento_y, prueba_y)

    pipeline = construir_pipeline()
    resultados_cv = medir_validacion_cruzada(pipeline, entrenamiento_x, entrenamiento_y)
    registro.info(
        "Validacion cruzada: F1 %.4f (desviacion %.4f)",
        resultados_cv["f1_media"],
        resultados_cv["f1_desviacion"],
    )

    umbral = optimizar_umbral(pipeline, entrenamiento_x, entrenamiento_y)

    pipeline.fit(entrenamiento_x, entrenamiento_y)
    metricas_prueba = evaluar(pipeline, prueba_x, prueba_y, umbral)
    metricas_entrenamiento = evaluar(pipeline, entrenamiento_x, entrenamiento_y, umbral)

    registro.info("Metricas en prueba: %s", metricas_prueba)

    metadatos: dict[str, Any] = {
        "familia": "logistica",
        "umbral_decision": round(umbral, 6),
        "columnas_entrada": [
            columna for columna in atributos.columns if columna != ATRIBUTO_DERIVADO
        ],
        "columnas_modelo": list(atributos.columns),
        "atributos_derivados": [ATRIBUTO_DERIVADO],
        "hiperparametros": {
            "C": REGULARIZACION,
            "class_weight": PONDERACION,
            "max_iter": MAX_ITERACIONES,
        },
        "metricas_prueba": metricas_prueba,
        "metricas_entrenamiento": metricas_entrenamiento,
        "validacion_cruzada": resultados_cv,
        "verificacion_particion": verificacion.reporte,
        "registros_entrenamiento": len(entrenamiento_x),
        "registros_prueba": len(prueba_x),
        "semilla": SEMILLA,
    }

    guardar_modelo(pipeline, metadatos, destino_modelo, destino_metadatos)
    return metadatos


def configurar_registro(detallado: bool) -> None:
    """Configura la salida de registro del script."""
    logging.basicConfig(
        level=logging.DEBUG if detallado else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def analizar_argumentos(argumentos: Sequence[str] | None) -> argparse.Namespace:
    """Interpreta los argumentos de linea de comandos."""
    analizador = argparse.ArgumentParser(description="Pipeline de entrenamiento del modelo")
    analizador.add_argument("--atributos", type=Path, default=ATRIBUTOS_POR_DEFECTO)
    analizador.add_argument("--modelo", type=Path, default=MODELO_POR_DEFECTO)
    analizador.add_argument("--metadatos", type=Path, default=METADATOS_POR_DEFECTO)
    analizador.add_argument("--detallado", action="store_true")
    return analizador.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    opciones = analizar_argumentos(argumentos)
    configurar_registro(opciones.detallado)

    try:
        entrenar(opciones.atributos, opciones.modelo, opciones.metadatos)
    except ErrorDeParticion as error:
        registro.error("%s. No se entreno ningun modelo.", error)
        return 1
    except (ValueError, FileNotFoundError):
        registro.exception("El pipeline de entrenamiento fallo")
        return 1

    registro.info("Pipeline de entrenamiento completado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
