"""Pipeline de inferencia sobre datos nuevos.

Carga el modelo entrenado, aplica a los datos de entrada exactamente las mismas
transformaciones que se usaron al entrenar y genera las predicciones.

La garantia de equivalencia entre entrenamiento e inferencia no descansa en la
disciplina de quien escribe el codigo, sino en la estructura: las transformaciones se
importan del pipeline de atributos, y el preprocesamiento ajustado viaja dentro del
objeto serializado. No hay ninguna transformacion definida en este modulo que pueda
divergir de la del entrenamiento.

El umbral de decision se lee de los metadatos del modelo. Sin el, quien ejecute la
inferencia usaria el corte por defecto de 0.5 y obtendria un comportamiento distinto
al que se valido.

Ejecucion:

    uv run python src/pipelines/inference_pipeline/inference_pipeline.py \\
        --entrada clientes.csv --salida predicciones.csv
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
import pandas as pd

DIRECTORIO_FUENTES = Path(__file__).resolve().parents[2]
if str(DIRECTORIO_FUENTES) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_FUENTES))

from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    IDENTIFICADOR,
    OBJETIVO,
    construir_atributos,
)

RAIZ = Path(__file__).resolve().parents[3]
MODELO_POR_DEFECTO = RAIZ / "models" / "modelo_churn.joblib"
METADATOS_POR_DEFECTO = RAIZ / "models" / "modelo_churn_metadatos.json"
ENTRADA_POR_DEFECTO = RAIZ / "app" / "ejemplos" / "clientes_ejemplo.csv"
SALIDA_POR_DEFECTO = RAIZ / "data" / "07_model_output" / "predicciones.csv"

COLUMNA_PROBABILIDAD = "probabilidad_cancelacion"
COLUMNA_DECISION = "en_riesgo"

registro = logging.getLogger(__name__)


class ErrorDeInferencia(Exception):
    """Se levanta cuando los datos de entrada no permiten predecir."""


def cargar_modelo(ruta_modelo: Path, ruta_metadatos: Path) -> tuple[Any, dict[str, Any]]:
    """Recupera el pipeline entrenado y sus metadatos."""
    if not ruta_modelo.exists():
        mensaje = f"No existe el modelo {ruta_modelo}. Ejecute antes el entrenamiento."
        raise FileNotFoundError(mensaje)

    modelo = joblib.load(ruta_modelo)
    metadatos: dict[str, Any] = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
    registro.info(
        "Modelo cargado (%s, umbral %.4f)",
        metadatos.get("familia", "desconocido"),
        metadatos["umbral_decision"],
    )
    return modelo, metadatos


def leer_entrada(ruta: Path) -> pd.DataFrame:
    """Lee los datos sobre los que se va a predecir."""
    if not ruta.exists():
        mensaje = f"No existe el archivo de entrada {ruta}"
        raise FileNotFoundError(mensaje)

    datos = pd.read_parquet(ruta) if ruta.suffix == ".parquet" else pd.read_csv(ruta)
    registro.info("Entrada: %d registros, %d columnas", len(datos), datos.shape[1])
    return datos


def verificar_entrada(datos: pd.DataFrame, requeridas: Sequence[str]) -> None:
    """Comprueba que estan todas las columnas que el modelo necesita."""
    faltantes = [columna for columna in requeridas if columna not in datos.columns]
    if faltantes:
        mensaje = f"Faltan columnas requeridas por el modelo: {faltantes}"
        raise ErrorDeInferencia(mensaje)

    if datos.empty:
        mensaje = "El archivo de entrada no contiene registros"
        raise ErrorDeInferencia(mensaje)


def predecir(
    modelo: Any, datos: pd.DataFrame, columnas_modelo: Sequence[str], umbral: float
) -> pd.DataFrame:
    """Genera probabilidades y decisiones para cada registro.

    Se seleccionan explicitamente las columnas que el modelo espera, y en su orden.
    Un archivo de produccion puede traer columnas adicionales o en otro orden sin que
    eso altere el resultado.
    """
    probabilidades = modelo.predict_proba(datos[list(columnas_modelo)])[:, 1]

    return pd.DataFrame(
        {
            COLUMNA_PROBABILIDAD: probabilidades.round(4),
            COLUMNA_DECISION: probabilidades >= umbral,
        },
        index=datos.index,
    )


def componer_salida(originales: pd.DataFrame, predicciones: pd.DataFrame) -> pd.DataFrame:
    """Une las predicciones a los datos de entrada y ordena por riesgo.

    El orden descendente por probabilidad es el orden en que conviene contactar a los
    clientes: el valor operativo del modelo esta en la priorizacion, no en la
    clasificacion binaria.
    """
    columnas_contexto = [
        columna
        for columna in (IDENTIFICADOR, "Contract", "tenure", "MonthlyCharges")
        if columna in originales.columns
    ]

    salida = pd.concat([originales[columnas_contexto], predicciones], axis=1)
    return salida.sort_values(COLUMNA_PROBABILIDAD, ascending=False)


def guardar_predicciones(predicciones: pd.DataFrame, destino: Path) -> Path:
    """Persiste las predicciones en disco."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    predicciones.to_csv(destino, index=False)
    registro.info("Predicciones guardadas en %s", destino)
    return destino


def ejecutar(
    entrada: Path,
    salida: Path,
    ruta_modelo: Path = MODELO_POR_DEFECTO,
    ruta_metadatos: Path = METADATOS_POR_DEFECTO,
) -> pd.DataFrame:
    """Ejecuta la inferencia completa, desde el archivo de entrada hasta la salida."""
    modelo, metadatos = cargar_modelo(ruta_modelo, ruta_metadatos)
    columnas_entrada: list[str] = metadatos["columnas_entrada"]
    columnas_modelo: list[str] = metadatos["columnas_modelo"]
    umbral = float(metadatos["umbral_decision"])

    datos = leer_entrada(entrada)
    verificar_entrada(datos, columnas_entrada)

    if OBJETIVO in datos.columns:
        registro.warning(
            "La entrada contiene la columna objetivo '%s'. Se ignora para predecir.",
            OBJETIVO,
        )

    atributos = construir_atributos(datos)
    predicciones = predecir(modelo, atributos, columnas_modelo, umbral)
    resultado = componer_salida(datos, predicciones)

    en_riesgo = int(resultado[COLUMNA_DECISION].sum())
    registro.info(
        "Predicciones generadas: %d registros, %d en riesgo (%.1f%%)",
        len(resultado),
        en_riesgo,
        en_riesgo / len(resultado) * 100,
    )

    guardar_predicciones(resultado, salida)
    return resultado


def configurar_registro(detallado: bool) -> None:
    """Configura la salida de registro del script."""
    logging.basicConfig(
        level=logging.DEBUG if detallado else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def analizar_argumentos(argumentos: Sequence[str] | None) -> argparse.Namespace:
    """Interpreta los argumentos de linea de comandos."""
    analizador = argparse.ArgumentParser(description="Pipeline de inferencia")
    analizador.add_argument("--entrada", type=Path, default=ENTRADA_POR_DEFECTO)
    analizador.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    analizador.add_argument("--modelo", type=Path, default=MODELO_POR_DEFECTO)
    analizador.add_argument("--metadatos", type=Path, default=METADATOS_POR_DEFECTO)
    analizador.add_argument("--detallado", action="store_true")
    return analizador.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    opciones = analizar_argumentos(argumentos)
    configurar_registro(opciones.detallado)

    try:
        ejecutar(opciones.entrada, opciones.salida, opciones.modelo, opciones.metadatos)
    except ErrorDeInferencia as error:
        registro.error("%s. No se generaron predicciones.", error)
        return 1
    except (ValueError, FileNotFoundError, KeyError):
        registro.exception("El pipeline de inferencia fallo")
        return 1

    registro.info("Pipeline de inferencia completado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
