"""Pipeline de construccion de atributos.

Convierte los datos crudos de clientes en el conjunto de atributos que consumen el
pipeline de entrenamiento y el de inferencia.

Este modulo es la unica fuente de verdad sobre como se construyen los atributos. El
entrenamiento, la inferencia por lote y la aplicacion de despliegue lo importan, de
modo que las transformaciones aplicadas en produccion son necesariamente las mismas
del entrenamiento.

Antes de persistir, los atributos pasan por las validaciones de `validacion.py`. Si
alguna regla obligatoria falla, el pipeline se detiene y no escribe el archivo de
salida.

Ejecucion:

    uv run python src/pipelines/feature_pipeline/feature_pipeline.py
    uv run python src/pipelines/feature_pipeline/feature_pipeline.py --origen datos.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

# Permite ejecutar este archivo directamente como script. Al hacerlo, Python solo
# anade al path el directorio del propio archivo, no la raiz de `src`, de modo que
# el import absoluto no resolveria. Bajo pytest esto es redundante, porque la
# configuracion del proyecto ya declara `src` en pythonpath.
DIRECTORIO_FUENTES = Path(__file__).resolve().parents[2]
if str(DIRECTORIO_FUENTES) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_FUENTES))

from pipelines.feature_pipeline.validacion import (  # noqa: E402
    ErrorDeValidacion,
    ResultadoValidacion,
    validar,
)

RAIZ = Path(__file__).resolve().parents[3]
ORIGEN_POR_DEFECTO = RAIZ / "data" / "01_raw" / "telco_customer_churn.csv"
DESTINO_POR_DEFECTO = RAIZ / "data" / "04_feature" / "telco_features.parquet"

URL_FUENTE = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

IDENTIFICADOR = "customerID"
OBJETIVO = "Churn"
CLASE_POSITIVA = "Yes"

SI = "Yes"
NO = "No"
SIN_INTERNET = "No internet service"
SIN_TELEFONO = "No phone service"

MARCADORES_NULOS = ["", " ", "NA", "N/A", "na", "null", "NULL", "?", "-"]

COLUMNAS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

COLUMNAS_CATEGORICAS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

SERVICIOS_OPCIONALES = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ATRIBUTO_DERIVADO = "servicios_contratados"

COLUMNAS_ENTRADA = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS
COLUMNAS_SALIDA = [*COLUMNAS_ENTRADA, ATRIBUTO_DERIVADO]

registro = logging.getLogger(__name__)


def cargar_datos(origen: Path) -> pd.DataFrame:
    """Lee los datos crudos desde un archivo CSV local o desde la fuente original."""
    if origen.exists():
        registro.info("Leyendo datos locales desde %s", origen)
        return pd.read_csv(origen)

    registro.info("Archivo local ausente, descargando desde la fuente original")
    datos = pd.read_csv(URL_FUENTE)
    origen.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(origen, index=False)
    registro.info("Copia guardada en %s", origen)
    return datos


def unificar_nulos(datos: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza por nulos las formas textuales de ausencia y recorta espacios.

    Los datos de origen no traen nulos declarados: los ausentes llegan como cadenas
    en blanco. Sin esta unificacion, una columna numerica se leeria como texto.
    """
    resultado = datos.copy()

    for columna in resultado.select_dtypes(include=["object", "string"]).columns:
        serie = resultado[columna].astype("string").str.strip()
        resultado[columna] = serie.mask(serie.isin(MARCADORES_NULOS), pd.NA)

    return resultado


def convertir_tipos(datos: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada columna su tipo correcto.

    `SeniorCitizen` llega como entero binario en la fuente original y como texto en
    los datos ya procesados. Se homologa a texto para que el tratamiento sea igual
    al del resto de atributos binarios.
    """
    resultado = datos.copy()

    if "SeniorCitizen" in resultado.columns and pd.api.types.is_numeric_dtype(
        resultado["SeniorCitizen"]
    ):
        resultado["SeniorCitizen"] = resultado["SeniorCitizen"].map({0: NO, 1: SI})

    for columna in COLUMNAS_NUMERICAS:
        if columna in resultado.columns:
            resultado[columna] = pd.to_numeric(resultado[columna], errors="coerce")

    for columna in COLUMNAS_CATEGORICAS:
        if columna in resultado.columns:
            resultado[columna] = resultado[columna].astype("category")

    if IDENTIFICADOR in resultado.columns:
        resultado[IDENTIFICADOR] = resultado[IDENTIFICADOR].astype("string")

    return resultado


def contar_servicios(datos: pd.DataFrame) -> pd.Series:
    """Cuenta cuantos servicios tiene contratados cada cliente.

    Mide el grado de vinculacion del cliente con la compania. Se contabiliza el
    servicio de internet cuando existe, mas cada servicio opcional con valor
    afirmativo. Los valores `No internet service` y `No phone service` no cuentan,
    porque indican ausencia del servicio base.
    """
    conteo = pd.Series(0, index=datos.index, dtype="int64")

    for columna in SERVICIOS_OPCIONALES:
        if columna in datos.columns:
            conteo = conteo + (datos[columna].astype("string") == SI).astype("int64")

    if "InternetService" in datos.columns:
        tiene_internet = (datos["InternetService"].astype("string") != NO).astype("int64")
        conteo = conteo + tiene_internet

    return conteo


def agregar_atributos_derivados(datos: pd.DataFrame) -> pd.DataFrame:
    """Anade al conjunto los atributos calculados a partir de los originales."""
    resultado = datos.copy()
    resultado[ATRIBUTO_DERIVADO] = contar_servicios(resultado)
    return resultado


def construir_atributos(datos: pd.DataFrame) -> pd.DataFrame:
    """Aplica la secuencia completa de construccion de atributos.

    Es el punto de entrada que deben usar el entrenamiento, la inferencia y la
    aplicacion de despliegue. La imputacion, el escalado y la codificacion no se
    hacen aqui: pertenecen al pipeline de scikit-learn que viaja serializado junto
    al modelo, para que se ajusten sobre el conjunto de entrenamiento y no sobre
    los datos completos.
    """
    resultado = unificar_nulos(datos)
    resultado = convertir_tipos(resultado)
    return agregar_atributos_derivados(resultado)


def verificar_columnas(datos: pd.DataFrame, requeridas: Sequence[str]) -> list[str]:
    """Devuelve las columnas requeridas que no estan presentes."""
    return [columna for columna in requeridas if columna not in datos.columns]


def aplicar_validaciones(atributos: pd.DataFrame) -> ResultadoValidacion:
    """Valida los atributos y detiene el pipeline si incumplen las reglas.

    Las advertencias se registran y permiten continuar. Los errores impiden la
    persistencia: es preferible no producir un archivo a producir uno que las
    etapas siguientes consumirian como si fuera correcto.
    """
    resultado = validar(atributos)

    for advertencia in resultado.advertencias:
        registro.warning(advertencia)

    if not resultado.valido:
        for error in resultado.errores:
            registro.error(error)
        mensaje = f"Los datos no superaron la validacion ({len(resultado.errores)} errores)"
        raise ErrorDeValidacion(mensaje)

    registro.info("Validacion superada: esquema, integridad y distribucion")
    return resultado


def guardar_atributos(datos: pd.DataFrame, destino: Path) -> Path:
    """Persiste los atributos en formato columnar, conservando los tipos."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    datos.to_parquet(destino, index=False)
    registro.info("Atributos guardados en %s (%d registros)", destino, len(datos))
    return destino


def ejecutar(origen: Path, destino: Path) -> pd.DataFrame:
    """Ejecuta el pipeline completo, desde los datos crudos hasta el archivo final."""
    crudos = cargar_datos(origen)
    registro.info("Datos crudos: %d registros, %d columnas", len(crudos), crudos.shape[1])

    faltantes = verificar_columnas(crudos, COLUMNAS_ENTRADA)
    if faltantes:
        mensaje = f"Faltan columnas requeridas en el origen: {faltantes}"
        raise ValueError(mensaje)

    atributos = construir_atributos(crudos)
    registro.info("Atributos construidos: %d columnas", atributos.shape[1])

    aplicar_validaciones(atributos)

    guardar_atributos(atributos, destino)
    return atributos


def configurar_registro(detallado: bool) -> None:
    """Configura la salida de registro del script."""
    logging.basicConfig(
        level=logging.DEBUG if detallado else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def analizar_argumentos(argumentos: Sequence[str] | None) -> argparse.Namespace:
    """Interpreta los argumentos de linea de comandos."""
    analizador = argparse.ArgumentParser(description="Pipeline de construccion de atributos")
    analizador.add_argument(
        "--origen",
        type=Path,
        default=ORIGEN_POR_DEFECTO,
        help="Ruta del archivo CSV de datos crudos",
    )
    analizador.add_argument(
        "--destino",
        type=Path,
        default=DESTINO_POR_DEFECTO,
        help="Ruta del archivo parquet de salida",
    )
    analizador.add_argument(
        "--detallado",
        action="store_true",
        help="Aumenta el nivel de detalle del registro",
    )
    return analizador.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> int:
    """Punto de entrada del script."""
    opciones = analizar_argumentos(argumentos)
    configurar_registro(opciones.detallado)

    try:
        ejecutar(opciones.origen, opciones.destino)
    except ErrorDeValidacion as error:
        # Los mensajes de cada regla incumplida ya se registraron. La traza no
        # aportaria nada: el fallo es del dato, no del codigo.
        registro.error("%s. No se genero archivo de salida.", error)
        return 1
    except (ValueError, FileNotFoundError):
        registro.exception("El pipeline de atributos fallo")
        return 1

    registro.info("Pipeline de atributos completado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
