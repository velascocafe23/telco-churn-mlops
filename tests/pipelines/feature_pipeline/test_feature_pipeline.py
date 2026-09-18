"""Pruebas del pipeline de construccion de atributos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import (
    ATRIBUTO_DERIVADO,
    COLUMNAS_NUMERICAS,
    agregar_atributos_derivados,
    construir_atributos,
    contar_servicios,
    convertir_tipos,
    ejecutar,
    guardar_atributos,
    unificar_nulos,
    verificar_columnas,
)

CLIENTES_ESPERADOS = 3
SERVICIOS_CLIENTE_COMPLETO = 9
SERVICIOS_CLIENTE_SOLO_TELEFONO = 1
SERVICIOS_CLIENTE_MEDIO = 3


@pytest.fixture
def datos_crudos() -> pd.DataFrame:
    """Conjunto sintetico con los problemas de calidad de la fuente original.

    Incluye deliberadamente: TotalCharges como texto con un valor en blanco,
    SeniorCitizen como entero binario y espacios sobrantes en una categoria.
    """
    return pd.DataFrame(
        {
            "customerID": ["0001-AAAA", "0002-BBBB", "0003-CCCC"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [1, 0, 36],
            "PhoneService": ["Yes", "Yes", "Yes"],
            "MultipleLines": ["Yes", "No", "No"],
            "InternetService": ["Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["Yes", "No internet service", "Yes"],
            "OnlineBackup": ["Yes", "No internet service", "No"],
            "DeviceProtection": ["Yes", "No internet service", "No"],
            "TechSupport": ["Yes", "No internet service", "No"],
            "StreamingTV": ["Yes", "No internet service", "No"],
            "StreamingMovies": ["Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "Month-to-month", " Two year "],
            "PaperlessBilling": ["Yes", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [105.5, 20.25, 55.0],
            "TotalCharges": ["105.5", " ", "1980.0"],
            "Churn": ["Yes", "No", "No"],
        }
    )


def test_unificar_nulos_convierte_blancos(datos_crudos: pd.DataFrame) -> None:
    """Las cadenas en blanco deben quedar como valores nulos."""
    resultado = unificar_nulos(datos_crudos)

    assert resultado["TotalCharges"].isna().sum() == 1
    assert resultado.loc[1, "TotalCharges"] is pd.NA


def test_unificar_nulos_recorta_espacios(datos_crudos: pd.DataFrame) -> None:
    """Los espacios sobrantes deben eliminarse para no duplicar categorias."""
    resultado = unificar_nulos(datos_crudos)

    assert resultado.loc[2, "Contract"] == "Two year"


def test_convertir_tipos_homologa_adulto_mayor(datos_crudos: pd.DataFrame) -> None:
    """SeniorCitizen debe pasar de entero binario a las mismas etiquetas del resto."""
    resultado = convertir_tipos(unificar_nulos(datos_crudos))

    assert set(resultado["SeniorCitizen"].dropna().unique()) <= {"Yes", "No"}


def test_convertir_tipos_produce_numericos(datos_crudos: pd.DataFrame) -> None:
    """Las tres magnitudes deben quedar como numericas."""
    resultado = convertir_tipos(unificar_nulos(datos_crudos))

    for columna in COLUMNAS_NUMERICAS:
        assert pd.api.types.is_numeric_dtype(resultado[columna])


def test_contar_servicios_cliente_completo(datos_crudos: pd.DataFrame) -> None:
    """Un cliente con todos los servicios debe sumar ocho opcionales mas internet."""
    conteo = contar_servicios(datos_crudos)

    assert conteo.iloc[0] == SERVICIOS_CLIENTE_COMPLETO


def test_contar_servicios_ignora_ausencia_de_servicio_base(
    datos_crudos: pd.DataFrame,
) -> None:
    """Los valores que indican ausencia del servicio base no deben contar."""
    conteo = contar_servicios(datos_crudos)

    assert conteo.iloc[1] == SERVICIOS_CLIENTE_SOLO_TELEFONO
    assert conteo.iloc[2] == SERVICIOS_CLIENTE_MEDIO


def test_agregar_atributos_derivados_no_altera_originales(
    datos_crudos: pd.DataFrame,
) -> None:
    """El atributo derivado se anade sin modificar las columnas existentes."""
    resultado = agregar_atributos_derivados(datos_crudos)

    assert ATRIBUTO_DERIVADO in resultado.columns
    assert resultado.shape[1] == datos_crudos.shape[1] + 1
    assert list(datos_crudos.columns) == list(resultado.columns)[:-1]


def test_construir_atributos_end_to_end(datos_crudos: pd.DataFrame) -> None:
    """La secuencia completa produce tipos correctos y el atributo derivado."""
    resultado = construir_atributos(datos_crudos)

    assert len(resultado) == CLIENTES_ESPERADOS
    assert ATRIBUTO_DERIVADO in resultado.columns
    assert pd.api.types.is_numeric_dtype(resultado["TotalCharges"])
    assert resultado["Contract"].dtype == "category"


def test_verificar_columnas_detecta_faltantes(datos_crudos: pd.DataFrame) -> None:
    """La verificacion debe reportar las columnas ausentes."""
    incompleto = datos_crudos.drop(columns=["Contract"])

    assert verificar_columnas(incompleto, ["Contract", "tenure"]) == ["Contract"]
    assert verificar_columnas(datos_crudos, ["Contract", "tenure"]) == []


def test_guardar_atributos_conserva_registros(datos_crudos: pd.DataFrame, tmp_path: Path) -> None:
    """El archivo persistido debe recuperarse con el mismo numero de registros."""
    destino = tmp_path / "salida" / "atributos.parquet"
    atributos = construir_atributos(datos_crudos)

    guardar_atributos(atributos, destino)
    recuperado = pd.read_parquet(destino)

    assert destino.exists()
    assert len(recuperado) == len(atributos)
    assert ATRIBUTO_DERIVADO in recuperado.columns


def test_ejecutar_falla_con_columnas_faltantes(datos_crudos: pd.DataFrame, tmp_path: Path) -> None:
    """El pipeline debe detenerse si el origen no tiene las columnas requeridas."""
    origen = tmp_path / "incompleto.csv"
    datos_crudos.drop(columns=["Contract"]).to_csv(origen, index=False)

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        ejecutar(origen, tmp_path / "salida.parquet")


def test_ejecutar_procesa_archivo_completo(datos_crudos: pd.DataFrame, tmp_path: Path) -> None:
    """El pipeline completo lee, transforma y persiste sin intervencion."""
    origen = tmp_path / "crudos.csv"
    destino = tmp_path / "atributos.parquet"
    datos_crudos.to_csv(origen, index=False)

    resultado = ejecutar(origen, destino)

    assert destino.exists()
    assert len(resultado) == CLIENTES_ESPERADOS
    assert ATRIBUTO_DERIVADO in resultado.columns
