"""Pruebas de las validaciones de datos e integridad."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import construir_atributos, ejecutar
from pipelines.feature_pipeline.validacion import (
    ErrorDeValidacion,
    validar,
    validar_acumulado_solo_nulo_sin_antiguedad,
    validar_esquema,
    validar_integridad,
    validar_servicios_de_internet,
    validar_servicios_de_telefonia,
)

REGISTROS_BASE = 3
ERRORES_DE_INTEGRIDAD_ESPERADOS = 2


def _cliente(**cambios: object) -> dict[str, object]:
    """Construye un cliente valido, con los campos indicados sobrescritos."""
    base: dict[str, object] = {
        "customerID": "0001-AAAA",
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 60.0,
        "TotalCharges": 1440.0,
        "Churn": "No",
        "servicios_contratados": 4,
    }
    base.update(cambios)
    return base


@pytest.fixture
def atributos_validos() -> pd.DataFrame:
    """Conjunto minimo que cumple todas las reglas."""
    return pd.DataFrame(
        [
            _cliente(),
            _cliente(customerID="0002-BBBB", Contract="Two year", Churn="No"),
            _cliente(
                customerID="0003-CCCC",
                Contract="Month-to-month",
                tenure=2,
                TotalCharges=120.0,
                Churn="Yes",
            ),
        ]
    )


def test_conjunto_valido_no_produce_errores(atributos_validos: pd.DataFrame) -> None:
    """Un conjunto correcto debe pasar todas las validaciones obligatorias."""
    resultado = validar(atributos_validos)

    assert resultado.valido
    assert resultado.errores == []


def test_categoria_desconocida_es_error(atributos_validos: pd.DataFrame) -> None:
    """Un valor fuera del conjunto cerrado de categorias debe detectarse."""
    invalido = atributos_validos.copy()
    invalido.loc[0, "Contract"] = "Contrato mensual"

    errores = validar_esquema(invalido)

    assert len(errores) == 1
    assert "Contract" in errores[0]


def test_antiguedad_fuera_de_rango_es_error(atributos_validos: pd.DataFrame) -> None:
    """La antiguedad no puede superar el maximo observado en la poblacion."""
    invalido = atributos_validos.copy()
    invalido.loc[0, "tenure"] = 200

    errores = validar_esquema(invalido)

    assert any("tenure" in mensaje for mensaje in errores)


def test_identificador_repetido_es_error(atributos_validos: pd.DataFrame) -> None:
    """El identificador de cliente debe ser unico en todo el lote."""
    invalido = atributos_validos.copy()
    invalido.loc[1, "customerID"] = invalido.loc[0, "customerID"]

    errores = validar_esquema(invalido)

    assert any("customerID" in mensaje for mensaje in errores)


def test_nulo_en_acumulado_sin_antiguedad_es_valido(
    atributos_validos: pd.DataFrame,
) -> None:
    """Un cliente sin primera factura puede tener el acumulado ausente."""
    nuevo = atributos_validos.copy()
    nuevo.loc[0, "tenure"] = 0
    nuevo.loc[0, "TotalCharges"] = None

    assert validar_acumulado_solo_nulo_sin_antiguedad(nuevo) == []


def test_nulo_en_acumulado_con_antiguedad_es_error(
    atributos_validos: pd.DataFrame,
) -> None:
    """Un acumulado ausente con antiguedad positiva indica fallo de facturacion."""
    invalido = atributos_validos.copy()
    invalido.loc[0, "TotalCharges"] = None

    errores = validar_acumulado_solo_nulo_sin_antiguedad(invalido)

    assert len(errores) == 1
    assert "antiguedad mayor que cero" in errores[0]


def test_incoherencia_de_telefonia_es_error(atributos_validos: pd.DataFrame) -> None:
    """Sin servicio telefonico, las lineas multiples deben reflejarlo."""
    invalido = atributos_validos.copy()
    invalido.loc[0, "PhoneService"] = "No"

    errores = validar_servicios_de_telefonia(invalido)

    assert len(errores) == 1
    assert "MultipleLines" in errores[0]


def test_incoherencia_de_internet_es_error(atributos_validos: pd.DataFrame) -> None:
    """Sin servicio de internet, los complementos deben declarar esa ausencia."""
    invalido = atributos_validos.copy()
    invalido.loc[0, "InternetService"] = "No"

    errores = validar_servicios_de_internet(invalido)

    assert errores
    assert any("OnlineSecurity" in mensaje for mensaje in errores)


def test_integridad_agrupa_todas_las_reglas(atributos_validos: pd.DataFrame) -> None:
    """La verificacion de integridad debe reunir los fallos de todas las reglas."""
    invalido = atributos_validos.copy()
    invalido.loc[0, "PhoneService"] = "No"
    invalido.loc[1, "TotalCharges"] = None

    errores = validar_integridad(invalido)

    assert len(errores) >= ERRORES_DE_INTEGRIDAD_ESPERADOS


def test_distribucion_produce_advertencia_no_error(
    atributos_validos: pd.DataFrame,
) -> None:
    """Una tasa de cancelacion atipica advierte pero no detiene el pipeline."""
    desviado = atributos_validos.copy()
    desviado["Churn"] = "Yes"

    resultado = validar(desviado)

    assert resultado.valido
    assert resultado.advertencias


def test_datos_reales_pasan_validacion() -> None:
    """Los datos del proyecto deben superar todas las validaciones obligatorias."""
    origen = Path("data/01_raw/telco_customer_churn.csv")
    if not origen.exists():
        pytest.skip("El archivo de datos crudos no esta disponible")

    atributos = construir_atributos(pd.read_csv(origen))
    resultado = validar(atributos)

    assert resultado.valido, f"Errores inesperados: {resultado.errores}"


def test_pipeline_no_persiste_si_la_validacion_falla(tmp_path: Path) -> None:
    """Ante un fallo de validacion, el archivo de salida no debe crearse."""
    invalido = pd.DataFrame([_cliente(Contract="Mensual")]).drop(columns=["servicios_contratados"])
    origen = tmp_path / "invalido.csv"
    destino = tmp_path / "salida.parquet"
    invalido.to_csv(origen, index=False)

    with pytest.raises(ErrorDeValidacion):
        ejecutar(origen, destino)

    assert not destino.exists()


def test_pipeline_persiste_con_datos_validos(
    atributos_validos: pd.DataFrame, tmp_path: Path
) -> None:
    """Con datos correctos, el pipeline completa y escribe el archivo."""
    origen = tmp_path / "valido.csv"
    destino = tmp_path / "salida.parquet"
    atributos_validos.drop(columns=["servicios_contratados"]).to_csv(origen, index=False)

    resultado = ejecutar(origen, destino)

    assert destino.exists()
    assert len(resultado) == REGISTROS_BASE
