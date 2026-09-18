"""Pruebas del pipeline de inferencia.

Las pruebas usan un modelo entrenado sobre datos sinteticos, no el modelo del
proyecto. Asi verifican el comportamiento del pipeline con independencia de la
calidad del modelo, que es responsabilidad del pipeline de entrenamiento.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import construir_atributos
from pipelines.inference_pipeline.inference_pipeline import (
    COLUMNA_DECISION,
    COLUMNA_PROBABILIDAD,
    ErrorDeInferencia,
    cargar_modelo,
    componer_salida,
    ejecutar,
    guardar_predicciones,
    leer_entrada,
    predecir,
    verificar_entrada,
)
from pipelines.training_pipeline.train_pipeline import construir_pipeline

REGISTROS = 150
SEMILLA = 5
UMBRAL_PRUEBAS = 0.5


def generar_clientes(n: int = REGISTROS) -> pd.DataFrame:
    """Genera clientes sinteticos con la estructura que espera el modelo."""
    generador = np.random.default_rng(SEMILLA)
    contrato = generador.choice(["Month-to-month", "One year", "Two year"], size=n)
    antiguedad = generador.integers(0, 73, size=n)
    cargo = generador.uniform(20, 118, size=n)

    return pd.DataFrame(
        {
            "customerID": [f"{indice:05d}-TEST" for indice in range(n)],
            "gender": generador.choice(["Female", "Male"], size=n),
            "SeniorCitizen": generador.choice(["No", "Yes"], size=n),
            "Partner": generador.choice(["No", "Yes"], size=n),
            "Dependents": generador.choice(["No", "Yes"], size=n),
            "tenure": antiguedad,
            "PhoneService": generador.choice(["No", "Yes"], size=n),
            "MultipleLines": generador.choice(["No", "Yes"], size=n),
            "InternetService": generador.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": generador.choice(["No", "Yes"], size=n),
            "OnlineBackup": generador.choice(["No", "Yes"], size=n),
            "DeviceProtection": generador.choice(["No", "Yes"], size=n),
            "TechSupport": generador.choice(["No", "Yes"], size=n),
            "StreamingTV": generador.choice(["No", "Yes"], size=n),
            "StreamingMovies": generador.choice(["No", "Yes"], size=n),
            "Contract": contrato,
            "PaperlessBilling": generador.choice(["No", "Yes"], size=n),
            "PaymentMethod": generador.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)"], size=n
            ),
            "MonthlyCharges": cargo,
            "TotalCharges": antiguedad * cargo,
        }
    )


@pytest.fixture
def clientes() -> pd.DataFrame:
    """Conjunto de clientes sin etiqueta, como llegaria en produccion."""
    return generar_clientes()


@pytest.fixture
def modelo_de_prueba(tmp_path: Path, clientes: pd.DataFrame) -> tuple[Path, Path]:
    """Entrena y serializa un modelo sobre datos sinteticos con sus metadatos."""
    atributos = construir_atributos(clientes)
    objetivo = (atributos["Contract"].astype("string") == "Month-to-month").astype(int)
    columnas_modelo = [columna for columna in atributos.columns if columna != "customerID"]

    pipeline = construir_pipeline()
    pipeline.fit(atributos[columnas_modelo], objetivo)

    ruta_modelo = tmp_path / "modelo.joblib"
    ruta_metadatos = tmp_path / "metadatos.json"
    joblib.dump(pipeline, ruta_modelo)
    ruta_metadatos.write_text(
        json.dumps(
            {
                "familia": "logistica",
                "umbral_decision": UMBRAL_PRUEBAS,
                "columnas_entrada": [
                    columna for columna in columnas_modelo if columna != "servicios_contratados"
                ],
                "columnas_modelo": columnas_modelo,
            }
        ),
        encoding="utf-8",
    )
    return ruta_modelo, ruta_metadatos


def test_cargar_modelo_falla_si_no_existe(tmp_path: Path) -> None:
    """Un modelo ausente debe producir un error explicito, no un fallo oscuro."""
    with pytest.raises(FileNotFoundError, match="Ejecute antes el entrenamiento"):
        cargar_modelo(tmp_path / "ausente.joblib", tmp_path / "ausente.json")


def test_cargar_modelo_recupera_metadatos(modelo_de_prueba: tuple[Path, Path]) -> None:
    """El umbral debe recuperarse junto con el modelo."""
    _, metadatos = cargar_modelo(*modelo_de_prueba)

    assert metadatos["umbral_decision"] == UMBRAL_PRUEBAS
    assert "columnas_modelo" in metadatos


def test_leer_entrada_acepta_csv(clientes: pd.DataFrame, tmp_path: Path) -> None:
    """La entrada en formato CSV debe leerse completa."""
    ruta = tmp_path / "clientes.csv"
    clientes.to_csv(ruta, index=False)

    assert len(leer_entrada(ruta)) == len(clientes)


def test_leer_entrada_acepta_parquet(clientes: pd.DataFrame, tmp_path: Path) -> None:
    """La entrada en formato columnar tambien debe soportarse."""
    ruta = tmp_path / "clientes.parquet"
    clientes.to_parquet(ruta, index=False)

    assert len(leer_entrada(ruta)) == len(clientes)


def test_verificar_entrada_detecta_columnas_faltantes(clientes: pd.DataFrame) -> None:
    """Sin las columnas que el modelo espera no se puede predecir."""
    incompleto = clientes.drop(columns=["Contract"])

    with pytest.raises(ErrorDeInferencia, match="Faltan columnas"):
        verificar_entrada(incompleto, ["Contract", "tenure"])


def test_verificar_entrada_rechaza_archivo_vacio(clientes: pd.DataFrame) -> None:
    """Un archivo sin registros es un error de entrada, no un resultado vacio."""
    with pytest.raises(ErrorDeInferencia, match="no contiene registros"):
        verificar_entrada(clientes.iloc[:0], ["Contract"])


def test_predecir_devuelve_probabilidades_validas(
    clientes: pd.DataFrame, modelo_de_prueba: tuple[Path, Path]
) -> None:
    """Las probabilidades deben estar acotadas entre cero y uno."""
    modelo, metadatos = cargar_modelo(*modelo_de_prueba)
    atributos = construir_atributos(clientes)

    resultado = predecir(modelo, atributos, metadatos["columnas_modelo"], UMBRAL_PRUEBAS)

    assert len(resultado) == len(clientes)
    assert resultado[COLUMNA_PROBABILIDAD].between(0, 1).all()
    assert resultado[COLUMNA_DECISION].dtype == bool


def test_decision_respeta_el_umbral(
    clientes: pd.DataFrame, modelo_de_prueba: tuple[Path, Path]
) -> None:
    """La decision debe derivarse del umbral, no del corte por defecto."""
    modelo, metadatos = cargar_modelo(*modelo_de_prueba)
    atributos = construir_atributos(clientes)

    exigente = predecir(modelo, atributos, metadatos["columnas_modelo"], 0.9)
    permisivo = predecir(modelo, atributos, metadatos["columnas_modelo"], 0.1)

    assert exigente[COLUMNA_DECISION].sum() <= permisivo[COLUMNA_DECISION].sum()


def test_salida_ordenada_por_riesgo(clientes: pd.DataFrame) -> None:
    """El resultado debe venir ordenado para priorizar el contacto."""
    predicciones = pd.DataFrame(
        {
            COLUMNA_PROBABILIDAD: [0.1, 0.9, 0.5],
            COLUMNA_DECISION: [False, True, False],
        },
        index=[0, 1, 2],
    )

    salida = componer_salida(clientes.iloc[:3], predicciones)

    assert list(salida[COLUMNA_PROBABILIDAD]) == [0.9, 0.5, 0.1]
    assert "customerID" in salida.columns


def test_guardar_predicciones_crea_directorio(tmp_path: Path) -> None:
    """El destino debe crearse aunque el directorio no exista."""
    predicciones = pd.DataFrame({COLUMNA_PROBABILIDAD: [0.5], COLUMNA_DECISION: [False]})
    destino = guardar_predicciones(predicciones, tmp_path / "nuevo" / "salida.csv")

    assert destino.exists()
    assert len(pd.read_csv(destino)) == 1


def test_ejecucion_completa_produce_archivo(
    clientes: pd.DataFrame, modelo_de_prueba: tuple[Path, Path], tmp_path: Path
) -> None:
    """La inferencia de punta a punta debe dejar las predicciones en disco."""
    entrada = tmp_path / "entrada.csv"
    salida = tmp_path / "salida.csv"
    clientes.to_csv(entrada, index=False)

    resultado = ejecutar(entrada, salida, *modelo_de_prueba)

    assert salida.exists()
    assert len(resultado) == len(clientes)
    assert COLUMNA_PROBABILIDAD in pd.read_csv(salida).columns


def test_ejecucion_tolera_columnas_adicionales(
    clientes: pd.DataFrame, modelo_de_prueba: tuple[Path, Path], tmp_path: Path
) -> None:
    """Un archivo de produccion puede traer columnas extra sin romper la inferencia."""
    entrada = tmp_path / "entrada.csv"
    salida = tmp_path / "salida.csv"
    con_extras = clientes.assign(segmento="premium", fecha_carga="2026-01-01")
    con_extras.to_csv(entrada, index=False)

    resultado = ejecutar(entrada, salida, *modelo_de_prueba)

    assert len(resultado) == len(clientes)


def test_ejecucion_falla_con_entrada_incompleta(
    clientes: pd.DataFrame, modelo_de_prueba: tuple[Path, Path], tmp_path: Path
) -> None:
    """Sin las columnas requeridas no debe generarse ningun archivo de salida."""
    entrada = tmp_path / "entrada.csv"
    salida = tmp_path / "salida.csv"
    clientes.drop(columns=["Contract"]).to_csv(entrada, index=False)

    with pytest.raises(ErrorDeInferencia):
        ejecutar(entrada, salida, *modelo_de_prueba)

    assert not salida.exists()
