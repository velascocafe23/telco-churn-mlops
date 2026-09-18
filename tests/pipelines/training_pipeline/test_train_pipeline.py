"""Pruebas del pipeline de entrenamiento."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pipelines.training_pipeline.train_pipeline import (
    cargar_atributos,
    construir_pipeline,
    construir_preprocesador,
    dividir_datos,
    entrenar,
    evaluar,
    medir_validacion_cruzada,
    optimizar_umbral,
    separar_objetivo,
)

REGISTROS_SINTETICOS = 400
ANTIGUEDAD_DE_RIESGO = 12
SEMILLA_PRUEBAS = 7
UMBRAL_MINIMO = 0.0
UMBRAL_MAXIMO = 1.0
PROPORCION_PRUEBA_ESPERADA = 0.2
TOLERANCIA_PROPORCION = 0.02
METRICAS_ESPERADAS = {
    "f1",
    "precision",
    "exhaustividad",
    "exactitud_balanceada",
    "roc_auc",
}


def generar_atributos(n: int = REGISTROS_SINTETICOS) -> pd.DataFrame:
    """Genera un conjunto sintetico con la estructura de los atributos reales.

    El objetivo depende del contrato y la antiguedad, de modo que exista senal
    aprendible y las metricas no sean puro azar.
    """
    generador = np.random.default_rng(SEMILLA_PRUEBAS)

    contrato = generador.choice(["Month-to-month", "One year", "Two year"], size=n)
    antiguedad = generador.integers(0, 73, size=n)
    cargo_mensual = generador.uniform(20, 118, size=n)

    riesgo = (contrato == "Month-to-month").astype(float) * 0.5 + (
        antiguedad < ANTIGUEDAD_DE_RIESGO
    ) * 0.3
    objetivo = generador.random(n) < riesgo

    return pd.DataFrame(
        {
            "customerID": [f"{indice:05d}-XXXX" for indice in range(n)],
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
            "MonthlyCharges": cargo_mensual,
            "TotalCharges": antiguedad * cargo_mensual,
            "servicios_contratados": generador.integers(1, 10, size=n),
            "Churn": np.where(objetivo, "Yes", "No"),
        }
    )


@pytest.fixture
def atributos_sinteticos() -> pd.DataFrame:
    """Conjunto sintetico reutilizable entre pruebas."""
    return generar_atributos()


def test_separar_objetivo_descarta_identificador(atributos_sinteticos: pd.DataFrame) -> None:
    """El identificador y la etiqueta no deben quedar entre los atributos."""
    atributos, objetivo = separar_objetivo(atributos_sinteticos)

    assert "customerID" not in atributos.columns
    assert "Churn" not in atributos.columns
    assert set(objetivo.unique()) <= {0, 1}


def test_separar_objetivo_falla_sin_etiqueta(atributos_sinteticos: pd.DataFrame) -> None:
    """Sin columna objetivo el entrenamiento no tiene sentido y debe fallar."""
    with pytest.raises(ValueError, match="columna objetivo"):
        separar_objetivo(atributos_sinteticos.drop(columns=["Churn"]))


def test_preprocesador_incluye_atributo_derivado() -> None:
    """El atributo derivado debe entrar por la rama numerica."""
    preprocesador = construir_preprocesador()
    nombres, _, columnas = preprocesador.transformers[0]

    assert nombres == "numericas"
    assert "servicios_contratados" in columnas


def test_pipeline_tiene_preprocesamiento_y_modelo() -> None:
    """El objeto entrenable debe encapsular ambas etapas."""
    pipeline = construir_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocesamiento", "modelo"]


def test_division_conserva_proporcion_de_clases(atributos_sinteticos: pd.DataFrame) -> None:
    """La particion estratificada debe conservar la tasa de positivos."""
    atributos, objetivo = separar_objetivo(atributos_sinteticos)
    entrenamiento_x, prueba_x, entrenamiento_y, prueba_y = dividir_datos(atributos, objetivo)

    assert len(prueba_x) + len(entrenamiento_x) == len(atributos)
    assert abs(entrenamiento_y.mean() - prueba_y.mean()) < TOLERANCIA_PROPORCION
    assert abs(len(prueba_x) / len(atributos) - PROPORCION_PRUEBA_ESPERADA) < TOLERANCIA_PROPORCION


def test_umbral_optimo_esta_en_rango_valido(atributos_sinteticos: pd.DataFrame) -> None:
    """El umbral debe ser una probabilidad y por tanto quedar entre cero y uno."""
    atributos, objetivo = separar_objetivo(atributos_sinteticos)
    umbral = optimizar_umbral(construir_pipeline(), atributos, objetivo)

    assert UMBRAL_MINIMO < umbral < UMBRAL_MAXIMO


def test_evaluar_devuelve_todas_las_metricas(atributos_sinteticos: pd.DataFrame) -> None:
    """La evaluacion debe reportar el conjunto completo de metricas."""
    atributos, objetivo = separar_objetivo(atributos_sinteticos)
    pipeline = construir_pipeline()
    pipeline.fit(atributos, objetivo)

    metricas = evaluar(pipeline, atributos, objetivo, umbral=0.5)

    assert set(metricas) == METRICAS_ESPERADAS
    assert all(UMBRAL_MINIMO <= valor <= UMBRAL_MAXIMO for valor in metricas.values())


def test_validacion_cruzada_reporta_dispersion(atributos_sinteticos: pd.DataFrame) -> None:
    """La validacion cruzada debe devolver media y desviacion entre pliegues."""
    atributos, objetivo = separar_objetivo(atributos_sinteticos)
    resultado = medir_validacion_cruzada(construir_pipeline(), atributos, objetivo)

    assert set(resultado) == {"f1_media", "f1_desviacion"}
    assert resultado["f1_desviacion"] >= 0


def test_cargar_atributos_falla_sin_archivo(tmp_path: Path) -> None:
    """Un archivo de atributos ausente debe producir un error explicito."""
    with pytest.raises(FileNotFoundError, match="pipeline de atributos"):
        cargar_atributos(tmp_path / "inexistente.parquet")


def test_entrenar_produce_modelo_y_metadatos(
    atributos_sinteticos: pd.DataFrame, tmp_path: Path
) -> None:
    """El entrenamiento completo debe dejar en disco el modelo y sus metadatos."""
    origen = tmp_path / "atributos.parquet"
    destino_modelo = tmp_path / "modelo.joblib"
    destino_metadatos = tmp_path / "metadatos.json"
    atributos_sinteticos.to_parquet(origen, index=False)

    metadatos = entrenar(origen, destino_modelo, destino_metadatos)

    assert destino_modelo.exists()
    assert destino_metadatos.exists()
    assert set(metadatos["metricas_prueba"]) == METRICAS_ESPERADAS
    assert UMBRAL_MINIMO < metadatos["umbral_decision"] < UMBRAL_MAXIMO


def test_metadatos_distinguen_columnas_de_entrada_y_de_modelo(
    atributos_sinteticos: pd.DataFrame, tmp_path: Path
) -> None:
    """Los metadatos deben separar lo que aporta el usuario de lo que se deriva."""
    origen = tmp_path / "atributos.parquet"
    atributos_sinteticos.to_parquet(origen, index=False)

    metadatos = entrenar(origen, tmp_path / "m.joblib", tmp_path / "m.json")

    assert "servicios_contratados" not in metadatos["columnas_entrada"]
    assert "servicios_contratados" in metadatos["columnas_modelo"]
    assert metadatos["atributos_derivados"] == ["servicios_contratados"]


def test_modelo_recuperado_predice_igual(
    atributos_sinteticos: pd.DataFrame, tmp_path: Path
) -> None:
    """El modelo serializado debe reproducir las mismas probabilidades al cargarse."""
    origen = tmp_path / "atributos.parquet"
    destino_modelo = tmp_path / "modelo.joblib"
    atributos_sinteticos.to_parquet(origen, index=False)

    entrenar(origen, destino_modelo, tmp_path / "metadatos.json")

    atributos, _ = separar_objetivo(atributos_sinteticos)
    recuperado = joblib.load(destino_modelo)
    probabilidades = recuperado.predict_proba(atributos)[:, 1]

    assert len(probabilidades) == len(atributos)
    assert np.all((probabilidades >= 0) & (probabilidades <= 1))


def test_metadatos_son_json_valido(atributos_sinteticos: pd.DataFrame, tmp_path: Path) -> None:
    """Los metadatos deben poder leerse desde otro proceso."""
    origen = tmp_path / "atributos.parquet"
    destino_metadatos = tmp_path / "metadatos.json"
    atributos_sinteticos.to_parquet(origen, index=False)

    entrenar(origen, tmp_path / "modelo.joblib", destino_metadatos)
    recuperados = json.loads(destino_metadatos.read_text(encoding="utf-8"))

    assert recuperados["familia"] == "logistica"
    assert "hiperparametros" in recuperados
