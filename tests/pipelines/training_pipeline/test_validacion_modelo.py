"""Pruebas de la validacion del modelo entrenado."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines.training_pipeline.train_pipeline import construir_pipeline
from pipelines.training_pipeline.validacion_modelo import (
    AJUSTE_ADECUADO,
    AJUSTE_SOBREAJUSTE,
    AJUSTE_SUBAJUSTE,
    clasificar_ajuste,
    comparar_con_referencia,
    evaluar_generalizacion,
    generar_curva_aprendizaje,
    guardar_evidencia,
    validar_modelo,
)

F1_ALTO = 0.95
F1_BUENO = 0.64
F1_MEDIO = 0.63
F1_BAJO = 0.30
DESVIACION = 0.02


def test_modelo_bien_ajustado_se_diagnostica_adecuado() -> None:
    """Brechas pequenas entre conjuntos indican generalizacion correcta."""
    assert clasificar_ajuste(0.6377, 0.6308, 0.6205) == AJUSTE_ADECUADO


def test_brecha_amplia_se_diagnostica_sobreajuste() -> None:
    """Un desempeno muy superior en entrenamiento delata memorizacion."""
    assert clasificar_ajuste(F1_ALTO, 0.60, 0.59) == AJUSTE_SOBREAJUSTE


def test_desempeno_bajo_generalizado_se_diagnostica_subajuste() -> None:
    """Un desempeno bajo en todos los conjuntos indica falta de capacidad."""
    assert clasificar_ajuste(F1_BAJO, 0.29, 0.28) == AJUSTE_SUBAJUSTE


def test_sobreajuste_produce_advertencia() -> None:
    """La brecha excesiva debe reportarse explicitamente."""
    errores, advertencias = evaluar_generalizacion(F1_ALTO, 0.60, 0.59)

    assert errores == []
    assert any("sobreajuste" in mensaje.lower() for mensaje in advertencias)


def test_deterioro_en_prueba_produce_advertencia() -> None:
    """Una caida fuerte de validacion a prueba advierte de estimacion optimista."""
    _, advertencias = evaluar_generalizacion(0.70, 0.69, 0.55)

    assert any("optimista" in mensaje for mensaje in advertencias)


def test_desempeno_insuficiente_es_error() -> None:
    """Un F1 por debajo del minimo aceptable impide el despliegue."""
    errores, _ = evaluar_generalizacion(0.55, 0.52, 0.45)

    assert len(errores) == 1
    assert "no es apto" in errores[0]


def test_modelo_sin_mejora_apreciable_advierte() -> None:
    """Igualar al modelo base es una conclusion que debe quedar registrada."""
    advertencias = comparar_con_referencia(0.6205)

    assert len(advertencias) == 1
    assert "no es apreciable" in advertencias[0]


def test_modelo_peor_que_la_referencia_advierte() -> None:
    """Quedar por debajo del modelo base significa no justificar la complejidad."""
    advertencias = comparar_con_referencia(0.55)

    assert len(advertencias) == 1
    assert "no justifica" in advertencias[0]


def test_mejora_clara_no_advierte() -> None:
    """Una mejora sustancial sobre la referencia no genera observaciones."""
    assert comparar_con_referencia(0.75) == []


def test_validacion_completa_produce_reporte() -> None:
    """El reporte debe contener las brechas y la comparacion con la referencia."""
    resultado = validar_modelo(0.6377, 0.6308, 0.6205, DESVIACION)

    assert resultado.valido
    assert resultado.diagnostico == AJUSTE_ADECUADO
    assert resultado.reporte["brecha_entrenamiento_validacion"] == pytest.approx(0.0069)
    assert resultado.reporte["mejora_sobre_base"] == pytest.approx(0.0005)


def test_guardar_evidencia_produce_json_legible(tmp_path: Path) -> None:
    """La evidencia debe quedar en disco y poder leerse desde otro proceso."""
    resultado = validar_modelo(F1_BUENO, F1_MEDIO, F1_MEDIO, DESVIACION)
    destino = guardar_evidencia(resultado.reporte, tmp_path / "reporte.json")

    recuperado = json.loads(destino.read_text(encoding="utf-8"))

    assert destino.exists()
    assert recuperado["diagnostico"] == AJUSTE_ADECUADO


def test_curva_de_aprendizaje_se_genera(tmp_path: Path) -> None:
    """La curva de aprendizaje debe producirse como evidencia grafica."""
    generador = np.random.default_rng(3)
    n = 200
    contrato = generador.choice(["Month-to-month", "One year", "Two year"], size=n)
    antiguedad = generador.integers(0, 73, size=n)

    atributos = pd.DataFrame(
        {
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
            "PaymentMethod": generador.choice(["Electronic check", "Mailed check"], size=n),
            "MonthlyCharges": generador.uniform(20, 118, size=n),
            "TotalCharges": generador.uniform(0, 8000, size=n),
            "servicios_contratados": generador.integers(1, 10, size=n),
        }
    )
    objetivo = pd.Series((contrato == "Month-to-month").astype(int))

    destino = generar_curva_aprendizaje(
        construir_pipeline(), atributos, objetivo, tmp_path / "curva.png"
    )

    assert destino.exists()
    assert destino.stat().st_size > 0
