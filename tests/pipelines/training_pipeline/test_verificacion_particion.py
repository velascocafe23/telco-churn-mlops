"""Pruebas de la verificacion de la particion entre entrenamiento y prueba."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from pipelines.training_pipeline.verificacion_particion import (
    comparar_distribuciones_numericas,
    detectar_categorias_no_vistas,
    detectar_perfiles_compartidos,
    verificar_indices_disjuntos,
    verificar_particion,
    verificar_tamanos,
    verificar_tasa_positiva,
)

REGISTROS = 300
SEMILLA = 11
PROPORCION_POSITIVA = 0.3
ALFA_PRUEBAS = 0.01
REGISTROS_ENTRENAMIENTO = 240


@pytest.fixture
def conjunto() -> pd.DataFrame:
    """Conjunto sintetico con atributos numericos y categoricos."""
    generador = np.random.default_rng(SEMILLA)
    return pd.DataFrame(
        {
            "tenure": generador.integers(0, 72, size=REGISTROS),
            "MonthlyCharges": generador.uniform(20, 120, size=REGISTROS),
            "Contract": generador.choice(
                ["Month-to-month", "One year", "Two year"], size=REGISTROS
            ),
        }
    )


@pytest.fixture
def objetivo() -> pd.Series:
    """Vector objetivo con proporcion conocida de positivos."""
    generador = np.random.default_rng(SEMILLA)
    return pd.Series((generador.random(REGISTROS) < PROPORCION_POSITIVA).astype(int))


def test_particion_correcta_no_produce_errores(conjunto: pd.DataFrame, objetivo: pd.Series) -> None:
    """Una particion estratificada y disjunta debe considerarse valida."""
    partes = train_test_split(
        conjunto, objetivo, test_size=0.2, random_state=SEMILLA, stratify=objetivo
    )
    resultado = verificar_particion(partes[0], partes[1], partes[2], partes[3])

    assert resultado.valida
    assert resultado.errores == []


def test_indices_compartidos_son_error(conjunto: pd.DataFrame) -> None:
    """Un registro presente en ambos conjuntos indica fuga de informacion."""
    entrenamiento = conjunto.iloc[:200]
    prueba = conjunto.iloc[150:]

    errores = verificar_indices_disjuntos(entrenamiento, prueba)

    assert len(errores) == 1
    assert "Fuga" in errores[0]


def test_indices_disjuntos_no_producen_error(conjunto: pd.DataFrame) -> None:
    """Sin solapamiento de indices no debe reportarse fuga."""
    assert verificar_indices_disjuntos(conjunto.iloc[:200], conjunto.iloc[200:]) == []


def test_conjunto_vacio_es_error(conjunto: pd.DataFrame) -> None:
    """Una particion que deja un lado vacio es inutilizable."""
    errores = verificar_tamanos(conjunto, conjunto.iloc[:0])

    assert len(errores) == 1
    assert "prueba" in errores[0]


def test_tasa_positiva_desbalanceada_es_error() -> None:
    """Si la estratificacion falla, las tasas difieren mas de lo tolerado."""
    entrenamiento = pd.Series([1] * 80 + [0] * 20)
    prueba = pd.Series([1] * 20 + [0] * 80)

    errores = verificar_tasa_positiva(entrenamiento, prueba)

    assert len(errores) == 1
    assert "estratificada" in errores[0]


def test_tasa_positiva_similar_no_es_error() -> None:
    """Diferencias pequenas de tasa son propias del muestreo y no alertan."""
    entrenamiento = pd.Series([1] * 30 + [0] * 70)
    prueba = pd.Series([1] * 31 + [0] * 69)

    assert verificar_tasa_positiva(entrenamiento, prueba) == []


def test_distribuciones_distintas_producen_advertencia(conjunto: pd.DataFrame) -> None:
    """Un desplazamiento fuerte de un atributo numerico debe detectarse."""
    entrenamiento = conjunto.copy()
    prueba = conjunto.copy()
    prueba["tenure"] = prueba["tenure"] + 200

    advertencias, valores_p = comparar_distribuciones_numericas(entrenamiento, prueba)

    assert any("tenure" in mensaje for mensaje in advertencias)
    assert valores_p["tenure"] < ALFA_PRUEBAS


def test_distribuciones_iguales_no_advierten(conjunto: pd.DataFrame) -> None:
    """Dos mitades de un mismo conjunto no deberian diferir significativamente."""
    advertencias, _ = comparar_distribuciones_numericas(conjunto.iloc[:150], conjunto.iloc[150:])

    assert advertencias == []


def test_categoria_nueva_en_prueba_advierte(conjunto: pd.DataFrame) -> None:
    """Una categoria que el modelo no vera al entrenar debe reportarse."""
    entrenamiento = conjunto[conjunto["Contract"] != "Two year"]
    prueba = conjunto[conjunto["Contract"] == "Two year"]

    advertencias = detectar_categorias_no_vistas(entrenamiento, prueba)

    assert any("Two year" in mensaje for mensaje in advertencias)


def test_perfiles_compartidos_son_advertencia_no_error() -> None:
    """Clientes distintos con perfil identico no constituyen fuga."""
    perfil = {"tenure": 1, "MonthlyCharges": 70.0, "Contract": "Month-to-month"}
    entrenamiento = pd.DataFrame([perfil, perfil], index=[0, 1])
    prueba = pd.DataFrame([perfil], index=[2])

    advertencias = detectar_perfiles_compartidos(entrenamiento, prueba)

    assert len(advertencias) == 1
    assert "no fuga" in advertencias[0]
    assert verificar_indices_disjuntos(entrenamiento, prueba) == []


def test_reporte_incluye_metricas_de_la_particion(
    conjunto: pd.DataFrame, objetivo: pd.Series
) -> None:
    """El reporte debe permitir dejar constancia de lo verificado."""
    resultado = verificar_particion(
        conjunto.iloc[:240], conjunto.iloc[240:], objetivo.iloc[:240], objetivo.iloc[240:]
    )

    assert resultado.reporte["registros_entrenamiento"] == REGISTROS_ENTRENAMIENTO
    assert "tasa_positiva_prueba" in resultado.reporte
    assert "valor_p_tenure" in resultado.reporte
