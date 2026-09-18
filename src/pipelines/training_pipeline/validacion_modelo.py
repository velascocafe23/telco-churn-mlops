"""Validacion del rendimiento y la generalizacion del modelo.

Convierte tres numeros que el pipeline ya calcula (desempeno en entrenamiento, en
validacion cruzada y en prueba) en un diagnostico explicito, con umbrales declarados
en lugar de apreciaciones.

El criterio central es que un modelo puede fallar de dos formas opuestas:

- **Sobreajuste**: aprende particularidades del conjunto de entrenamiento que no se
  repiten fuera de el. Se manifiesta como una brecha amplia entre el desempeno en
  entrenamiento y el de validacion.
- **Subajuste**: carece de capacidad para representar la relacion existente. Se
  manifiesta como un desempeno bajo en todos los conjuntos por igual.

Un tercer caso no es un fallo del modelo pero si una conclusion relevante: que el
modelo generalice bien y aun asi no supere al modelo base significa que el limite
esta en la informacion disponible, no en el algoritmo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    learning_curve,
)
from sklearn.pipeline import Pipeline

BRECHA_SOBREAJUSTE = 0.05
BRECHA_DETERIORO = 0.05
DESEMPENO_MINIMO_ACEPTABLE = 0.50
MEJORA_APRECIABLE = 0.01
F1_MODELO_BASE = 0.620
N_TAMANOS_CURVA = 8
N_PLIEGUES_CURVA = 5
SEMILLA = 42
PROPORCION_MINIMA_CURVA = 0.2

AJUSTE_ADECUADO = "adecuado"
AJUSTE_SOBREAJUSTE = "sobreajuste"
AJUSTE_SUBAJUSTE = "subajuste"


@dataclass
class ResultadoValidacionModelo:
    """Diagnostico del modelo entrenado."""

    diagnostico: str = AJUSTE_ADECUADO
    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    reporte: dict[str, float | str] = field(default_factory=dict)

    @property
    def valido(self) -> bool:
        """Indica si el modelo cumple los criterios minimos para desplegarse."""
        return not self.errores


def clasificar_ajuste(f1_entrenamiento: float, f1_validacion: float, f1_prueba: float) -> str:
    """Determina si el modelo sobreajusta, subajusta o generaliza correctamente."""
    if f1_prueba < DESEMPENO_MINIMO_ACEPTABLE and f1_entrenamiento < DESEMPENO_MINIMO_ACEPTABLE:
        return AJUSTE_SUBAJUSTE

    if f1_entrenamiento - f1_validacion > BRECHA_SOBREAJUSTE:
        return AJUSTE_SOBREAJUSTE

    return AJUSTE_ADECUADO


def evaluar_generalizacion(
    f1_entrenamiento: float, f1_validacion: float, f1_prueba: float
) -> tuple[list[str], list[str]]:
    """Comprueba que el desempeno se mantenga entre conjuntos."""
    errores: list[str] = []
    advertencias: list[str] = []

    brecha_ajuste = f1_entrenamiento - f1_validacion
    brecha_generalizacion = f1_validacion - f1_prueba

    if brecha_ajuste > BRECHA_SOBREAJUSTE:
        advertencias.append(
            f"Generalizacion: la brecha entre entrenamiento y validacion es "
            f"{brecha_ajuste:.4f}, por encima de {BRECHA_SOBREAJUSTE}. Indica sobreajuste."
        )

    if brecha_generalizacion > BRECHA_DETERIORO:
        advertencias.append(
            f"Generalizacion: el desempeno cae {brecha_generalizacion:.4f} de validacion "
            f"a prueba, por encima de {BRECHA_DETERIORO}. La estimacion de validacion "
            "resulta optimista."
        )

    if f1_prueba < DESEMPENO_MINIMO_ACEPTABLE:
        errores.append(
            f"Desempeno: el F1 en prueba es {f1_prueba:.4f}, por debajo del minimo "
            f"aceptable de {DESEMPENO_MINIMO_ACEPTABLE}. El modelo no es apto para uso."
        )

    return errores, advertencias


def comparar_con_referencia(f1_prueba: float, referencia: float = F1_MODELO_BASE) -> list[str]:
    """Contrasta el modelo contra la referencia establecida en la prueba de concepto.

    No superar la referencia no invalida el modelo, pero es una conclusion que debe
    quedar registrada: indica que la complejidad adicional no se esta traduciendo en
    mejor desempeno.
    """
    mejora = f1_prueba - referencia

    if mejora < 0:
        return [
            f"Referencia: el F1 en prueba ({f1_prueba:.4f}) es inferior al del modelo "
            f"base ({referencia:.4f}). El modelo no justifica su complejidad."
        ]

    if mejora < MEJORA_APRECIABLE:
        return [
            f"Referencia: el F1 en prueba ({f1_prueba:.4f}) supera al modelo base "
            f"({referencia:.4f}) en solo {mejora:.4f}. La mejora no es apreciable: el "
            "limite esta en la informacion disponible, no en el algoritmo."
        ]

    return []


def generar_curva_aprendizaje(
    pipeline: Pipeline,
    atributos: pd.DataFrame,
    objetivo: pd.Series,
    destino: Path,
) -> Path:
    """Produce la curva de aprendizaje como evidencia grafica de la validacion."""
    # Los pliegues se estratifican y la fraccion minima no baja del 20%. Con
    # fracciones menores, un subconjunto puede quedar con una sola clase y el ajuste
    # falla, lo que introduciria valores nulos en la curva sin interrumpir la
    # ejecucion: evidencia corrupta en silencio.
    tamanos, puntaje_entrenamiento, puntaje_validacion = learning_curve(
        pipeline,
        atributos,
        objetivo,
        train_sizes=np.linspace(PROPORCION_MINIMA_CURVA, 1.0, N_TAMANOS_CURVA),
        cv=StratifiedKFold(n_splits=N_PLIEGUES_CURVA, shuffle=True, random_state=SEMILLA),
        scoring="f1",
        n_jobs=-1,
    )

    figura, eje = plt.subplots(figsize=(7, 4.5))
    eje.plot(tamanos, puntaje_entrenamiento.mean(axis=1), marker="o", label="entrenamiento")
    eje.plot(tamanos, puntaje_validacion.mean(axis=1), marker="s", label="validacion")
    eje.axhline(F1_MODELO_BASE, color="darkgreen", linestyle=":", label="modelo base")
    eje.set_title("Curva de aprendizaje del modelo desplegado")
    eje.set_xlabel("registros de entrenamiento")
    eje.set_ylabel("F1")
    eje.legend()
    figura.tight_layout()

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino)
    plt.close(figura)

    return destino


def guardar_evidencia(reporte: dict[str, float | str], destino: Path) -> Path:
    """Persiste el reporte de validacion para dejar constancia reproducible."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def validar_modelo(
    f1_entrenamiento: float,
    f1_validacion: float,
    f1_prueba: float,
    desviacion_validacion: float,
) -> ResultadoValidacionModelo:
    """Aplica el conjunto completo de comprobaciones sobre el modelo entrenado."""
    errores, advertencias = evaluar_generalizacion(f1_entrenamiento, f1_validacion, f1_prueba)
    advertencias.extend(comparar_con_referencia(f1_prueba))

    diagnostico = clasificar_ajuste(f1_entrenamiento, f1_validacion, f1_prueba)

    reporte: dict[str, float | str] = {
        "diagnostico": diagnostico,
        "f1_entrenamiento": round(f1_entrenamiento, 4),
        "f1_validacion_cruzada": round(f1_validacion, 4),
        "f1_prueba": round(f1_prueba, 4),
        "desviacion_validacion": round(desviacion_validacion, 4),
        "brecha_entrenamiento_validacion": round(f1_entrenamiento - f1_validacion, 4),
        "brecha_validacion_prueba": round(f1_validacion - f1_prueba, 4),
        "f1_modelo_base": F1_MODELO_BASE,
        "mejora_sobre_base": round(f1_prueba - F1_MODELO_BASE, 4),
    }

    return ResultadoValidacionModelo(
        diagnostico=diagnostico,
        errores=errores,
        advertencias=advertencias,
        reporte=reporte,
    )
