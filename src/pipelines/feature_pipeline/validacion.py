"""Validacion de datos e integridad para el pipeline de atributos.

Las reglas provienen del analisis exploratorio (notebook 3, seccion 3.6) y se agrupan
en tres niveles:

- **Esquema**: tipos, rangos, categorias validas y unicidad. Declarado con Pandera.
- **Integridad entre campos**: relaciones que deben cumplirse entre columnas de un
  mismo registro. No expresables como restricciones de columna aislada.
- **Distribucion**: verificaciones sobre el lote completo. Producen advertencias, no
  errores, porque una desviacion puede reflejar un cambio real en la poblacion.

Un fallo de esquema o de integridad detiene el pipeline y **los atributos no se
persisten**. Un fallo de distribucion se registra pero permite continuar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa

SI = "Yes"
NO = "No"
SIN_INTERNET = "No internet service"
SIN_TELEFONO = "No phone service"

ANTIGUEDAD_MAXIMA = 72
CARGO_MENSUAL_MINIMO = 10.0
CARGO_MENSUAL_MAXIMO = 200.0
SERVICIOS_MAXIMOS = 9

PCT_NULOS_MAXIMO = 1.0
TASA_CANCELACION_MINIMA = 0.15
TASA_CANCELACION_MAXIMA = 0.40
TOLERANCIA_ACUMULADO = 0.35
PCT_INCOHERENCIA_MAXIMA = 5.0

SERVICIOS_DE_INTERNET = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORIAS_VALIDAS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [NO, SI],
    "Partner": [NO, SI],
    "Dependents": [NO, SI],
    "PhoneService": [NO, SI],
    "MultipleLines": [NO, SI, SIN_TELEFONO],
    "InternetService": ["DSL", "Fiber optic", NO],
    "OnlineSecurity": [NO, SI, SIN_INTERNET],
    "OnlineBackup": [NO, SI, SIN_INTERNET],
    "DeviceProtection": [NO, SI, SIN_INTERNET],
    "TechSupport": [NO, SI, SIN_INTERNET],
    "StreamingTV": [NO, SI, SIN_INTERNET],
    "StreamingMovies": [NO, SI, SIN_INTERNET],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": [NO, SI],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


class ErrorDeValidacion(Exception):
    """Se levanta cuando los datos no superan las validaciones obligatorias."""


@dataclass
class ResultadoValidacion:
    """Resultado de aplicar el conjunto completo de validaciones."""

    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)

    @property
    def valido(self) -> bool:
        """Indica si los datos pueden continuar por el pipeline."""
        return not self.errores


def _columna_categorica(nombre: str) -> pa.Column:
    """Construye la definicion de una columna categorica con su conjunto cerrado."""
    return pa.Column(
        nullable=False,
        checks=pa.Check.isin(CATEGORIAS_VALIDAS[nombre]),
        coerce=False,
    )


def construir_esquema() -> pa.DataFrameSchema:
    """Declara el esquema esperado del conjunto de atributos.

    `TotalCharges` es la unica columna que admite nulos, y su ausencia esta acotada
    por una regla de integridad: solo puede faltar cuando la antiguedad es cero.
    """
    columnas: dict[str, pa.Column] = {
        nombre: _columna_categorica(nombre) for nombre in CATEGORIAS_VALIDAS
    }

    columnas["tenure"] = pa.Column(
        nullable=False,
        checks=pa.Check.in_range(0, ANTIGUEDAD_MAXIMA),
    )
    columnas["MonthlyCharges"] = pa.Column(
        nullable=False,
        checks=pa.Check.in_range(CARGO_MENSUAL_MINIMO, CARGO_MENSUAL_MAXIMO),
    )
    columnas["TotalCharges"] = pa.Column(
        nullable=True,
        checks=pa.Check.ge(0),
    )
    columnas["servicios_contratados"] = pa.Column(
        nullable=False,
        checks=pa.Check.in_range(0, SERVICIOS_MAXIMOS),
    )
    columnas["customerID"] = pa.Column(nullable=False, unique=True, required=False)
    columnas["Churn"] = pa.Column(
        nullable=False,
        checks=pa.Check.isin([NO, SI]),
        required=False,
    )

    return pa.DataFrameSchema(columnas, strict=False, coerce=False)


ESQUEMA_ATRIBUTOS = construir_esquema()


def _resumir_fallos(fallos: pd.DataFrame) -> list[str]:
    """Agrupa los fallos de esquema en mensajes legibles, uno por regla incumplida."""
    mensajes: list[str] = []
    agrupados = fallos.groupby(["column", "check"], dropna=False).size()

    for (columna, regla), cantidad in agrupados.items():
        ejemplo = fallos.loc[
            (fallos["column"] == columna) & (fallos["check"] == regla), "failure_case"
        ].iloc[0]
        mensajes.append(
            f"Esquema: la columna '{columna}' incumple '{regla}' "
            f"en {int(cantidad)} registros (ejemplo: {ejemplo!r})"
        )

    return mensajes


def validar_esquema(datos: pd.DataFrame) -> list[str]:
    """Verifica tipos, rangos, categorias validas y unicidad del identificador."""
    try:
        ESQUEMA_ATRIBUTOS.validate(datos, lazy=True)
    except pa.errors.SchemaErrors as error:
        return _resumir_fallos(error.failure_cases)
    return []


def validar_acumulado_solo_nulo_sin_antiguedad(datos: pd.DataFrame) -> list[str]:
    """`TotalCharges` solo puede ser nulo cuando la antiguedad es cero.

    El analisis exploratorio demostro que los once ausentes del conjunto original
    corresponden exactamente a los once clientes sin primera factura emitida. Un
    nulo con antiguedad positiva indicaria un fallo del proceso de facturacion.
    """
    if "TotalCharges" not in datos.columns or "tenure" not in datos.columns:
        return []

    incoherentes = datos["TotalCharges"].isna() & (datos["tenure"] > 0)
    if not incoherentes.any():
        return []

    return [
        f"Integridad: {int(incoherentes.sum())} registros tienen 'TotalCharges' nulo "
        "con antiguedad mayor que cero"
    ]


def validar_servicios_de_telefonia(datos: pd.DataFrame) -> list[str]:
    """Sin servicio telefonico, las lineas multiples deben indicar esa ausencia."""
    if not {"PhoneService", "MultipleLines"} <= set(datos.columns):
        return []

    sin_telefono = datos["PhoneService"].astype("string") == NO
    esperado = datos["MultipleLines"].astype("string") == SIN_TELEFONO
    incoherentes = sin_telefono & ~esperado

    if not incoherentes.any():
        return []

    return [
        f"Integridad: {int(incoherentes.sum())} registros sin servicio telefonico "
        f"no declaran '{SIN_TELEFONO}' en 'MultipleLines'"
    ]


def validar_servicios_de_internet(datos: pd.DataFrame) -> list[str]:
    """Sin servicio de internet, los seis complementos deben indicar esa ausencia."""
    if "InternetService" not in datos.columns:
        return []

    sin_internet = datos["InternetService"].astype("string") == NO
    mensajes: list[str] = []

    for columna in SERVICIOS_DE_INTERNET:
        if columna not in datos.columns:
            continue
        esperado = datos[columna].astype("string") == SIN_INTERNET
        incoherentes = sin_internet & ~esperado
        if incoherentes.any():
            mensajes.append(
                f"Integridad: {int(incoherentes.sum())} registros sin internet no "
                f"declaran '{SIN_INTERNET}' en '{columna}'"
            )

    return mensajes


def validar_coherencia_del_acumulado(datos: pd.DataFrame) -> list[str]:
    """El cargo acumulado debe guardar relacion con antiguedad por cargo mensual.

    No se exige igualdad: las tarifas cambian y hay servicios puntuales. Se verifica
    que la desviacion respecto al producto esperado se mantenga dentro de una banda
    amplia para la mayoria de los registros.
    """
    requeridas = {"TotalCharges", "tenure", "MonthlyCharges"}
    if not requeridas <= set(datos.columns):
        return []

    evaluables = datos[datos["tenure"] > 0].dropna(subset=["TotalCharges"])
    if evaluables.empty:
        return []

    esperado = evaluables["tenure"] * evaluables["MonthlyCharges"]
    desviacion = (evaluables["TotalCharges"] - esperado).abs() / esperado
    fuera_de_banda = (desviacion > TOLERANCIA_ACUMULADO).mean() * 100

    if fuera_de_banda <= PCT_INCOHERENCIA_MAXIMA:
        return []

    return [
        f"Integridad: {fuera_de_banda:.1f}% de los registros tienen 'TotalCharges' "
        f"fuera de la banda de tolerancia respecto al producto de antiguedad por "
        f"cargo mensual (maximo admitido {PCT_INCOHERENCIA_MAXIMA}%)"
    ]


def validar_integridad(datos: pd.DataFrame) -> list[str]:
    """Aplica todas las reglas de relacion entre campos."""
    return [
        *validar_acumulado_solo_nulo_sin_antiguedad(datos),
        *validar_servicios_de_telefonia(datos),
        *validar_servicios_de_internet(datos),
        *validar_coherencia_del_acumulado(datos),
    ]


def validar_distribucion(datos: pd.DataFrame) -> list[str]:
    """Verifica que el lote se parezca a la poblacion conocida.

    Produce advertencias en lugar de errores: una desviacion puede deberse a un
    cambio real de la poblacion, y detener el pipeline por ello seria excesivo. Pero
    conviene dejar constancia, porque tambien puede indicar un error de extraccion.
    """
    advertencias: list[str] = []

    if "TotalCharges" in datos.columns:
        pct_nulos = datos["TotalCharges"].isna().mean() * 100
        if pct_nulos > PCT_NULOS_MAXIMO:
            advertencias.append(
                f"Distribucion: 'TotalCharges' tiene {pct_nulos:.2f}% de nulos, "
                f"por encima del {PCT_NULOS_MAXIMO}% habitual"
            )

    if "Churn" in datos.columns:
        tasa = (datos["Churn"].astype("string") == SI).mean()
        if not TASA_CANCELACION_MINIMA <= tasa <= TASA_CANCELACION_MAXIMA:
            advertencias.append(
                f"Distribucion: la tasa de cancelacion del lote es {tasa:.1%}, fuera "
                f"del rango esperado "
                f"[{TASA_CANCELACION_MINIMA:.0%}, {TASA_CANCELACION_MAXIMA:.0%}]"
            )

    return advertencias


def validar(datos: pd.DataFrame) -> ResultadoValidacion:
    """Aplica esquema, integridad y distribucion sobre el conjunto de atributos."""
    return ResultadoValidacion(
        errores=[*validar_esquema(datos), *validar_integridad(datos)],
        advertencias=validar_distribucion(datos),
    )
