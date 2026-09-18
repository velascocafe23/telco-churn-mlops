"""Verificacion de la particion entre entrenamiento y prueba.

Comprueba dos cosas distintas que suelen confundirse:

- **Fuga de informacion**: que ningun registro este simultaneamente en ambos
  conjuntos. Un solo caso invalida la evaluacion, porque el modelo estaria siendo
  juzgado sobre datos que ya vio.
- **Representatividad**: que ambos conjuntos procedan de la misma distribucion. Si
  difieren, las metricas de prueba no estiman el desempeno sobre la poblacion.

La distincion entre error y advertencia es deliberada. El solapamiento de indices
indica un fallo del mecanismo de particion y detiene el pipeline. Una diferencia de
distribucion, en cambio, puede deberse al azar del muestreo y solo se advierte.

Caso particular de este proyecto: el analisis exploratorio identifico 42 registros
con atributos identicos entre si y distinto identificador. Al partir, algunos de esos
pares quedan repartidos entre ambos conjuntos. **No es fuga**: son clientes distintos
cuyo perfil coincide, consecuencia de la baja cardinalidad de los atributos. Se
reporta como advertencia, no como error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

ALFA = 0.01
CATEGORIAS_MINIMAS = 2
TOLERANCIA_TASA_POSITIVA = 0.02


@dataclass
class ResultadoParticion:
    """Resultado de verificar una particion entre entrenamiento y prueba."""

    errores: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    reporte: dict[str, float] = field(default_factory=dict)

    @property
    def valida(self) -> bool:
        """Indica si la particion puede usarse para entrenar y evaluar."""
        return not self.errores


def verificar_indices_disjuntos(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> list[str]:
    """Ningun registro puede aparecer en ambos conjuntos.

    Es la comprobacion mas directa de fuga: si un mismo registro esta en los dos
    lados, la evaluacion mide memorizacion y no capacidad de generalizar.
    """
    compartidos = entrenamiento.index.intersection(prueba.index)
    if compartidos.empty:
        return []

    return [
        f"Fuga: {len(compartidos)} registros aparecen en entrenamiento y prueba "
        f"simultaneamente (indices {list(compartidos[:5])}...)"
    ]


def verificar_tamanos(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> list[str]:
    """Ambos conjuntos deben contener registros."""
    errores: list[str] = []
    if entrenamiento.empty:
        errores.append("Particion: el conjunto de entrenamiento esta vacio")
    if prueba.empty:
        errores.append("Particion: el conjunto de prueba esta vacio")
    return errores


def verificar_tasa_positiva(
    objetivo_entrenamiento: pd.Series, objetivo_prueba: pd.Series
) -> list[str]:
    """La estratificacion debe conservar la proporcion de la clase positiva."""
    diferencia = abs(float(objetivo_entrenamiento.mean()) - float(objetivo_prueba.mean()))
    if diferencia <= TOLERANCIA_TASA_POSITIVA:
        return []

    return [
        f"Particion: la tasa de positivos difiere en {diferencia:.4f} entre conjuntos, "
        f"por encima de la tolerancia de {TOLERANCIA_TASA_POSITIVA}. "
        "La particion podria no ser estratificada."
    ]


def detectar_perfiles_compartidos(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> list[str]:
    """Registros de prueba cuyo perfil completo aparece tambien en entrenamiento.

    No implica fuga por si mismo. Con atributos de baja cardinalidad es esperable
    que clientes distintos compartan perfil. Se reporta porque un porcentaje alto si
    seria sospechoso, y porque acota el desempeno alcanzable cuando ademas presentan
    desenlaces distintos.
    """
    comunes = [columna for columna in prueba.columns if columna in entrenamiento.columns]
    if not comunes:
        return []

    claves_entrenamiento = set(map(tuple, entrenamiento[comunes].astype("string").to_numpy()))
    claves_prueba = list(map(tuple, prueba[comunes].astype("string").to_numpy()))
    coincidentes = sum(clave in claves_entrenamiento for clave in claves_prueba)

    if coincidentes == 0:
        return []

    porcentaje = coincidentes / len(claves_prueba) * 100
    return [
        f"Particion: {coincidentes} registros de prueba ({porcentaje:.1f}%) tienen un "
        "perfil identico a alguno de entrenamiento. Son clientes distintos con "
        "atributos coincidentes, no fuga."
    ]


def comparar_distribuciones_numericas(
    entrenamiento: pd.DataFrame, prueba: pd.DataFrame
) -> tuple[list[str], dict[str, float]]:
    """Compara cada atributo numerico con la prueba de Kolmogorov y Smirnov."""
    advertencias: list[str] = []
    valores_p: dict[str, float] = {}

    for columna in entrenamiento.select_dtypes(include="number").columns:
        if columna not in prueba.columns:
            continue
        resultado = ks_2samp(entrenamiento[columna].dropna(), prueba[columna].dropna())
        valor_p = float(resultado.pvalue)
        valores_p[columna] = round(valor_p, 4)
        if valor_p < ALFA:
            advertencias.append(
                f"Distribucion: '{columna}' difiere entre conjuntos "
                f"(Kolmogorov-Smirnov, valor p {valor_p:.4f})"
            )

    return advertencias, valores_p


def comparar_distribuciones_categoricas(
    entrenamiento: pd.DataFrame, prueba: pd.DataFrame
) -> tuple[list[str], dict[str, float]]:
    """Compara cada atributo categorico con la prueba de chi cuadrado."""
    advertencias: list[str] = []
    valores_p: dict[str, float] = {}

    categoricas = entrenamiento.select_dtypes(include=["category", "object", "string"])

    for columna in categoricas.columns:
        if columna not in prueba.columns:
            continue

        tabla = pd.crosstab(
            pd.concat(
                [
                    pd.Series(["entrenamiento"] * len(entrenamiento)),
                    pd.Series(["prueba"] * len(prueba)),
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    entrenamiento[columna].astype("string").reset_index(drop=True),
                    prueba[columna].astype("string").reset_index(drop=True),
                ],
                ignore_index=True,
            ),
        )

        if tabla.shape[1] < CATEGORIAS_MINIMAS:
            continue

        valor_p = float(chi2_contingency(tabla).pvalue)
        valores_p[columna] = round(valor_p, 4)
        if valor_p < ALFA:
            advertencias.append(
                f"Distribucion: '{columna}' difiere entre conjuntos "
                f"(chi cuadrado, valor p {valor_p:.4f})"
            )

    return advertencias, valores_p


def detectar_categorias_no_vistas(entrenamiento: pd.DataFrame, prueba: pd.DataFrame) -> list[str]:
    """Categorias presentes en prueba que el modelo no vera al entrenar.

    El codificador esta configurado para ignorarlas, de modo que no provocan un
    fallo, pero el modelo carece de informacion sobre ellas y sus predicciones para
    esos registros son menos fiables.
    """
    advertencias: list[str] = []
    categoricas = entrenamiento.select_dtypes(include=["category", "object", "string"])

    for columna in categoricas.columns:
        if columna not in prueba.columns:
            continue
        vistas = set(entrenamiento[columna].astype("string").dropna().unique())
        nuevas = set(prueba[columna].astype("string").dropna().unique()) - vistas
        if nuevas:
            advertencias.append(
                f"Particion: '{columna}' tiene en prueba categorias ausentes en "
                f"entrenamiento: {sorted(nuevas)}"
            )

    return advertencias


def verificar_particion(
    entrenamiento: pd.DataFrame,
    prueba: pd.DataFrame,
    objetivo_entrenamiento: pd.Series,
    objetivo_prueba: pd.Series,
) -> ResultadoParticion:
    """Aplica todas las comprobaciones sobre la particion."""
    advertencias_numericas, p_numericos = comparar_distribuciones_numericas(entrenamiento, prueba)
    advertencias_categoricas, p_categoricos = comparar_distribuciones_categoricas(
        entrenamiento, prueba
    )

    reporte: dict[str, float] = {
        "registros_entrenamiento": float(len(entrenamiento)),
        "registros_prueba": float(len(prueba)),
        "tasa_positiva_entrenamiento": round(float(objetivo_entrenamiento.mean()), 4),
        "tasa_positiva_prueba": round(float(objetivo_prueba.mean()), 4),
    }
    reporte.update({f"valor_p_{nombre}": valor for nombre, valor in p_numericos.items()})
    reporte.update({f"valor_p_{nombre}": valor for nombre, valor in p_categoricos.items()})

    return ResultadoParticion(
        errores=[
            *verificar_tamanos(entrenamiento, prueba),
            *verificar_indices_disjuntos(entrenamiento, prueba),
            *verificar_tasa_positiva(objetivo_entrenamiento, objetivo_prueba),
        ],
        advertencias=[
            *advertencias_numericas,
            *advertencias_categoricas,
            *detectar_categorias_no_vistas(entrenamiento, prueba),
            *detectar_perfiles_compartidos(entrenamiento, prueba),
        ],
        reporte=reporte,
    )
