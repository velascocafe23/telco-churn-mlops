"""Demostracion del modelo de prediccion de cancelacion de clientes.

Aplicacion Streamlit con dos modos de uso:

- Prediccion individual: formulario para evaluar un cliente.
- Procesamiento por lote: carga de un archivo con multiples clientes.

Ejecucion local:

    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).parent

# La logica de construccion de atributos vive en el pipeline, no aqui. Importarla
# garantiza que la aplicacion aplique exactamente las mismas transformaciones que
# se usaron durante el entrenamiento.
DIRECTORIO_FUENTES = RAIZ / "src"
if str(DIRECTORIO_FUENTES) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_FUENTES))

from pipelines.feature_pipeline.feature_pipeline import (  # noqa: E402
    construir_atributos,
)

MODELO_FILE = RAIZ / "models" / "modelo_churn.joblib"
METADATOS_FILE = RAIZ / "models" / "modelo_churn_metadatos.json"
EJEMPLO_FILE = RAIZ / "app" / "ejemplos" / "clientes_ejemplo.csv"

SIN_INTERNET = "No internet service"
SIN_TELEFONO = "No phone service"
SI = "Yes"
NO = "No"

OPCIONES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [NO, SI],
    "Partner": [NO, SI],
    "Dependents": [NO, SI],
    "PhoneService": [SI, NO],
    "MultipleLines": [NO, SI, SIN_TELEFONO],
    "InternetService": ["DSL", "Fiber optic", NO],
    "OnlineSecurity": [NO, SI, SIN_INTERNET],
    "OnlineBackup": [NO, SI, SIN_INTERNET],
    "DeviceProtection": [NO, SI, SIN_INTERNET],
    "TechSupport": [NO, SI, SIN_INTERNET],
    "StreamingTV": [NO, SI, SIN_INTERNET],
    "StreamingMovies": [NO, SI, SIN_INTERNET],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": [SI, NO],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

RANGOS_NUMERICOS: dict[str, tuple[float, float, float]] = {
    "tenure": (0.0, 72.0, 12.0),
    "MonthlyCharges": (18.0, 120.0, 70.0),
    "TotalCharges": (0.0, 9000.0, 840.0),
}

ETIQUETAS: dict[str, str] = {
    "gender": "Género",
    "SeniorCitizen": "Adulto mayor",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes",
    "tenure": "Antigüedad (meses)",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Líneas múltiples",
    "InternetService": "Servicio de internet",
    "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección de dispositivo",
    "TechSupport": "Soporte técnico",
    "StreamingTV": "Televisión por streaming",
    "StreamingMovies": "Películas por streaming",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Factura electrónica",
    "PaymentMethod": "Método de pago",
    "MonthlyCharges": "Cargo mensual",
    "TotalCharges": "Cargo acumulado",
}

COLUMNAS_NUMERICAS = list(RANGOS_NUMERICOS)


@st.cache_resource
def cargar_modelo() -> tuple[Any, dict[str, Any]]:
    """Carga el pipeline entrenado y sus metadatos desde disco."""
    modelo = joblib.load(MODELO_FILE)
    metadatos = json.loads(METADATOS_FILE.read_text(encoding="utf-8"))
    return modelo, metadatos


def normalizar(datos: pd.DataFrame) -> pd.DataFrame:
    """Prepara los datos de entrada aplicando la construccion de atributos oficial.

    Delega en el pipeline de atributos, que unifica los valores ausentes, convierte
    los tipos y calcula los atributos derivados. Duplicar esa logica aqui produciria
    tarde o temprano una divergencia silenciosa entre lo que ve el modelo en
    entrenamiento y lo que recibe en produccion.
    """
    return construir_atributos(datos)


def validar_columnas(datos: pd.DataFrame, esperadas: list[str]) -> list[str]:
    """Devuelve las columnas requeridas que faltan en el archivo cargado."""
    return [columna for columna in esperadas if columna not in datos.columns]


def predecir(modelo: Any, datos: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Calcula probabilidad de cancelacion y decision segun el umbral dado."""
    probabilidades = modelo.predict_proba(datos)[:, 1]
    resultado = pd.DataFrame(
        {
            "probabilidad_cancelacion": probabilidades.round(4),
            "en_riesgo": probabilidades >= umbral,
        }
    )
    return resultado


def construir_formulario(esperadas: list[str]) -> pd.DataFrame:
    """Dibuja el formulario de cliente y devuelve una fila lista para el modelo."""
    valores: dict[str, Any] = {}
    columnas = st.columns(3)

    for indice, atributo in enumerate(esperadas):
        contenedor = columnas[indice % 3]
        etiqueta = ETIQUETAS.get(atributo, atributo)
        if atributo in RANGOS_NUMERICOS:
            minimo, maximo, inicial = RANGOS_NUMERICOS[atributo]
            valores[atributo] = contenedor.number_input(
                etiqueta,
                min_value=minimo,
                max_value=maximo,
                value=inicial,
                key=f"campo_{atributo}",
            )
        else:
            valores[atributo] = contenedor.selectbox(
                etiqueta,
                OPCIONES[atributo],
                key=f"campo_{atributo}",
            )

    return pd.DataFrame([valores])


def mostrar_resultado_individual(probabilidad: float, umbral: float) -> None:
    """Presenta la probabilidad estimada y la decision correspondiente."""
    izquierda, derecha = st.columns(2)
    izquierda.metric("Probabilidad de cancelación", f"{probabilidad:.1%}")
    derecha.metric("Umbral de decisión", f"{umbral:.1%}")

    st.progress(min(probabilidad, 1.0))

    if probabilidad >= umbral:
        st.error("Cliente en riesgo de cancelación. Se recomienda acción de retención.")
    else:
        st.success("Cliente sin señales de riesgo según el modelo.")

    st.caption(
        "La probabilidad es la salida del modelo. La decisión depende del umbral, "
        "que es un parámetro de negocio: cuanto más caro sea perder un cliente frente "
        "al costo de una oferta de retención, más bajo debe fijarse."
    )


def pestana_individual(modelo: Any, esperadas: list[str], umbral: float) -> None:
    """Modo de prediccion para un unico cliente."""
    st.subheader("Evaluación de un cliente")
    st.write(
        "Complete los datos del cliente y obtenga la probabilidad estimada de que "
        "cancele su servicio en el próximo ciclo de facturación."
    )

    cliente = construir_formulario(esperadas)

    if st.button("Calcular predicción", type="primary"):
        resultado = predecir(modelo, normalizar(cliente), umbral)
        probabilidad = float(resultado["probabilidad_cancelacion"].iloc[0])
        mostrar_resultado_individual(probabilidad, umbral)

        with st.expander("Datos enviados al modelo"):
            st.dataframe(cliente.T.rename(columns={0: "valor"}))


def pestana_lote(modelo: Any, esperadas: list[str], umbral: float) -> None:
    """Modo de procesamiento de multiples clientes desde un archivo."""
    st.subheader("Procesamiento por lote")
    st.write(
        "Cargue un archivo CSV con varios clientes para obtener las predicciones de "
        "todos y descargar el resultado."
    )

    if EJEMPLO_FILE.exists():
        st.download_button(
            "Descargar archivo de ejemplo",
            data=EJEMPLO_FILE.read_bytes(),
            file_name="clientes_ejemplo.csv",
            mime="text/csv",
        )

    archivo = st.file_uploader("Archivo CSV de clientes", type=["csv"])
    if archivo is None:
        st.info("Cargue un archivo para continuar.")
        return

    datos = pd.read_csv(archivo)
    faltantes = validar_columnas(datos, esperadas)

    if faltantes:
        st.error(f"El archivo no tiene las columnas requeridas: {faltantes}")
        return

    st.success(f"Archivo válido: {len(datos)} registros.")

    preparados = normalizar(datos)
    resultado = predecir(modelo, preparados, umbral)

    salida = datos.copy()
    salida["probabilidad_cancelacion"] = resultado["probabilidad_cancelacion"].to_numpy()
    salida["en_riesgo"] = resultado["en_riesgo"].to_numpy()
    salida = salida.sort_values("probabilidad_cancelacion", ascending=False)

    en_riesgo = int(salida["en_riesgo"].sum())
    izquierda, centro, derecha = st.columns(3)
    izquierda.metric("Clientes evaluados", len(salida))
    centro.metric("En riesgo", en_riesgo)
    derecha.metric("Proporción en riesgo", f"{en_riesgo / len(salida):.1%}")

    st.dataframe(salida, height=400)

    st.download_button(
        "Descargar predicciones",
        data=salida.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_churn.csv",
        mime="text/csv",
        type="primary",
    )

    st.caption(
        "El resultado viene ordenado por probabilidad descendente, que es como el "
        "equipo de retención prioriza a quién contactar primero."
    )


def main() -> None:
    """Punto de entrada de la aplicacion."""
    st.set_page_config(page_title="Predicción de cancelación", page_icon="📊", layout="wide")
    st.title("Predicción de cancelación de clientes")

    modelo, metadatos = cargar_modelo()
    esperadas: list[str] = metadatos["columnas_entrada"]
    umbral_modelo = float(metadatos["umbral_decision"])

    with st.sidebar:
        st.header("Modelo")
        st.write(f"**Familia:** {metadatos['familia']}")
        metricas = metadatos["metricas_prueba"]
        st.write(f"**F1:** {metricas['f1']}")
        st.write(f"**ROC-AUC:** {metricas['roc_auc']}")
        st.write(f"**Exhaustividad:** {metricas['exhaustividad']}")

        st.divider()
        st.header("Umbral de decisión")
        umbral = st.slider(
            "Probabilidad a partir de la cual se marca al cliente",
            min_value=0.05,
            max_value=0.95,
            value=umbral_modelo,
            step=0.01,
        )
        st.caption(
            f"El valor que maximiza el F1 es {umbral_modelo:.3f}. Bajarlo detecta más "
            "cancelaciones a costa de contactar más clientes que no iban a irse."
        )

    individual, lote = st.tabs(["Predicción individual", "Procesamiento por lote"])

    with individual:
        pestana_individual(modelo, esperadas, umbral)

    with lote:
        pestana_lote(modelo, esperadas, umbral)


if __name__ == "__main__":
    main()
