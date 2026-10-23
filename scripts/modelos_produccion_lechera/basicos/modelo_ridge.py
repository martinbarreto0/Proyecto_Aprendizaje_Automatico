"""Entrena una regresión Ridge básica, explicable y sin ajustar hiperparámetros."""

# Joblib permite guardar el pipeline completo después del entrenamiento.
from joblib import dump

# Pandas se utiliza para guardar los coeficientes en una tabla CSV.
import pandas as pd

# ColumnTransformer aplica transformaciones diferentes a categorías y números.
from sklearn.compose import ColumnTransformer

# SimpleImputer completa los valores numéricos faltantes.
from sklearn.impute import SimpleImputer

# Ridge es una regresión lineal con regularización L2.
from sklearn.linear_model import Ridge

# Pipeline mantiene ordenados el preprocesamiento y el modelo.
from sklearn.pipeline import Pipeline

# OneHotEncoder convierte categorías en columnas y StandardScaler escala números.
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Se reutilizan las mismas funciones y constantes de los demás modelos.
from datos_lecheria import (
    CARPETA_RESULTADOS,
    OBJETIVO,
    VARIABLES_CATEGORICAS,
    VARIABLES_MODELO,
    VARIABLES_NUMERICAS,
    calcular_metricas,
    construir_dataset,
    dividir_dataset,
    guardar_resultados,
)


def variable_original(nombre):
    """Recupera la variable antes del one-hot encoding."""
    nombre = nombre.split("__", maxsplit=1)[-1]
    for variable in VARIABLES_CATEGORICAS:
        if nombre == variable or nombre.startswith(f"{variable}_"):
            return variable
    return nombre


def guardar_coeficientes(modelo):
    """Guarda coeficientes e importancia absoluta por variable transformada y original."""

    # Se obtiene el preprocesamiento ya aprendido con los datos de entrenamiento.
    preprocesamiento = modelo.named_steps["preprocesamiento"]

    # Se recupera el nombre final de cada columna después del preprocesamiento.
    nombres_variables = preprocesamiento.get_feature_names_out()

    # Se obtiene el coeficiente que Ridge asignó a cada columna transformada.
    coeficientes = modelo.named_steps["modelo"].coef_

    # Se construye una tabla con una fila por variable transformada.
    tabla = pd.DataFrame(
        {
            "variable": nombres_variables,
            "coeficiente": coeficientes,
        }
    )

    # El signo facilita distinguir asociaciones positivas y negativas.
    tabla["signo"] = tabla["coeficiente"].map(
        lambda valor: "positivo" if valor >= 0 else "negativo"
    )

    # El valor absoluto permite ordenar sin perder el signo original.
    tabla["coeficiente_absoluto"] = tabla["coeficiente"].abs()

    # Se identifica la variable de origen de cada categoría creada por one-hot encoding.
    tabla["variable_original"] = tabla["variable"].map(variable_original)

    # Las variables con coeficientes de mayor magnitud aparecen primero.
    tabla = tabla.sort_values(
        "coeficiente_absoluto",
        ascending=False,
    ).reset_index(drop=True)

    # Se crea la carpeta de resultados si todavía no existe.
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Se define la ubicación del CSV explicativo.
    ruta_coeficientes = CARPETA_RESULTADOS / "coeficientes_ridge.csv"

    # Se guarda la tabla para poder revisarla sin cargar el modelo.
    tabla.to_csv(ruta_coeficientes, index=False, encoding="utf-8-sig")

    # Para una variable categórica se suman las magnitudes de todas sus categorías.
    resumen = (
        tabla.groupby("variable_original", as_index=False)["coeficiente_absoluto"]
        .sum()
        .rename(columns={"coeficiente_absoluto": "importancia_absoluta"})
        .sort_values("importancia_absoluta", ascending=False)
        .reset_index(drop=True)
    )
    ruta_importancia = CARPETA_RESULTADOS / "importancia_ridge_por_variable.csv"
    resumen.to_csv(ruta_importancia, index=False, encoding="utf-8-sig")

    # Se devuelve la ruta para mostrarla al finalizar.
    return ruta_coeficientes, ruta_importancia


def main():
    """Construye los datos, entrena Ridge y evalúa únicamente en validación."""

    # Se construye exactamente el mismo dataset de los modelos anteriores.
    dataset = construir_dataset()

    # Se repite la misma división aleatoria 70/15/15.
    entrenamiento, validacion, _prueba_reservada = dividir_dataset(dataset)

    # Se seleccionan las variables permitidas para el entrenamiento.
    x_entrenamiento = entrenamiento[VARIABLES_MODELO]

    # Se seleccionan las mismas variables para la validación.
    x_validacion = validacion[VARIABLES_MODELO]

    # Se extraen los litros reales utilizados para ajustar Ridge.
    y_entrenamiento = entrenamiento[OBJETIVO]

    # Se extraen los litros reales utilizados para calcular las métricas.
    y_validacion = validacion[OBJETIVO]

    # Las cantidades numéricas primero se imputan y después se estandarizan.
    preprocesamiento_numerico = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
        ]
    )

    # Se define cómo preparar cada grupo de variables.
    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "categorias",
                # Se quita la primera categoría para usarla como referencia.
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                VARIABLES_CATEGORICAS,
            ),
            (
                "numeros",
                preprocesamiento_numerico,
                VARIABLES_NUMERICAS,
            ),
        ]
    )

    # Se crea Ridge con alpha=1.0, que es el valor inicial predeterminado.
    # Este valor no fue buscado ni ajustado con los datos de validación.
    regresion = Ridge(alpha=1.0)

    # El pipeline evita aprender imputaciones, escalas o categorías desde validación.
    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("modelo", regresion),
        ]
    )

    # Se ajustan el preprocesamiento y Ridge usando solo entrenamiento.
    modelo.fit(x_entrenamiento, y_entrenamiento)

    # Se generan predicciones para el conjunto de validación.
    predicciones = modelo.predict(x_validacion)

    # Se calculan las mismas métricas usadas por la mediana y Random Forest.
    metricas = calcular_metricas("Ridge_base", y_validacion, predicciones)

    # Se guardan las métricas y las predicciones de validación.
    ruta_metricas, ruta_predicciones = guardar_resultados(
        "ridge",
        validacion,
        predicciones,
        metricas,
    )

    # Se guardan los coeficientes para interpretar el modelo lineal.
    ruta_coeficientes, ruta_importancia = guardar_coeficientes(modelo)

    # Se define el archivo que contendrá el pipeline completo.
    ruta_modelo = CARPETA_RESULTADOS / "modelo_ridge.joblib"

    # Se guarda el preprocesamiento y Ridge en un único archivo reutilizable.
    dump(modelo, ruta_modelo)

    # Se muestran las métricas principales en la terminal.
    print("\nMétricas de validación:")

    # Se imprime cada métrica en una línea separada.
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor}")

    # Se informa la ubicación del CSV de métricas.
    print(f"\nMétricas guardadas en: {ruta_metricas}")

    # Se informa la ubicación del CSV de predicciones.
    print(f"Predicciones guardadas en: {ruta_predicciones}")

    # Se informa la ubicación de la tabla explicativa.
    print(f"Coeficientes guardados en: {ruta_coeficientes}")

    # Se informa el resumen de importancia agrupado por variable original.
    print(f"Importancias por variable guardadas en: {ruta_importancia}")

    # Se informa la ubicación del pipeline serializado.
    print(f"Modelo guardado en: {ruta_modelo}")


# Este bloque evita que el entrenamiento se ejecute al importar el archivo.
if __name__ == "__main__":
    # El modelo se entrena únicamente al ejecutar este script directamente.
    main()
