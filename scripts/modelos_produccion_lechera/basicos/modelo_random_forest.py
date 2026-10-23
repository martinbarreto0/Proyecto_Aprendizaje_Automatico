"""Entrena un Random Forest básico, sin búsqueda ni ajuste de hiperparámetros."""

# Joblib permite guardar el pipeline completo después del entrenamiento.
from joblib import dump

# Pandas se utiliza para guardar las importancias de las variables.
import pandas as pd

# ColumnTransformer aplica una transformación distinta a categorías y números.
from sklearn.compose import ColumnTransformer

# SimpleImputer completa los faltantes numéricos usando la mediana de entrenamiento.
from sklearn.impute import SimpleImputer

# RandomForestRegressor es el primer modelo supervisado de la propuesta.
from sklearn.ensemble import RandomForestRegressor

# Pipeline mantiene juntos el preprocesamiento y el modelo.
from sklearn.pipeline import Pipeline

# OneHotEncoder transforma los códigos categóricos en columnas binarias.
from sklearn.preprocessing import OneHotEncoder

# Se importan las funciones y constantes compartidas con el modelo de mediana.
from datos_lecheria import (
    CARPETA_RESULTADOS,
    OBJETIVO,
    SEMILLA_PRINCIPAL,
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


def guardar_importancias(modelo):
    """Guarda la importancia de Random Forest por columna y por variable original."""
    preprocesamiento = modelo.named_steps["preprocesamiento"]
    bosque = modelo.named_steps["modelo"]
    tabla = pd.DataFrame(
        {
            "variable": preprocesamiento.get_feature_names_out(),
            "importancia": bosque.feature_importances_,
        }
    )
    tabla["variable_original"] = tabla["variable"].map(variable_original)
    tabla = tabla.sort_values("importancia", ascending=False).reset_index(drop=True)

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta_detalle = CARPETA_RESULTADOS / "importancia_random_forest.csv"
    tabla.to_csv(ruta_detalle, index=False, encoding="utf-8-sig")

    resumen = (
        tabla.groupby("variable_original", as_index=False)["importancia"]
        .sum()
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    ruta_resumen = CARPETA_RESULTADOS / "importancia_random_forest_por_variable.csv"
    resumen.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")
    return ruta_detalle, ruta_resumen


def main():
    """Construye los datos, entrena Random Forest y evalúa en validación."""

    # Se construye el mismo dataset utilizado por la referencia de mediana.
    dataset = construir_dataset()

    # Se repite exactamente la misma división aleatoria 70/15/15.
    entrenamiento, validacion, _prueba_reservada = dividir_dataset(dataset)

    # Se seleccionan únicamente las variables permitidas para entrenar.
    x_entrenamiento = entrenamiento[VARIABLES_MODELO]

    # Se seleccionan las mismas variables para validar.
    x_validacion = validacion[VARIABLES_MODELO]

    # Se extraen los litros reales del conjunto de entrenamiento.
    y_entrenamiento = entrenamiento[OBJETIVO]

    # Se extraen los litros reales del conjunto de validación.
    y_validacion = validacion[OBJETIVO]

    # Las categorías se codifican y las cantidades faltantes se completan con la mediana.
    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(handle_unknown="ignore"),
                VARIABLES_CATEGORICAS,
            ),
            (
                "numeros",
                SimpleImputer(strategy="median"),
                VARIABLES_NUMERICAS,
            ),
        ]
    )

    # Se crea Random Forest con sus valores predeterminados de scikit-learn.
    # Solo se fija la semilla para reproducibilidad y n_jobs para usar los núcleos disponibles.
    bosque = RandomForestRegressor(
        random_state=SEMILLA_PRINCIPAL,
        n_jobs=-1,
    )

    # El pipeline garantiza que el preprocesamiento se aprenda solo con entrenamiento.
    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("modelo", bosque),
        ]
    )

    # Se ajustan el preprocesamiento y Random Forest sobre el 70 % de entrenamiento.
    modelo.fit(x_entrenamiento, y_entrenamiento)

    # Se generan predicciones para el 15 % de validación.
    predicciones = modelo.predict(x_validacion)

    # Se calculan las mismas métricas utilizadas por el modelo de mediana.
    metricas = calcular_metricas("RandomForest_base", y_validacion, predicciones)

    # Se guardan las métricas y las predicciones de validación.
    ruta_metricas, ruta_predicciones = guardar_resultados(
        "random_forest",
        validacion,
        predicciones,
        metricas,
    )

    # Se calculan las importancias después de ajustar el bosque y el preprocesamiento.
    ruta_importancia, ruta_importancia_resumen = guardar_importancias(modelo)

    # Se crea la carpeta de resultados si todavía no existe.
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Se define el archivo que contendrá el pipeline completo.
    ruta_modelo = CARPETA_RESULTADOS / "modelo_random_forest.joblib"

    # Se guarda preprocesamiento y modelo en un único archivo reutilizable.
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

    # Se informa dónde revisar las importancias detalladas y agrupadas.
    print(f"Importancias guardadas en: {ruta_importancia}")
    print(f"Importancias por variable guardadas en: {ruta_importancia_resumen}")

    # Se informa la ubicación del pipeline serializado.
    print(f"Modelo guardado en: {ruta_modelo}")


# Este bloque evita que el entrenamiento se ejecute al importar el archivo.
if __name__ == "__main__":
    # El modelo se entrena únicamente cuando se ejecuta este script directamente.
    main()
