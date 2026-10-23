"""Compara Ridge y Random Forest usando la misma construcción de datos base.

La selección de variables se aplica después de construir y dividir el dataset.
No modifica los scripts ni los resultados de modelos_produccion_lechera/basicos.
"""
from pathlib import Path
import sys

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent
BASICOS = BASE.parent / "basicos"
sys.path.insert(0, str(BASICOS))

# Se reutiliza el mismo constructor y la misma división aleatoria de los modelos base.
from datos_lecheria import (  # noqa: E402
    CARPETA_PAA,
    OBJETIVO,
    SEMILLA_PRINCIPAL,
    calcular_metricas,
    construir_dataset,
    dividir_dataset,
)


RESULTADOS = CARPETA_PAA / "reports" / "modelos_seleccion_variables"

# Ridge conserva las categorías con aporte conjunto y las numéricas más relevantes.
RIDGE_CATEGORICAS = [
    "EstratoCodigo",
    "DepartamentoCodigo",
    "NaturalezaJuridicaCodigo",
    "GiroCodigo",
]
RIDGE_NUMERICAS = [
    "vacas_ordene",
    "UnidadesGanaderas",
    "SuperficieGanadera",
    "total_bovinos_leche",
    "vaquillonas_1_2",
    "Superficie",
    "terneros_menos_1",
]

# Random Forest conserva las cinco variables que concentraron 96,8 % de importancia.
RF_NUMERICAS = [
    "vacas_ordene",
    "UnidadesGanaderas",
    "terneras_menos_1",
    "total_bovinos_leche",
    "vacas_secas",
]


def guardar_resultado(nombre, modelo, validacion, predicciones, metricas, variables):
    """Guarda métricas, predicciones, modelo y variables de una prueba aislada."""
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metricas]).to_csv(
        RESULTADOS / f"metricas_{nombre}.csv", index=False, encoding="utf-8-sig"
    )
    salida = validacion.copy()
    salida["prediccion_litros"] = predicciones
    salida.to_csv(
        RESULTADOS / f"predicciones_{nombre}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"variable": variables}).to_csv(
        RESULTADOS / f"variables_{nombre}.csv", index=False, encoding="utf-8-sig"
    )
    dump(modelo, RESULTADOS / f"modelo_{nombre}.joblib")


def entrenar_ridge(entrenamiento, validacion):
    """Entrena Ridge después de seleccionar las variables indicadas para Ridge."""
    variables = RIDGE_CATEGORICAS + RIDGE_NUMERICAS
    preprocesamiento = ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                RIDGE_CATEGORICAS,
            ),
            (
                "numeros",
                Pipeline(
                    steps=[
                        ("imputacion", SimpleImputer(strategy="median")),
                        ("escala", StandardScaler()),
                    ]
                ),
                RIDGE_NUMERICAS,
            ),
        ]
    )
    modelo = Pipeline(
        steps=[("preprocesamiento", preprocesamiento), ("modelo", Ridge(alpha=1.0))]
    )
    modelo.fit(entrenamiento[variables], entrenamiento[OBJETIVO])
    predicciones = modelo.predict(validacion[variables])
    metricas = calcular_metricas("Ridge_variables_seleccionadas", validacion[OBJETIVO], predicciones)
    guardar_resultado("ridge_seleccion", modelo, validacion, predicciones, metricas, variables)
    return metricas


def entrenar_random_forest(entrenamiento, validacion):
    """Entrena Random Forest después de seleccionar las cinco variables principales."""
    preprocesamiento = SimpleImputer(strategy="median")
    modelo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento),
            ("modelo", RandomForestRegressor(random_state=SEMILLA_PRINCIPAL, n_jobs=-1)),
        ]
    )
    modelo.fit(entrenamiento[RF_NUMERICAS], entrenamiento[OBJETIVO])
    predicciones = modelo.predict(validacion[RF_NUMERICAS])
    metricas = calcular_metricas(
        "RandomForest_variables_seleccionadas", validacion[OBJETIVO], predicciones
    )
    guardar_resultado("random_forest_seleccion", modelo, validacion, predicciones, metricas, RF_NUMERICAS)
    return metricas


def main():
    """Construye una vez el dataset base y compara los dos modelos seleccionados."""
    dataset = construir_dataset()
    entrenamiento, validacion, _prueba_reservada = dividir_dataset(dataset)
    resultados = [
        entrenar_ridge(entrenamiento, validacion),
        entrenar_random_forest(entrenamiento, validacion),
    ]
    pd.DataFrame(resultados).to_csv(
        RESULTADOS / "resumen_metricas.csv", index=False, encoding="utf-8-sig"
    )
    print(pd.DataFrame(resultados).to_string(index=False))
    print(f"Resultados guardados en: {RESULTADOS}")


if __name__ == "__main__":
    main()
