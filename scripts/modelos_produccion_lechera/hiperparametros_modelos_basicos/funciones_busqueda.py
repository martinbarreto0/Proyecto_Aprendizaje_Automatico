"""Funciones compartidas por las búsquedas de Ridge y Random Forest."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(__file__).resolve().parent
BASICOS = BASE.parent / "basicos"
sys.path.insert(0, str(BASICOS))

from datos_lecheria import (  # noqa: E402
    CARPETA_PAA,
    OBJETIVO,
    SEMILLA_PRINCIPAL,
    VARIABLES_CATEGORICAS,
    VARIABLES_MODELO,
    VARIABLES_NUMERICAS,
    calcular_metricas,
    construir_dataset,
    dividir_dataset,
)


RESULTADOS = CARPETA_PAA / "reports" / "hiperparametros_modelos_basicos"


def wape(y_real, y_predicho):
    """Calcula WAPE; GridSearchCV cambiará el signo para maximizar el scorer."""
    denominador = np.abs(y_real).sum()
    return np.abs(y_real - y_predicho).sum() / denominador * 100


SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "wape": make_scorer(wape, greater_is_better=False),
}


def preprocesamiento_ridge():
    """Reproduce exactamente el preprocesamiento del Ridge básico."""
    numerico = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                VARIABLES_CATEGORICAS,
            ),
            ("numeros", numerico, VARIABLES_NUMERICAS),
        ]
    )


def preprocesamiento_random_forest():
    """Reproduce exactamente el preprocesamiento del Random Forest básico."""
    return ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(handle_unknown="ignore"),
                VARIABLES_CATEGORICAS,
            ),
            ("numeros", SimpleImputer(strategy="median"), VARIABLES_NUMERICAS),
        ]
    )


def espacio_ridge():
    """Devuelve las 228 combinaciones exhaustivas definidas para Ridge."""
    alphas = [
        0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3,
        10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000,
    ]
    return {
        "modelo__alpha": alphas,
        "modelo__fit_intercept": [True, False],
        "modelo__tol": [0.0001, 0.001, 0.01],
        "modelo__solver": ["auto", "lsqr"],
    }


def espacio_random_forest():
    """Devuelve 384 combinaciones exhaustivas y válidas de Random Forest."""
    parametros_comunes = {
        "modelo__n_estimators": [300, 700],
        "modelo__max_depth": [None, 20, 40, 60],
        "modelo__min_samples_split": [2, 10],
        "modelo__min_samples_leaf": [1, 4],
        "modelo__max_features": ["sqrt", 0.6],
        "modelo__criterion": ["squared_error", "poisson"],
    }
    return [
        {
            **parametros_comunes,
            "modelo__bootstrap": [True],
            "modelo__max_samples": [0.75, 1.0],
        },
        {
            **parametros_comunes,
            "modelo__bootstrap": [False],
            "modelo__max_samples": [None],
        },
    ]


def guardar_busqueda(nombre, busqueda, validacion, test_reservado, resultados_cv=None):
    """Evalúa el ganador solo en validación y guarda todos los artefactos."""
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    predicciones = busqueda.best_estimator_.predict(validacion[VARIABLES_MODELO])
    metricas = calcular_metricas(
        f"{nombre}_hiperparametros", validacion[OBJETIVO], predicciones
    )
    pd.DataFrame([metricas]).to_csv(
        RESULTADOS / f"metricas_validacion_{nombre}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    salida = validacion.copy()
    salida["prediccion_litros"] = predicciones
    salida.to_csv(
        RESULTADOS / f"predicciones_validacion_{nombre}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cv = (pd.DataFrame(busqueda.cv_results_)
          if resultados_cv is None else resultados_cv.copy())
    cv = cv.sort_values(["rank_test_mae"])
    cv.to_csv(
        RESULTADOS / f"resultados_cv_{nombre}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    parametros = {
        "modelo": nombre,
        "criterio_seleccion_cv": "menor MAE medio en 5 pliegues de entrenamiento",
        "mejores_parametros": busqueda.best_params_,
        "filas_test_reservado_sin_evaluar": len(test_reservado),
        "test_evaluado": False,
    }
    (RESULTADOS / f"mejores_parametros_{nombre}.json").write_text(
        json.dumps(parametros, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dump(busqueda.best_estimator_, RESULTADOS / f"modelo_{nombre}.joblib")
    return metricas


def buscar_ridge(entrenamiento, validacion, test_reservado, hilos):
    """Ejecuta la búsqueda exhaustiva de Ridge solo dentro de entrenamiento."""
    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento_ridge()),
            ("modelo", Ridge()),
        ]
    )
    busqueda = GridSearchCV(
        pipeline,
        espacio_ridge(),
        scoring=SCORING,
        refit="mae",
        cv=5,
        n_jobs=hilos,
        return_train_score=False,
        verbose=2,
    )
    busqueda.fit(entrenamiento[VARIABLES_MODELO], entrenamiento[OBJETIVO])
    return guardar_busqueda("ridge", busqueda, validacion, test_reservado)


def buscar_random_forest(entrenamiento, validacion, test_reservado, hilos):
    """Ejecuta el grid exhaustivo de Random Forest solo en entrenamiento."""
    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento_random_forest()),
            (
                "modelo",
                RandomForestRegressor(random_state=SEMILLA_PRINCIPAL, n_jobs=1),
            ),
        ]
    )
    busqueda = GridSearchCV(
        pipeline,
        espacio_random_forest(),
        scoring=SCORING,
        refit="mae",
        cv=5,
        n_jobs=hilos,
        return_train_score=False,
        verbose=2,
    )
    busqueda.fit(entrenamiento[VARIABLES_MODELO], entrenamiento[OBJETIVO])
    return guardar_busqueda("random_forest", busqueda, validacion, test_reservado)


def actualizar_resumen(metricas):
    """Consolida las búsquedas disponibles y las compara con los básicos."""
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    nuevas = []
    for ruta in sorted(RESULTADOS.glob("metricas_validacion_*.csv")):
        nuevas.append(pd.read_csv(ruta))
    if not nuevas:
        return
    resumen = pd.concat(nuevas, ignore_index=True)
    resumen.to_csv(RESULTADOS / "resumen_metricas_validacion.csv", index=False, encoding="utf-8-sig")

    basicos = CARPETA_PAA / "reports" / "modelos_basicos"
    comparacion = []
    for nombre, archivo_base in [
        ("ridge", "metricas_ridge.csv"),
        ("random_forest", "metricas_random_forest.csv"),
    ]:
        nueva = RESULTADOS / f"metricas_validacion_{nombre}.csv"
        anterior = basicos / archivo_base
        if nueva.exists() and anterior.exists():
            base = pd.read_csv(anterior).iloc[0]
            ajustado = pd.read_csv(nueva).iloc[0]
            comparacion.append(
                {
                    "modelo": nombre,
                    "mae_base": base.mae_litros,
                    "mae_ajustado": ajustado.mae_litros,
                    "delta_mae": ajustado.mae_litros - base.mae_litros,
                    "rmse_base": base.rmse_litros,
                    "rmse_ajustado": ajustado.rmse_litros,
                    "delta_rmse": ajustado.rmse_litros - base.rmse_litros,
                    "r2_base": base.r2,
                    "r2_ajustado": ajustado.r2,
                    "wape_base": base.wape_porcentaje,
                    "wape_ajustado": ajustado.wape_porcentaje,
                }
            )
    pd.DataFrame(comparacion).to_csv(
        RESULTADOS / "comparacion_con_modelos_basicos.csv",
        index=False,
        encoding="utf-8-sig",
    )


def preparar_datos():
    """Construye y divide el dataset, dejando el test identificado y sin usar."""
    dataset = construir_dataset()
    entrenamiento, validacion, test_reservado = dividir_dataset(dataset)
    print(
        f"[control] test reservado: {len(test_reservado)} filas; "
        "no se calcularán predicciones ni métricas sobre él."
    )
    return entrenamiento, validacion, test_reservado
