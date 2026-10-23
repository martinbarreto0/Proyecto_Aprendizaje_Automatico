"""Entrena CatBoost básico con el mismo dataset y la misma división 70/15/15."""

import argparse
import json
from time import perf_counter

import pandas as pd


from datos_lecheria import (
    CARPETA_PAA,
    CLAVES_UNION,
    OBJETIVO,
    SEMILLA_PRINCIPAL,
    VARIABLES_CATEGORICAS,
    VARIABLES_MODELO,
    VARIABLES_NUMERICAS,
    calcular_metricas,
    construir_dataset,
    dividir_dataset,
)


RESULTADOS = CARPETA_PAA / "reports" / "modelos_catboost" / "basico"


def preparar_variables(datos):
    """Conserva las 16 variables y los tipos que CatBoost necesita."""
    variables = datos[VARIABLES_MODELO].copy()

    # CatBoost recibe las categorías como texto, incluido el faltante explícito.
    for columna in VARIABLES_CATEGORICAS:
        variables[columna] = variables[columna].fillna("SIN_DATO").astype(str)

    # Los faltantes numéricos se dejan como NaN para que CatBoost los procese.
    for columna in VARIABLES_NUMERICAS:
        variables[columna] = pd.to_numeric(
            variables[columna], errors="coerce"
        ).astype("float64")

    return variables


def crear_modelo_basico(hilos):
    """Crea la misma configuración básica para todos los experimentos CatBoost."""
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        random_seed=SEMILLA_PRINCIPAL,
        thread_count=hilos,
        verbose=False,
        allow_writing_files=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hilos", type=int, default=-1,
        help="Hilos de CatBoost; -1 usa todos los disponibles (predeterminado).",
    )
    args = parser.parse_args()
    if args.hilos == 0 or args.hilos < -1:
        parser.error("--hilos debe ser -1 o un entero positivo")

    try:
        from catboost import __version__ as version_catboost
    except ImportError as error:
        raise SystemExit(
            "Falta CatBoost. Instalar con: "
            "PAA/.venv_modelos/Scripts/python.exe -m pip install -r "
            "PAA/scripts/modelos_produccion_lechera/catboost/requirements.txt"
        ) from error

    # Se reutilizan sin cambios la construcción y la división de los modelos base.
    dataset = construir_dataset()
    entrenamiento, validacion, test_reservado = dividir_dataset(dataset)
    x_entrenamiento = preparar_variables(entrenamiento)
    x_validacion = preparar_variables(validacion)

    # La validación no interviene en el ajuste ni se pasa como eval_set.
    modelo = crear_modelo_basico(args.hilos)
    inicio = perf_counter()
    modelo.fit(
        x_entrenamiento,
        entrenamiento[OBJETIVO],
        cat_features=VARIABLES_CATEGORICAS,
    )
    segundos_entrenamiento = perf_counter() - inicio

    predicciones = modelo.predict(x_validacion)
    metricas = calcular_metricas(
        "CatBoost_base", validacion[OBJETIVO], predicciones
    )

    # Una fila por perfil de validación, con el mismo signo de error que los básicos.
    tabla_predicciones = validacion[CLAVES_UNION].copy()
    tabla_predicciones["Litros_reales"] = validacion[OBJETIVO].to_numpy()
    tabla_predicciones["Litros_predichos"] = predicciones
    tabla_predicciones["Error"] = (
        tabla_predicciones["Litros_predichos"] - tabla_predicciones["Litros_reales"]
    )

    # La importancia se obtiene del modelo entrenado; no utiliza validación.
    importancia = pd.DataFrame({
        "variable": VARIABLES_MODELO,
        "tipo": [
            "categorica" if variable in VARIABLES_CATEGORICAS else "numerica"
            for variable in VARIABLES_MODELO
        ],
        "importancia": modelo.get_feature_importance(type="PredictionValuesChange"),
    })
    importancia = importancia.sort_values(
        "importancia", ascending=False
    ).reset_index(drop=True)
    importancia.insert(0, "posicion", importancia.index + 1)

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metricas]).to_csv(
        RESULTADOS / "metricas_catboost_basico.csv",
        index=False, encoding="utf-8-sig",
    )
    tabla_predicciones.to_csv(
        RESULTADOS / "predicciones_catboost_basico.csv",
        index=False, encoding="utf-8-sig",
    )
    importancia.to_csv(
        RESULTADOS / "importancia_catboost_basico.csv",
        index=False, encoding="utf-8-sig",
    )
    modelo.save_model(str(RESULTADOS / "modelo_catboost_basico.cbm"))

    metadatos = {
        "modelo": "CatBoost_base",
        "version_catboost": version_catboost,
        "filas_dataset": len(dataset),
        "filas_entrenamiento": len(entrenamiento),
        "filas_validacion": len(validacion),
        "filas_test_reservado": len(test_reservado),
        "test_evaluado": False,
        "segundos_entrenamiento": segundos_entrenamiento,
        "variables_modelo": VARIABLES_MODELO,
        "variables_categoricas": VARIABLES_CATEGORICAS,
        "tipo_importancia": "PredictionValuesChange",
        "parametros_efectivos": modelo.get_all_params(),
    }
    (RESULTADOS / "metadatos_catboost_basico.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nMétricas de validación:")
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor}")
    print(f"Importancia y modelo guardados en: {RESULTADOS}")
    print("Test reservado sin evaluar.")


if __name__ == "__main__":
    main()
