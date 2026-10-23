"""Consolida los ocho bloques y evalúa un único ganador en validación."""
import argparse
import json

import pandas as pd
from joblib import dump

from utilidades_bloques import (
    CONFIGURACIONES_POR_BLOQUE,
    OBJETIVO,
    RESULTADOS,
    RESULTADOS_BLOQUES,
    SCORING,
    TOTAL_BLOQUES,
    TOTAL_CONFIGURACIONES,
    VARIABLES_MODELO,
    cargar_plan,
    crear_pipeline_random_forest,
    guardar_csv_atomico,
    guardar_json_atomico,
    preparar_datos,
    ruta_resultado_bloque,
    validar_resultado_bloque,
    verificar_datos,
)
from funciones_busqueda import calcular_metricas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hilos",
        type=int,
        default=6,
        help="Hilos usados solo para reentrenar el ganador final.",
    )
    args = parser.parse_args()

    manifiesto, _, metadatos = cargar_plan()
    resultados = []
    faltantes = []
    for numero in range(1, TOTAL_BLOQUES + 1):
        ruta = ruta_resultado_bloque(numero)
        if not ruta.exists():
            faltantes.append(numero)
            continue
        bloque = pd.read_csv(
            ruta,
            dtype={
                "id_configuracion": str,
                "parametros_json": str,
                "sha256_grid": str,
                "sha256_entrenamiento": str,
                "version_sklearn": str,
            },
        )
        manifiesto_bloque = manifiesto[manifiesto["bloque"] == numero]
        validar_resultado_bloque(bloque, manifiesto_bloque, numero)
        if bloque["sha256_grid"].nunique() != 1:
            raise RuntimeError(f"El bloque {numero} contiene más de una versión.")
        if bloque["sha256_grid"].iloc[0] != metadatos["sha256_grid"]:
            raise RuntimeError(f"El bloque {numero} pertenece a otro grid.")
        if bloque["sha256_entrenamiento"].iloc[0] != metadatos["sha256_entrenamiento"]:
            raise RuntimeError(f"El bloque {numero} pertenece a otros datos.")
        if bloque["version_sklearn"].astype(str).iloc[0] != str(
            metadatos["version_sklearn"]
        ):
            raise RuntimeError(f"El bloque {numero} pertenece a otra versión.")
        resultados.append(bloque)

    if faltantes:
        raise RuntimeError(
            "No se puede consolidar. Faltan los bloques: "
            + ", ".join(map(str, faltantes))
        )

    consolidado = pd.concat(resultados, ignore_index=True)
    if len(consolidado) != TOTAL_CONFIGURACIONES:
        raise RuntimeError("La consolidación no contiene 384 configuraciones.")
    if consolidado["id_configuracion"].nunique() != TOTAL_CONFIGURACIONES:
        raise RuntimeError("La consolidación contiene configuraciones duplicadas.")
    if set(consolidado["id_configuracion"]) != set(manifiesto["id_configuracion"]):
        raise RuntimeError("La consolidación no coincide con el manifiesto.")

    for metrica in SCORING:
        columna_media = f"mean_test_{metrica}"
        consolidado[f"rank_test_{metrica}"] = (
            consolidado[columna_media]
            .rank(method="min", ascending=False)
            .astype(int)
        )
    consolidado = consolidado.sort_values(
        ["rank_test_mae", "id_configuracion"]
    ).reset_index(drop=True)
    ruta_consolidado = RESULTADOS_BLOQUES / "resultados_cv_consolidados.csv"
    guardar_csv_atomico(consolidado, ruta_consolidado)

    mejor = consolidado.iloc[0]
    fila_manifiesto = manifiesto[
        manifiesto["id_configuracion"] == mejor["id_configuracion"]
    ].iloc[0]
    mejores_parametros = json.loads(fila_manifiesto["parametros_json"])

    entrenamiento, validacion, test_reservado = preparar_datos()
    verificar_datos(metadatos, entrenamiento, validacion, test_reservado)

    modelo = crear_pipeline_random_forest(n_jobs_modelo=args.hilos)
    modelo.set_params(**mejores_parametros)
    modelo.fit(entrenamiento[VARIABLES_MODELO], entrenamiento[OBJETIVO])
    predicciones = modelo.predict(validacion[VARIABLES_MODELO])
    metricas = calcular_metricas(
        "random_forest_hiperparametros_bloques",
        validacion[OBJETIVO],
        predicciones,
    )

    guardar_csv_atomico(
        pd.DataFrame([metricas]),
        RESULTADOS_BLOQUES / "metricas_validacion_random_forest.csv",
    )
    salida = validacion.copy()
    salida["prediccion_litros"] = predicciones
    guardar_csv_atomico(
        salida,
        RESULTADOS_BLOQUES / "predicciones_validacion_random_forest.csv",
    )

    parametros_salida = {
        "modelo": "random_forest_hiperparametros_bloques",
        "criterio_seleccion_cv": "menor MAE medio en cinco pliegues",
        "configuraciones_consolidadas": len(consolidado),
        "id_configuracion_ganadora": mejor["id_configuracion"],
        "mejores_parametros": mejores_parametros,
        "mae_medio_cv": -float(mejor["mean_test_mae"]),
        "filas_test_reservado_sin_evaluar": len(test_reservado),
        "test_evaluado": False,
    }
    guardar_json_atomico(
        parametros_salida,
        RESULTADOS_BLOQUES / "mejores_parametros_random_forest.json",
    )

    ruta_temporal = RESULTADOS_BLOQUES / "modelo_random_forest.joblib.tmp"
    ruta_modelo = RESULTADOS_BLOQUES / "modelo_random_forest.joblib"
    dump(modelo, ruta_temporal)
    ruta_temporal.replace(ruta_modelo)

    ruta_base = RESULTADOS.parent / "modelos_basicos" / "metricas_random_forest.csv"
    if ruta_base.exists():
        base = pd.read_csv(ruta_base).iloc[0]
        ajustado = pd.Series(metricas)
        comparacion = pd.DataFrame(
            [
                {
                    "modelo": "random_forest",
                    "mae_base": base["mae_litros"],
                    "mae_ajustado": ajustado["mae_litros"],
                    "delta_mae": ajustado["mae_litros"] - base["mae_litros"],
                    "rmse_base": base["rmse_litros"],
                    "rmse_ajustado": ajustado["rmse_litros"],
                    "delta_rmse": ajustado["rmse_litros"] - base["rmse_litros"],
                    "r2_base": base["r2"],
                    "r2_ajustado": ajustado["r2"],
                    "wape_base": base["wape_porcentaje"],
                    "wape_ajustado": ajustado["wape_porcentaje"],
                }
            ]
        )
        guardar_csv_atomico(
            comparacion,
            RESULTADOS_BLOQUES / "comparacion_con_random_forest_basico.csv",
        )

    print(pd.DataFrame([metricas]).to_string(index=False))
    print(f"Resultados consolidados en: {RESULTADOS_BLOQUES}")
    print(f"Test reservado sin evaluar: {len(test_reservado)} filas")


if __name__ == "__main__":
    main()
