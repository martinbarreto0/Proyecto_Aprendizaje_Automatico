"""Ejecuta un bloque de 48 configuraciones de Random Forest."""
import argparse
from time import perf_counter

import pandas as pd
from sklearn.model_selection import GridSearchCV

from utilidades_bloques import (
    CONFIGURACIONES_POR_BLOQUE,
    OBJETIVO,
    SCORING,
    TOTAL_BLOQUES,
    VARIABLES_MODELO,
    cargar_plan,
    crear_pipeline_random_forest,
    crear_splits_cv,
    fecha_utc,
    grid_desde_manifiesto,
    guardar_csv_atomico,
    guardar_json_atomico,
    id_configuracion,
    parametros_a_json,
    preparar_datos,
    ruta_estado_bloque,
    ruta_resultado_bloque,
    validar_resultado_bloque,
    verificar_datos,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bloque", type=int, required=True, choices=range(1, 9))
    parser.add_argument("--hilos", type=int, default=6)
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Vuelve a ejecutar un bloque que ya terminó correctamente.",
    )
    args = parser.parse_args()

    manifiesto, asignacion, metadatos = cargar_plan()
    manifiesto_bloque = manifiesto[manifiesto["bloque"] == args.bloque].copy()
    if len(manifiesto_bloque) != CONFIGURACIONES_POR_BLOQUE:
        raise RuntimeError("El manifiesto del bloque no contiene 48 configuraciones.")

    entrenamiento, validacion, test_reservado = preparar_datos()
    verificar_datos(metadatos, entrenamiento, validacion, test_reservado)

    ruta_salida = ruta_resultado_bloque(args.bloque)
    if ruta_salida.exists() and not args.forzar:
        resultado_existente = pd.read_csv(
            ruta_salida,
            dtype={
                "id_configuracion": str,
                "parametros_json": str,
                "sha256_grid": str,
                "sha256_entrenamiento": str,
                "version_sklearn": str,
            },
        )
        validar_resultado_bloque(
            resultado_existente, manifiesto_bloque, args.bloque
        )
        controles = {
            "sha256_grid": metadatos["sha256_grid"],
            "sha256_entrenamiento": metadatos["sha256_entrenamiento"],
            "version_sklearn": metadatos["version_sklearn"],
        }
        for columna, esperado in controles.items():
            if resultado_existente[columna].nunique() != 1:
                raise RuntimeError(f"El bloque contiene varios valores de {columna}.")
            if str(resultado_existente[columna].iloc[0]) != str(esperado):
                raise RuntimeError(f"El bloque existente no coincide en {columna}.")
        print(f"El bloque {args.bloque} ya está completo: {ruta_salida}")
        return

    splits = crear_splits_cv(asignacion, len(entrenamiento))

    busqueda = GridSearchCV(
        crear_pipeline_random_forest(n_jobs_modelo=1),
        grid_desde_manifiesto(manifiesto_bloque),
        scoring=SCORING,
        refit=False,
        cv=splits,
        n_jobs=args.hilos,
        return_train_score=False,
        verbose=2,
    )

    inicio = perf_counter()
    busqueda.fit(entrenamiento[VARIABLES_MODELO], entrenamiento[OBJETIVO])
    duracion = perf_counter() - inicio

    resultado = pd.DataFrame(busqueda.cv_results_)
    resultado["id_configuracion"] = resultado["params"].map(id_configuracion)
    resultado["parametros_json"] = resultado["params"].map(parametros_a_json)
    resultado["bloque"] = args.bloque
    resultado["sha256_grid"] = metadatos["sha256_grid"]
    resultado["sha256_entrenamiento"] = metadatos["sha256_entrenamiento"]
    resultado["version_sklearn"] = metadatos["version_sklearn"]
    for metrica in SCORING:
        columna = f"rank_test_{metrica}"
        resultado = resultado.rename(columns={columna: columna + "_bloque"})

    validar_resultado_bloque(resultado, manifiesto_bloque, args.bloque)
    guardar_csv_atomico(resultado, ruta_salida)

    mejor_indice = resultado["mean_test_mae"].idxmax()
    mejor = resultado.loc[mejor_indice]
    estado = {
        "bloque": args.bloque,
        "estado": "completo",
        "terminado_utc": fecha_utc(),
        "duracion_segundos": duracion,
        "hilos": args.hilos,
        "configuraciones": len(resultado),
        "ajustes_cv": len(resultado) * metadatos["pliegues"],
        "mejor_id_local": mejor["id_configuracion"],
        "mejor_mae_cv_local": -float(mejor["mean_test_mae"]),
        "test_evaluado": False,
    }
    guardar_json_atomico(estado, ruta_estado_bloque(args.bloque))

    print(f"Bloque {args.bloque}/{TOTAL_BLOQUES} terminado.")
    print(f"Configuraciones: {len(resultado)}")
    print(f"Ajustes de CV: {len(resultado) * metadatos['pliegues']}")
    print(f"Duración: {duracion / 3600:.2f} horas")
    print(f"Resultado: {ruta_salida}")


if __name__ == "__main__":
    main()
