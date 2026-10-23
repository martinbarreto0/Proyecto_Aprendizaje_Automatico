"""Busca hiperparámetros de Random Forest usando solo entrenamiento y validación."""
import argparse
import pandas as pd

from funciones_busqueda import (
    RESULTADOS,
    actualizar_resumen,
    buscar_random_forest,
    preparar_datos,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hilos", type=int, default=4)
    args = parser.parse_args()

    entrenamiento, validacion, test_reservado = preparar_datos()
    metricas = buscar_random_forest(
        entrenamiento,
        validacion,
        test_reservado,
        args.hilos,
    )
    actualizar_resumen([metricas])
    print(pd.DataFrame([metricas]).to_string(index=False))
    print(f"Resultados guardados en: {RESULTADOS}")


if __name__ == "__main__":
    main()
