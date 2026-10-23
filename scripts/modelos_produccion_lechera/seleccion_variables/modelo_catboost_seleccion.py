"""Compara CatBoost completo con subconjuntos elegidos por su propia importancia."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

CARPETA_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA_SCRIPT.parent / "basicos"))

from modelo_catboost_basico import crear_modelo_basico, preparar_variables  # noqa: E402
from datos_lecheria import (  # noqa: E402
    CARPETA_PAA,
    CLAVES_UNION,
    OBJETIVO,
    VARIABLES_CATEGORICAS,
    VARIABLES_MODELO,
    calcular_metricas,
    construir_dataset,
    dividir_dataset,
)


RESULTADOS = CARPETA_PAA / "reports" / "modelos_catboost" / "seleccion_variables"
TAMANIOS = (5, 8, 12)


def entrenar_y_guardar(nombre, columnas, x_entrenamiento, entrenamiento,
                      x_validacion, validacion, hilos):
    """Entrena solo con train, mide en validación y guarda una variante."""
    categorias = [col for col in VARIABLES_CATEGORICAS if col in columnas]
    modelo = crear_modelo_basico(hilos)

    inicio = perf_counter()
    modelo.fit(
        x_entrenamiento[columnas],
        entrenamiento[OBJETIVO],
        cat_features=categorias,
    )
    segundos = perf_counter() - inicio

    predicciones = modelo.predict(x_validacion[columnas])
    metricas = calcular_metricas(
        f"CatBoost_{nombre}", validacion[OBJETIVO], predicciones
    )
    metricas["numero_variables"] = len(columnas)
    metricas["segundos_entrenamiento"] = segundos

    salida = validacion[CLAVES_UNION].copy()
    salida["Litros_reales"] = validacion[OBJETIVO].to_numpy()
    salida["Litros_predichos"] = predicciones
    salida["Error"] = salida["Litros_predichos"] - salida["Litros_reales"]

    pd.DataFrame([metricas]).to_csv(
        RESULTADOS / f"metricas_{nombre}.csv", index=False, encoding="utf-8-sig"
    )
    salida.to_csv(
        RESULTADOS / f"predicciones_{nombre}.csv", index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"variable": columnas}).to_csv(
        RESULTADOS / f"variables_{nombre}.csv", index=False, encoding="utf-8-sig"
    )
    modelo.save_model(str(RESULTADOS / f"modelo_{nombre}.cbm"))
    return modelo, metricas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hilos", type=int, default=-1,
        help="Hilos por modelo; -1 usa todos los disponibles (predeterminado).",
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

    # La selección se hace después de construir y dividir el dataset completo.
    dataset = construir_dataset()
    entrenamiento, validacion, test_reservado = dividir_dataset(dataset)
    x_entrenamiento = preparar_variables(entrenamiento)
    x_validacion = preparar_variables(validacion)
    RESULTADOS.mkdir(parents=True, exist_ok=True)

    # El modelo completo genera el ranking, sin consultar validación ni test.
    completo, metrica_completo = entrenar_y_guardar(
        "completo", list(VARIABLES_MODELO), x_entrenamiento, entrenamiento,
        x_validacion, validacion, args.hilos,
    )
    importancia = pd.DataFrame({
        "variable": VARIABLES_MODELO,
        "tipo": [
            "categorica" if col in VARIABLES_CATEGORICAS else "numerica"
            for col in VARIABLES_MODELO
        ],
        "importancia": completo.get_feature_importance(
            type="PredictionValuesChange"
        ),
    })
    importancia = importancia.sort_values(
        ["importancia", "variable"], ascending=[False, True]
    ).reset_index(drop=True)
    importancia.insert(0, "posicion", importancia.index + 1)
    importancia.to_csv(
        RESULTADOS / "importancia_catboost_completo.csv",
        index=False, encoding="utf-8-sig",
    )

    resumen = [metrica_completo]
    variables_seleccionadas = {}
    for tamanio in TAMANIOS:
        columnas = importancia["variable"].head(tamanio).tolist()
        nombre = f"top_{tamanio}"
        variables_seleccionadas[nombre] = columnas
        _, metricas = entrenar_y_guardar(
            nombre, columnas, x_entrenamiento, entrenamiento,
            x_validacion, validacion, args.hilos,
        )
        resumen.append(metricas)

    pd.DataFrame(resumen).to_csv(
        RESULTADOS / "resumen_metricas.csv", index=False, encoding="utf-8-sig"
    )
    metadatos = {
        "version_catboost": version_catboost,
        "metodo_seleccion": "PredictionValuesChange del CatBoost completo ajustado solo con entrenamiento",
        "filas_dataset": len(dataset),
        "filas_entrenamiento": len(entrenamiento),
        "filas_validacion": len(validacion),
        "filas_test_reservado": len(test_reservado),
        "test_evaluado": False,
        "variables_completas": list(VARIABLES_MODELO),
        "variables_seleccionadas": variables_seleccionadas,
        "parametros_efectivos_modelo_completo": completo.get_all_params(),
    }
    (RESULTADOS / "metadatos_seleccion.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nMétricas de validación:")
    print(pd.DataFrame(resumen).to_string(index=False))
    print(f"Ranking, variables y modelos guardados en: {RESULTADOS}")
    print("Test reservado sin evaluar.")


if __name__ == "__main__":
    main()
