"""Evalúa una sola vez el CatBoost básico top 12 sobre el test reservado.

Sin --ejecutar-test solo reconstruye el dataset, verifica los insumos y reproduce
las métricas de validación. Con --ejecutar-test predice el test una vez y guarda
los resultados en una carpeta nueva. No vuelve a entrenar ni elige variables.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA_SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA_SCRIPT.parent / "basicos"))

from modelo_catboost_basico import preparar_variables  # noqa: E402
from datos_lecheria import (  # noqa: E402
    CARPETA_PAA,
    CLAVES_UNION,
    OBJETIVO,
    calcular_metricas,
    construir_dataset,
    dividir_dataset,
)


CARPETA_DATOS = CARPETA_PAA / "data"
CARPETA_SELECCION = CARPETA_PAA / "reports" / "modelos_catboost" / "seleccion_variables"
RUTA_MODELO = CARPETA_SELECCION / "modelo_top_12.cbm"
RUTA_VARIABLES = CARPETA_SELECCION / "variables_top_12.csv"
RUTA_METRICAS_VALIDACION = CARPETA_SELECCION / "metricas_top_12.csv"
RESULTADOS = CARPETA_PAA / "reports" / "modelos_catboost" / "evaluacion_final_test"

# Huellas SHA-256 fijadas en SELECCION_MODELO_FINAL_CATBOOST_TOP12_Y_TEST.md.
HUELLAS = {
    RUTA_MODELO: "5396CFF3131864D90119D4487B222FD039FF15E599B355DCF2B19611CC0D5B92",
    RUTA_VARIABLES: "30524109D0E50F3426EE6EE91E1E66DED8481C70C1171EBCC02E949D377827C7",
    CARPETA_DATOS / "DatosProduccionLeche.csv": "CB11E624A60E19673CF0310BCC487E96041603C5D25B4CBA02AF021CD6B0ACF9",
    CARPETA_DATOS / "DatosAnimales.csv": "ECF15FFE24B9B5A925DA78D56BBFA1811CCD85BA4527B95774DC05060B8F1EC7",
    CARPETA_DATOS / "DatosAnimalesDetallados.csv": "C6E36C5C45B4104F00C74507287A6F8F18B13A8E42483E7BA13CEBB9028605A3",
    CARPETA_DATOS / "DatosGenerales.csv": "6958A73A5D8F27804A9B35F05793597795D61CEB100B0063A4313D3C3F42DA7B",
}

FILAS_ESPERADAS = {"dataset": 16204, "entrenamiento": 11342, "validacion": 2431, "test": 2431}


def sha256(ruta):
    """Calcula la huella SHA-256 de un archivo en bloques para no cargarlo entero."""
    digest = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest().upper()


def verificar_insumos():
    """Detiene el proceso si algún insumo no coincide con la huella fijada."""
    obtenidas = {}
    for ruta, esperada in HUELLAS.items():
        obtenidas[ruta.name] = sha256(ruta)
        if obtenidas[ruta.name] != esperada:
            raise SystemExit(f"Huella distinta en {ruta.name}: {obtenidas[ruta.name]} != {esperada}")
        print(f"[huella ok] {ruta.name}")
    return obtenidas


def cargar_modelo(variables):
    """Carga el artefacto existente sin llamar a fit() y comprueba sus entradas."""
    from catboost import CatBoostRegressor

    modelo = CatBoostRegressor()
    modelo.load_model(str(RUTA_MODELO))
    if list(modelo.feature_names_) != variables:
        raise SystemExit(
            f"Las variables del modelo no coinciden con variables_top_12.csv: "
            f"{modelo.feature_names_} != {variables}"
        )
    return modelo


def predecir(modelo, datos, variables):
    """Prepara las mismas variables de entrenamiento y devuelve predicciones."""
    x = preparar_variables(datos)[variables]
    return modelo.predict(x)


def control_validacion(modelo, validacion, variables):
    """Reproduce las métricas de validación registradas antes de abrir test."""
    predicciones = predecir(modelo, validacion, variables)
    metricas = calcular_metricas("CatBoost_top_12", validacion[OBJETIVO], predicciones)
    referencia = pd.read_csv(RUTA_METRICAS_VALIDACION, encoding="utf-8-sig").iloc[0]
    for medida in ("mae_litros", "rmse_litros", "r2", "wape_porcentaje"):
        if not np.isclose(metricas[medida], referencia[medida], rtol=1e-6):
            raise SystemExit(
                f"El control sobre validación no coincide en {medida}: "
                f"{metricas[medida]} != {referencia[medida]}. Se detiene la evaluación."
            )
    print("[control validación] MAE, RMSE, R² y WAPE coinciden con metricas_top_12.csv")
    return metricas


def evaluar_test(modelo, test, variables, huellas, versiones):
    """Predice test una sola vez y guarda métricas, predicciones y metadatos."""
    if (RESULTADOS / "metricas_test.csv").exists():
        raise SystemExit("La evaluación de test ya fue ejecutada; no se repite.")

    predicciones = predecir(modelo, test, variables)
    metricas = calcular_metricas("CatBoost_top_12", test[OBJETIVO], predicciones)
    metricas["particion"] = "test"

    tabla = test[CLAVES_UNION].copy()
    tabla["Litros_reales"] = test[OBJETIVO].to_numpy()
    tabla["Litros_predichos"] = predicciones
    tabla["Error"] = tabla["Litros_predichos"] - tabla["Litros_reales"]

    RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metricas]).to_csv(
        RESULTADOS / "metricas_test.csv", index=False, encoding="utf-8-sig"
    )
    tabla.to_csv(RESULTADOS / "predicciones_test.csv", index=False, encoding="utf-8-sig")

    metadatos = {
        "modelo": "CatBoost_top_12",
        "fecha_evaluacion": datetime.now().isoformat(timespec="seconds"),
        "versiones": versiones,
        "huellas_sha256": huellas,
        "filas_test": len(test),
        "particion": "test",
        "incidencias": [],
    }
    (RESULTADOS / "metadatos_test.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metricas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ejecutar-test", action="store_true",
        help="Predice el test reservado una única vez. Sin esta bandera solo hace el control.",
    )
    args = parser.parse_args()

    from catboost import __version__ as version_catboost
    import sklearn

    huellas = verificar_insumos()
    versiones = {
        "catboost": version_catboost,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    print("Versiones:", versiones)

    dataset = construir_dataset()
    entrenamiento, validacion, test = dividir_dataset(dataset)
    tamanios = {
        "dataset": len(dataset),
        "entrenamiento": len(entrenamiento),
        "validacion": len(validacion),
        "test": len(test),
    }
    if tamanios != FILAS_ESPERADAS:
        raise SystemExit(f"Las filas no coinciden con lo esperado: {tamanios}")

    variables = pd.read_csv(RUTA_VARIABLES, encoding="utf-8-sig")["variable"].tolist()
    modelo = cargar_modelo(variables)
    control_validacion(modelo, validacion, variables)

    if not args.ejecutar_test:
        print("Test no evaluado: falta ejecutar con --ejecutar-test.")
        return

    metricas = evaluar_test(modelo, test, variables, huellas, versiones)
    print("\nMétricas de test:")
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor}")


if __name__ == "__main__":
    main()
