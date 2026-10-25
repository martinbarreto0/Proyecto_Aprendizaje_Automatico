"""Prepara los artefactos que usa el prototipo sin volver a entrenar el modelo.

1. Reconstruye el dataset analítico desde los CSV de PAA/data con datos_lecheria.py.
2. Guarda solo las particiones de entrenamiento y validación; la prueba reservada
   queda excluida hasta la entrega final (usar --incluir-prueba para agregarla).
3. Copia el modelo CatBoost de 12 variables ya entrenado.
4. Calcula, solo con validación, los umbrales que el prototipo usa para marcar
   perfiles atípicos y las bandas de confiabilidad.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA_PROTOTIPO = Path(__file__).resolve().parent
CARPETA_PAA = CARPETA_PROTOTIPO.parent
sys.path.insert(0, str(CARPETA_PAA / "scripts" / "modelos_produccion_lechera" / "basicos"))
sys.path.insert(0, str(CARPETA_PROTOTIPO))

from datos_lecheria import (  # noqa: E402
    CLAVES_UNION,
    OBJETIVO,
    VARIABLES_MODELO,
    construir_dataset,
    dividir_dataset,
)
import inferencia_lecheria as inf  # noqa: E402

CARPETA_MODELO = CARPETA_PAA / "reports" / "modelos_catboost" / "seleccion_variables"
MODELO_ORIGEN = CARPETA_MODELO / "modelo_top_12.cbm"
PREDICCIONES_ORIGEN = CARPETA_MODELO / "predicciones_top_12.csv"
# Resultado de la única evaluación sobre la prueba (22/10/2026); el dashboard lo muestra.
METRICAS_PRUEBA_ORIGEN = CARPETA_PAA / "reports" / "modelos_catboost" / "evaluacion_final_test" / "metricas_test.csv"

# Límites de los dos primeros quintiles de litros observados en validación
# (apartado 4.4 del informe); por debajo el error relativo supera el 50 %.
CORTES_BANDA = [149329.0, 349928.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incluir-prueba", action="store_true",
        help="Agrega la partición de prueba (solo para la entrega final).",
    )
    args = parser.parse_args()

    dataset = construir_dataset()
    entrenamiento, validacion, prueba = dividir_dataset(dataset)
    partes = [entrenamiento.assign(particion="entrenamiento"),
              validacion.assign(particion="validacion")]
    if args.incluir_prueba:
        partes.append(prueba.assign(particion="prueba"))
    columnas = CLAVES_UNION + list(VARIABLES_MODELO) + [OBJETIVO, "particion"]
    salida = pd.concat(partes)[columnas].reset_index(drop=True)

    inf.CARPETA_ARTEFACTOS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODELO_ORIGEN, inf.RUTA_MODELO)
    if METRICAS_PRUEBA_ORIGEN.exists():
        shutil.copy2(METRICAS_PRUEBA_ORIGEN, inf.RUTA_METRICAS_PRUEBA)
    salida.to_csv(inf.RUTA_DATASET, index=False, encoding="utf-8-sig")

    # Umbrales calculados solo con validación, con el mismo pipeline de inferencia.
    modelo = inf.cargar_modelo()
    provisorios = {"cortes_banda_litros": CORTES_BANDA}
    pred = inf.predecir(modelo, validacion, provisorios)
    real = validacion[OBJETIVO].to_numpy()
    est = pred["litros_estimados"].to_numpy()
    positivas = est > 0
    desvio = np.full_like(est, np.nan)
    desvio[positivas] = np.abs(real[positivas] - est[positivas]) / est[positivas]

    umbrales, wape_banda, filas_banda = {}, {}, {}
    for banda in inf.BANDAS:
        m = (pred["banda_confiabilidad"] == banda).to_numpy()
        umbrales[banda] = float(np.nanpercentile(desvio[m], 90))
        wape_banda[banda] = float(np.abs(real[m] - est[m]).sum() / real[m].sum() * 100)
        filas_banda[banda] = int(m.sum())

    # Control de reproducibilidad: las predicciones sin recortar deben coincidir,
    # fila a fila, con las guardadas al entrenar el modelo.
    originales = pd.read_csv(PREDICCIONES_ORIGEN, encoding="utf-8-sig")["Litros_predichos"].to_numpy()
    crudas = modelo.predict(inf.validar_entrada(validacion, modelo.feature_names_))
    max_dif = float(np.max(np.abs(originales - crudas)))

    metadatos = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modelo": "CatBoost con 12 variables seleccionadas (CatBoost top 12)",
        "modelo_origen": str(MODELO_ORIGEN.relative_to(CARPETA_PAA)).replace("\\", "/"),
        "variables_modelo": list(modelo.feature_names_),
        "prueba_incluida": bool(args.incluir_prueba),
        "filas_por_particion": salida["particion"].value_counts().to_dict(),
        "cortes_banda_litros": CORTES_BANDA,
        "umbral_desvio_p90": umbrales,
        "wape_validacion_por_banda": wape_banda,
        "filas_validacion_por_banda": filas_banda,
        "predicciones_negativas_validacion": int(pred["prediccion_recortada"].sum()),
        "mae_validacion_original": float(np.abs(real - crudas).mean()),
        "mae_validacion": float(np.abs(real - est).mean()),
        "wape_validacion": float(np.abs(real - est).sum() / real.sum() * 100),
        "max_diferencia_con_predicciones_originales": max_dif,
    }
    inf.RUTA_METADATOS.write_text(json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prototipo] filas guardadas: {metadatos['filas_por_particion']}")
    print(f"[prototipo] MAE validación: {metadatos['mae_validacion']:.2f}  WAPE: {metadatos['wape_validacion']:.4f} %")
    print(f"[prototipo] umbrales P90 por banda: {umbrales}")
    print(f"[prototipo] diferencia máxima con predicciones originales: {max_dif:.6f} litros")


if __name__ == "__main__":
    main()
