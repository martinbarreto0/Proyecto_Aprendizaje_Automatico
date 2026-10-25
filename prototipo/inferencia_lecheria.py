"""Pipeline de inferencia del prototipo de estimación de producción lechera.

Carga el modelo CatBoost de 12 variables ya entrenado, valida las entradas con
las mismas reglas usadas al construir el dataset y devuelve la estimación de
litros junto con una banda de confiabilidad. No reentrena nada.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA_PROTOTIPO = Path(__file__).resolve().parent
CARPETA_ARTEFACTOS = CARPETA_PROTOTIPO / "artefactos"
RUTA_MODELO = CARPETA_ARTEFACTOS / "modelo_catboost_top12.cbm"
RUTA_METADATOS = CARPETA_ARTEFACTOS / "metadatos_prototipo.json"
RUTA_DATASET = CARPETA_ARTEFACTOS / "dataset_prototipo.csv"
RUTA_METRICAS_PRUEBA = CARPETA_ARTEFACTOS / "metricas_prueba.csv"

# Existencias bovinas: cantidades de animales, no pueden ser negativas.
VARIABLES_CONTEO = [
    "total_bovinos_leche",
    "toros",
    "vacas_ordene",
    "vacas_secas",
    "vaquillonas_mas_2",
    "vaquillonas_1_2",
    "terneros_menos_1",
    "terneras_menos_1",
]

# Superficies en hectáreas y unidades ganaderas declaradas.
VARIABLES_SUPERFICIE = ["Superficie", "UnidadesGanaderas", "SuperficieGanadera"]

# Variables que el usuario debe aportar; carga_ganadera se calcula a partir de ellas.
VARIABLES_ENTRADA = VARIABLES_CONTEO + VARIABLES_SUPERFICIE

BANDAS = ("baja", "media", "alta")

# Códigos DICOSE de departamento (orden alfabético, mismo criterio que el ETL del PID).
DEPARTAMENTOS = {
    1: "Artigas", 2: "Canelones", 3: "Cerro Largo", 4: "Colonia", 5: "Durazno",
    6: "Flores", 7: "Florida", 8: "Lavalleja", 9: "Maldonado", 10: "Montevideo",
    11: "Paysandú", 12: "Río Negro", 13: "Rivera", 14: "Rocha", 15: "Salto",
    16: "San José", 17: "Soriano", 18: "Tacuarembó", 19: "Treinta y Tres",
}


class ErrorEntrada(ValueError):
    """Entrada que no cumple el esquema esperado por el modelo."""


def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el modelo CatBoost guardado; falla con un mensaje claro si no existe."""
    from catboost import CatBoostRegressor

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. "
            "Ejecutar primero PAA/prototipo/preparar_datos_prototipo.py."
        )
    modelo = CatBoostRegressor()
    modelo.load_model(str(ruta))
    return modelo


def cargar_metadatos(ruta=RUTA_METADATOS):
    """Lee los metadatos del prototipo (umbrales, métricas y particiones incluidas)."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontraron los metadatos en {ruta}. "
            "Ejecutar primero PAA/prototipo/preparar_datos_prototipo.py."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_dataset(ruta=RUTA_DATASET):
    """Lee el dataset analítico preparado para el prototipo."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {ruta}. "
            "Ejecutar primero PAA/prototipo/preparar_datos_prototipo.py."
        )
    return pd.read_csv(ruta, encoding="utf-8-sig", dtype={"GiroCodigo": str})


def cargar_metricas_prueba(ruta=RUTA_METRICAS_PRUEBA):
    """Métricas de la única evaluación sobre la prueba, o None si no se copiaron."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    fila = pd.read_csv(ruta, encoding="utf-8-sig").iloc[0]
    return {
        "filas": int(fila["filas"]),
        "mae": float(fila["mae_litros"]),
        "rmse": float(fila["rmse_litros"]),
        "r2": float(fila["r2"]),
        "wape": float(fila["wape_porcentaje"]),
    }


def calcular_carga_ganadera(datos):
    """Unidades ganaderas por hectárea ganadera, solo cuando la superficie es positiva."""
    superficie = datos["SuperficieGanadera"].where(datos["SuperficieGanadera"] > 0)
    return datos["UnidadesGanaderas"] / superficie


def validar_entrada(datos, variables_modelo):
    """Comprueba el esquema y devuelve una copia numérica lista para el modelo.

    Reglas: al menos una fila, todas las variables de entrada presentes, valores
    numéricos finitos, sin negativos y sin filas completamente vacías. Los
    faltantes puntuales se admiten como NaN, igual que en el entrenamiento.
    """
    if not isinstance(datos, pd.DataFrame):
        raise ErrorEntrada("La entrada debe ser una tabla (DataFrame) de pandas.")
    if datos.empty:
        raise ErrorEntrada("La entrada no contiene filas.")

    requeridas = [v for v in variables_modelo if v != "carga_ganadera"]
    faltantes = [v for v in requeridas if v not in datos.columns]
    if faltantes:
        raise ErrorEntrada(f"Faltan columnas requeridas: {', '.join(faltantes)}.")

    limpio = pd.DataFrame(index=datos.index)
    for columna in requeridas:
        valores = pd.to_numeric(datos[columna], errors="coerce")
        invalidos = valores.isna() & datos[columna].notna()
        if invalidos.any():
            raise ErrorEntrada(f"La columna {columna} contiene valores no numéricos.")
        if np.isinf(valores).any():
            raise ErrorEntrada(f"La columna {columna} contiene valores infinitos.")
        if (valores < 0).any():
            raise ErrorEntrada(f"La columna {columna} contiene valores negativos.")
        limpio[columna] = valores.astype("float64")

    if limpio[requeridas].isna().all(axis=1).any():
        raise ErrorEntrada("Hay filas sin ningún valor informado.")

    if "carga_ganadera" in variables_modelo:
        limpio["carga_ganadera"] = calcular_carga_ganadera(limpio)
    return limpio[list(variables_modelo)]


def banda_confiabilidad(litros, cortes):
    """Clasifica la estimación según la escala, porque el error relativo depende de ella."""
    return np.where(litros < cortes[0], "baja", np.where(litros < cortes[1], "media", "alta"))


def predecir(modelo, datos, metadatos):
    """Valida, predice y agrega la banda de confiabilidad de cada estimación."""
    variables = list(modelo.feature_names_)
    entrada = validar_entrada(datos, variables)
    crudas = np.asarray(modelo.predict(entrada), dtype="float64")
    salida = pd.DataFrame(index=datos.index)
    # La producción no puede ser negativa; se recorta y se deja constancia.
    salida["litros_estimados"] = np.clip(crudas, 0, None)
    salida["prediccion_recortada"] = crudas < 0
    salida["banda_confiabilidad"] = banda_confiabilidad(
        salida["litros_estimados"].to_numpy(), metadatos["cortes_banda_litros"]
    )
    return salida


def evaluar_desvio(declarados, estimados, banda, metadatos):
    """Compara lo declarado con lo estimado y marca el perfil como atípico si
    el desvío relativo supera el percentil 90 observado en validación para su banda."""
    if estimados <= 0:
        return {"desvio_relativo": np.nan, "umbral": np.nan, "atipico": False}
    desvio = (declarados - estimados) / estimados
    umbral = metadatos["umbral_desvio_p90"][banda]
    return {"desvio_relativo": desvio, "umbral": umbral, "atipico": abs(desvio) > umbral}
