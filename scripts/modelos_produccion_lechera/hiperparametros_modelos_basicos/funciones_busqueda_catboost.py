"""Grid exhaustivo de CatBoost, con bloques reanudables y test reservado."""

import json
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, __version__ as VERSION_CATBOOST
from sklearn import __version__ as VERSION_SKLEARN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent / "basicos"))
from datos_lecheria import (  # noqa: E402
    CARPETA_PAA, CLAVES_UNION, OBJETIVO, SEMILLA_PRINCIPAL,
    VARIABLES_CATEGORICAS, VARIABLES_MODELO, construir_dataset, dividir_dataset,
)
from modelo_catboost_basico import preparar_variables  # noqa: E402

RESULTADOS = CARPETA_PAA / "reports" / "modelos_catboost" / "hiperparametros"
SELECCION = CARPETA_PAA / "reports" / "modelos_catboost" / "seleccion_variables"
GRID = {
    "iterations": [500, 1000, 1500],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.08, 0.15],
    "l2_leaf_reg": [3, 10],
    "random_strength": [0, 1],
}
PLIEGUES = 5
BLOQUES_POR_VARIANTE = {"completo": 6, "top_12": 3}


def parametros_del_bloque(variante, numero):
    """Divide el grid sin perder ni repetir configuraciones."""
    total_bloques = BLOQUES_POR_VARIANTE.get(variante)
    if total_bloques is None:
        raise ValueError(f"Variante desconocida: {variante}")
    if numero not in range(1, total_bloques + 1):
        raise ValueError(f"El bloque de {variante} debe estar entre 1 y {total_bloques}")
    todos = list(ParameterGrid(GRID))
    assert len(todos) == 108
    if variante == "top_12":
        return [
            p for p in todos
            if p["iterations"] == GRID["iterations"][numero - 1]
        ]

    # En el completo, cada cantidad de árboles ocupa dos bloques de 18.
    indice_iterations = (numero - 1) // 2
    mitad = (numero - 1) % 2
    grupo_iterations = [
        p for p in todos
        if p["iterations"] == GRID["iterations"][indice_iterations]
    ]
    return grupo_iterations[mitad::2]


def id_configuracion(parametros):
    texto = json.dumps(parametros, sort_keys=True, separators=(",", ":"))
    return sha256(texto.encode("utf-8")).hexdigest()[:16]


def guardar_csv(tabla, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    tabla.to_csv(temporal, index=False, encoding="utf-8-sig")
    temporal.replace(ruta)


def guardar_json(datos, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    temporal.replace(ruta)


def huella_datos(tabla):
    """Detecta cambios en valores, orden, columnas o tipos entre bloques."""
    digest = sha256()
    digest.update(pd.util.hash_pandas_object(tabla, index=True).values.tobytes())
    digest.update(json.dumps(list(tabla.columns)).encode("utf-8"))
    digest.update(json.dumps([str(t) for t in tabla.dtypes]).encode("utf-8"))
    return digest.hexdigest()


def cargar_datos(variante):
    dataset = construir_dataset()
    train, validacion, test_reservado = dividir_dataset(dataset)
    if variante == "completo":
        variables = list(VARIABLES_MODELO)
    elif variante == "top_12":
        ruta = SELECCION / "variables_top_12.csv"
        variables = pd.read_csv(ruta)["variable"].tolist()
        ranking = pd.read_csv(SELECCION / "importancia_catboost_completo.csv")
        if variables != ranking["variable"].head(12).tolist():
            raise RuntimeError("Las 12 variables no coinciden con el ranking guardado")
    else:
        raise ValueError(f"Variante desconocida: {variante}")
    if len(variables) != (16 if variante == "completo" else 12):
        raise RuntimeError("Cantidad inesperada de variables")
    if len(set(variables)) != len(variables) or not set(variables) <= set(VARIABLES_MODELO):
        raise RuntimeError("Lista de variables inválida")
    x_train = preparar_variables(train)[variables]
    x_validacion = preparar_variables(validacion)[variables]
    categorias = [c for c in VARIABLES_CATEGORICAS if c in variables]
    return train, validacion, test_reservado, x_train, x_validacion, variables, categorias


def metadatos_actuales(variante, train, validacion, test, x_train, x_validacion,
                      variables):
    """Un bloque posterior debe encontrar exactamente el mismo experimento."""
    fuentes = [
        CARPETA.parent / "basicos" / "datos_lecheria.py",
        CARPETA.parent / "basicos" / "modelo_catboost_basico.py",
        Path(__file__),
    ]
    return {
        "variante": variante,
        "grid": GRID,
        "pliegues": PLIEGUES,
        "total_bloques": BLOQUES_POR_VARIANTE[variante],
        "configuraciones_por_bloque": len(parametros_del_bloque(variante, 1)),
        "semilla": SEMILLA_PRINCIPAL,
        "version_catboost": VERSION_CATBOOST,
        "version_sklearn": VERSION_SKLEARN,
        "variables": variables,
        "filas_train": len(train),
        "filas_validacion": len(validacion),
        "filas_test_reservado": len(test),
        "huella_train": huella_datos(pd.concat([x_train, train[[OBJETIVO]]], axis=1)),
        "huella_validacion": huella_datos(pd.concat([x_validacion, validacion[[OBJETIVO]]], axis=1)),
        "huella_test": huella_datos(test[CLAVES_UNION + [OBJETIVO]]),
        "huellas_codigo": {p.name: sha256(p.read_bytes()).hexdigest() for p in fuentes},
        "test_evaluado": False,
    }


def verificar_plan(carpeta, actual):
    ruta = carpeta / "plan.json"
    if ruta.exists():
        anterior = json.loads(ruta.read_text(encoding="utf-8"))
        if anterior != actual:
            raise RuntimeError("Cambió el grid, los datos, las variables, las versiones o el código. No se mezclarán resultados.")
    else:
        if any(carpeta.glob("bloque_*.csv")):
            raise RuntimeError("Hay bloques sin plan.json; no se pueden validar")
        guardar_json(actual, ruta)


def crear_modelo(parametros, hilos):
    return CatBoostRegressor(
        **parametros, loss_function="RMSE", random_seed=SEMILLA_PRINCIPAL,
        thread_count=hilos, verbose=False, allow_writing_files=False,
    )


def metricas(y, pred):
    reales = np.asarray(y, dtype=float)
    predichos = np.asarray(pred, dtype=float)
    return {
        "mae": mean_absolute_error(reales, predichos),
        "rmse": mean_squared_error(reales, predichos) ** 0.5,
        "r2": r2_score(reales, predichos),
        "wape": 100 * np.abs(reales - predichos).sum() / np.abs(reales).sum(),
    }


def validar_bloque(tabla, variante, numero):
    esperados = {
        id_configuracion(p): p for p in parametros_del_bloque(variante, numero)
    }
    if not set(tabla["id_configuracion"]) <= set(esperados):
        raise RuntimeError(f"El bloque {numero} contiene configuraciones ajenas")
    if tabla["id_configuracion"].duplicated().any():
        raise RuntimeError(f"El bloque {numero} contiene duplicados")
    for _, fila in tabla.iterrows():
        parametros = esperados[fila["id_configuracion"]]
        if int(fila["bloque"]) != numero or any(
            fila[nombre] != valor for nombre, valor in parametros.items()
        ):
            raise RuntimeError(f"Los parámetros del bloque {numero} no coinciden con el grid")
    columnas = [f"{m}_pliegue_{k}" for m in ("mae", "rmse", "r2", "wape")
               for k in range(1, PLIEGUES + 1)]
    columnas += [f"media_{m}" for m in ("mae", "rmse", "r2", "wape")]
    if not np.isfinite(tabla[columnas].to_numpy(dtype=float)).all():
        raise RuntimeError(f"El bloque {numero} tiene métricas faltantes o inválidas")


def ejecutar_bloque(variante, numero, hilos):
    if hilos == 0 or hilos < -1:
        raise ValueError("--hilos debe ser -1 o un entero positivo")
    train, validacion, test, x_train, x_validacion, variables, categorias = cargar_datos(variante)
    carpeta = RESULTADOS / variante
    plan = metadatos_actuales(variante, train, validacion, test, x_train, x_validacion, variables)
    verificar_plan(carpeta, plan)
    ruta = carpeta / f"bloque_{numero}.csv"
    tabla = pd.read_csv(ruta, dtype={"id_configuracion": str}) if ruta.exists() else pd.DataFrame()
    if not tabla.empty:
        validar_bloque(tabla, variante, numero)
    hechos = set(tabla["id_configuracion"]) if not tabla.empty else set()
    configuraciones = parametros_del_bloque(variante, numero)
    total_configuraciones = len(configuraciones)
    splits = list(KFold(n_splits=PLIEGUES, shuffle=False).split(x_train))
    print(
        f"[{variante}] bloque {numero}: "
        f"{len(hechos)}/{total_configuraciones} configuraciones completas",
        flush=True,
    )

    for posicion, parametros in enumerate(configuraciones, start=1):
        ident = id_configuracion(parametros)
        if ident in hechos:
            continue
        inicio = perf_counter()
        fila = {"id_configuracion": ident, "bloque": numero, **parametros}
        for k, (indices_train, indices_cv) in enumerate(splits, start=1):
            modelo = crear_modelo(parametros, hilos)
            modelo.fit(x_train.iloc[indices_train], train[OBJETIVO].iloc[indices_train],
                       cat_features=categorias)
            resultado = metricas(train[OBJETIVO].iloc[indices_cv],
                                 modelo.predict(x_train.iloc[indices_cv]))
            for nombre, valor in resultado.items():
                fila[f"{nombre}_pliegue_{k}"] = valor
        for nombre in ("mae", "rmse", "r2", "wape"):
            valores = [fila[f"{nombre}_pliegue_{k}"] for k in range(1, PLIEGUES + 1)]
            fila[f"media_{nombre}"] = float(np.mean(valores))
            fila[f"desvio_{nombre}"] = float(np.std(valores))
        fila["segundos"] = perf_counter() - inicio
        tabla = pd.concat([tabla, pd.DataFrame([fila])], ignore_index=True)
        validar_bloque(tabla, variante, numero)
        guardar_csv(tabla, ruta)
        print(f"[{variante}] bloque {numero}, "
              f"configuración {posicion}/{total_configuraciones}: "
              f"MAE CV={fila['media_mae']:,.0f}; {fila['segundos']:.1f} s", flush=True)
    print(f"Bloque completo: {ruta}")


def consolidar(variante, hilos):
    if hilos == 0 or hilos < -1:
        raise ValueError("--hilos debe ser -1 o un entero positivo")
    train, validacion, test, x_train, x_validacion, variables, categorias = cargar_datos(variante)
    carpeta = RESULTADOS / variante
    plan = metadatos_actuales(variante, train, validacion, test, x_train, x_validacion, variables)
    if not (carpeta / "plan.json").exists():
        raise RuntimeError("Falta plan.json; primero hay que ejecutar los bloques")
    verificar_plan(carpeta, plan)
    partes = []
    for numero in range(1, BLOQUES_POR_VARIANTE[variante] + 1):
        ruta = carpeta / f"bloque_{numero}.csv"
        if not ruta.exists():
            raise RuntimeError(f"Falta el bloque {numero}")
        tabla = pd.read_csv(ruta, dtype={"id_configuracion": str})
        validar_bloque(tabla, variante, numero)
        esperadas_en_bloque = len(parametros_del_bloque(variante, numero))
        if len(tabla) != esperadas_en_bloque:
            raise RuntimeError(
                f"El bloque {numero} está incompleto: "
                f"{len(tabla)}/{esperadas_en_bloque}"
            )
        partes.append(tabla)
    resultados = pd.concat(partes, ignore_index=True)
    esperados = {id_configuracion(p) for p in ParameterGrid(GRID)}
    if len(resultados) != 108 or set(resultados["id_configuracion"]) != esperados:
        raise RuntimeError("Los bloques no cubren las 108 configuraciones exactas")
    resultados = resultados.sort_values(["media_mae", "id_configuracion"]).reset_index(drop=True)
    guardar_csv(resultados, carpeta / "resultados_cv.csv")
    mejor = resultados.iloc[0]
    parametros = {nombre: mejor[nombre].item() if isinstance(mejor[nombre], np.generic)
                  else mejor[nombre] for nombre in GRID}
    modelo = crear_modelo(parametros, hilos)
    modelo.fit(x_train, train[OBJETIVO], cat_features=categorias)
    pred = modelo.predict(x_validacion)
    evaluacion = metricas(validacion[OBJETIVO], pred)
    guardar_csv(pd.DataFrame([{"modelo": f"CatBoost_{variante}_ajustado",
                              "particion": "validacion", "filas": len(validacion),
                              "mae_litros": evaluacion["mae"],
                              "rmse_litros": evaluacion["rmse"], "r2": evaluacion["r2"],
                              "wape_porcentaje": evaluacion["wape"]}]),
                carpeta / "metricas_validacion.csv")
    salida = validacion[CLAVES_UNION].copy()
    salida["Litros_reales"] = validacion[OBJETIVO].to_numpy()
    salida["Litros_predichos"] = pred
    salida["Error"] = salida["Litros_predichos"] - salida["Litros_reales"]
    guardar_csv(salida, carpeta / "predicciones_validacion.csv")
    guardar_json({"variante": variante, "variables": variables,
                 "criterio": "menor MAE medio en 5 pliegues de entrenamiento",
                 "mejores_parametros": parametros,
                 "mae_medio_cv": float(mejor["media_mae"]),
                 "id_configuracion": mejor["id_configuracion"],
                 "configuraciones": len(resultados), "test_evaluado": False,
                 "filas_test_reservado": len(test)}, carpeta / "mejor_modelo.json")
    temporal = carpeta / "modelo_ganador.cbm.tmp"
    modelo.save_model(str(temporal))
    temporal.replace(carpeta / "modelo_ganador.cbm")
    print(f"Ganador {variante}: {parametros}")
    print(f"Validación: {evaluacion}")
    print(f"Resultados: {carpeta}; test reservado sin evaluar")
