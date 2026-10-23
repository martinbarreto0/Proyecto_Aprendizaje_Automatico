"""Utilidades reproducibles para ejecutar Random Forest en bloques."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.pipeline import Pipeline


CARPETA_SCRIPTS = Path(__file__).resolve().parent
CARPETA_PADRE = CARPETA_SCRIPTS.parent
sys.path.insert(0, str(CARPETA_PADRE))

from funciones_busqueda import (  # noqa: E402
    OBJETIVO,
    RESULTADOS,
    SCORING,
    SEMILLA_PRINCIPAL,
    VARIABLES_MODELO,
    espacio_random_forest,
    preparar_datos,
    preprocesamiento_random_forest,
)


TOTAL_BLOQUES = 8
TOTAL_CONFIGURACIONES = 384
PLIEGUES = 5
CONFIGURACIONES_POR_BLOQUE = TOTAL_CONFIGURACIONES // TOTAL_BLOQUES
SEMILLA_DISTRIBUCION = 42

RESULTADOS_BLOQUES = RESULTADOS / "random_forest_por_bloques"
MANIFIESTO = RESULTADOS_BLOQUES / "manifiesto_configuraciones.csv"
ASIGNACION_PLIEGUES = RESULTADOS_BLOQUES / "asignacion_pliegues.csv"
METADATOS = RESULTADOS_BLOQUES / "metadatos_busqueda.json"

ARCHIVO_FUNCIONES_ORIGINAL = CARPETA_PADRE / "funciones_busqueda.py"
ARCHIVO_BUSQUEDA_ORIGINAL = CARPETA_PADRE / "buscar_hiperparametros_random_forest.py"


def fecha_utc():
    """Devuelve una fecha ISO reproducible para los metadatos."""
    return datetime.now(timezone.utc).isoformat()


def normalizar_valor(valor):
    """Convierte escalares de NumPy a tipos serializables por JSON."""
    return valor.item() if isinstance(valor, np.generic) else valor


def normalizar_parametros(parametros):
    """Ordena y normaliza un diccionario de hiperparámetros."""
    return {
        nombre: normalizar_valor(parametros[nombre])
        for nombre in sorted(parametros)
    }


def parametros_a_json(parametros):
    """Crea la representación canónica de una configuración."""
    return json.dumps(
        normalizar_parametros(parametros),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def id_configuracion(parametros):
    """Crea un identificador estable a partir de los parámetros."""
    return sha256(parametros_a_json(parametros).encode("utf-8")).hexdigest()[:16]


def hash_archivo(ruta):
    """Calcula SHA-256 de un archivo."""
    digest = sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def huella_dataframe(datos):
    """Resume contenido, orden, columnas y tipos de un conjunto de datos."""
    columnas = VARIABLES_MODELO + [OBJETIVO]
    vista = datos[columnas]
    digest = sha256()
    digest.update(pd.util.hash_pandas_object(vista, index=True).values.tobytes())
    digest.update("|".join(columnas).encode("utf-8"))
    digest.update("|".join(map(str, vista.dtypes)).encode("utf-8"))
    return digest.hexdigest()


def configuraciones_maestras():
    """Expande el grid original y valida sus 384 configuraciones."""
    configuraciones = [
        normalizar_parametros(configuracion)
        for configuracion in ParameterGrid(espacio_random_forest())
    ]
    identificadores = [id_configuracion(configuracion) for configuracion in configuraciones]
    if len(configuraciones) != TOTAL_CONFIGURACIONES:
        raise RuntimeError(
            f"Se esperaban {TOTAL_CONFIGURACIONES} configuraciones y se obtuvieron "
            f"{len(configuraciones)}."
        )
    if len(set(identificadores)) != TOTAL_CONFIGURACIONES:
        raise RuntimeError("El grid contiene configuraciones duplicadas.")
    return configuraciones


def crear_manifiesto():
    """Distribuye el grid en ocho bloques equilibrados y sin solapamientos."""
    configuraciones = configuraciones_maestras()
    estratos = {}
    for configuracion in configuraciones:
        clave = (
            configuracion["modelo__n_estimators"],
            configuracion["modelo__bootstrap"],
        )
        estratos.setdefault(clave, []).append(configuracion)

    bloques = {numero: [] for numero in range(1, TOTAL_BLOQUES + 1)}
    for (arboles, bootstrap), grupo in sorted(
        estratos.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        grupo = sorted(grupo, key=parametros_a_json)
        semilla = SEMILLA_DISTRIBUCION + arboles + (10_000 if bootstrap else 0)
        random.Random(semilla).shuffle(grupo)
        for posicion, configuracion in enumerate(grupo):
            bloque = posicion % TOTAL_BLOQUES + 1
            bloques[bloque].append(configuracion)

    registros = []
    for bloque, configuraciones_bloque in bloques.items():
        configuraciones_bloque = sorted(
            configuraciones_bloque, key=id_configuracion
        )
        if len(configuraciones_bloque) != CONFIGURACIONES_POR_BLOQUE:
            raise RuntimeError(
                f"El bloque {bloque} tiene {len(configuraciones_bloque)} "
                f"configuraciones; se esperaban {CONFIGURACIONES_POR_BLOQUE}."
            )
        for orden, configuracion in enumerate(configuraciones_bloque, start=1):
            registro = {
                "id_configuracion": id_configuracion(configuracion),
                "bloque": bloque,
                "orden_en_bloque": orden,
                "parametros_json": parametros_a_json(configuracion),
            }
            registro.update(
                {
                    nombre.replace("modelo__", ""): valor
                    for nombre, valor in configuracion.items()
                }
            )
            registros.append(registro)

    manifiesto = pd.DataFrame(registros).sort_values(
        ["bloque", "orden_en_bloque"]
    )
    if manifiesto["id_configuracion"].nunique() != TOTAL_CONFIGURACIONES:
        raise RuntimeError("La distribución generó duplicados o perdió configuraciones.")

    esperadas = {id_configuracion(configuracion) for configuracion in configuraciones}
    distribuidas = set(manifiesto["id_configuracion"])
    if distribuidas != esperadas:
        raise RuntimeError("La unión de los bloques no coincide con el grid original.")

    for bloque, grupo in manifiesto.groupby("bloque"):
        conteo_arboles = grupo["n_estimators"].value_counts().to_dict()
        conteo_bootstrap = grupo["bootstrap"].value_counts().to_dict()
        if conteo_arboles != {300: 24, 700: 24}:
            raise RuntimeError(f"El bloque {bloque} no está equilibrado por árboles.")
        if conteo_bootstrap != {True: 32, False: 16}:
            raise RuntimeError(f"El bloque {bloque} no está equilibrado por bootstrap.")
    return manifiesto.reset_index(drop=True)


def hash_grid(manifiesto):
    """Calcula una huella estable de todas las configuraciones."""
    valores = manifiesto.sort_values("id_configuracion")["parametros_json"]
    return sha256("\n".join(valores).encode("utf-8")).hexdigest()


def crear_asignacion_pliegues(cantidad_filas):
    """Fija los mismos cinco pliegues para todos los bloques."""
    asignacion = np.full(cantidad_filas, -1, dtype=int)
    separador = KFold(n_splits=PLIEGUES, shuffle=False)
    for pliegue, (_, indices_validacion) in enumerate(
        separador.split(np.arange(cantidad_filas))
    ):
        asignacion[indices_validacion] = pliegue
    if (asignacion < 0).any():
        raise RuntimeError("No todas las filas recibieron un pliegue.")
    return pd.DataFrame(
        {
            "fila_entrenamiento": np.arange(cantidad_filas),
            "pliegue": asignacion,
        }
    )


def crear_splits_cv(asignacion, cantidad_filas):
    """Convierte la asignación guardada al formato esperado por GridSearchCV."""
    if len(asignacion) != cantidad_filas:
        raise RuntimeError("La asignación de pliegues no coincide con entrenamiento.")
    if asignacion["fila_entrenamiento"].tolist() != list(range(cantidad_filas)):
        raise RuntimeError("El orden de filas de la asignación de pliegues cambió.")
    if sorted(asignacion["pliegue"].unique().tolist()) != list(range(PLIEGUES)):
        raise RuntimeError("La asignación no contiene exactamente cinco pliegues.")

    indices = np.arange(cantidad_filas)
    pliegues = asignacion["pliegue"].to_numpy()
    return [
        (indices[pliegues != pliegue], indices[pliegues == pliegue])
        for pliegue in range(PLIEGUES)
    ]


def crear_pipeline_random_forest(n_jobs_modelo=1):
    """Reproduce el pipeline del grid original."""
    return Pipeline(
        steps=[
            ("preprocesamiento", preprocesamiento_random_forest()),
            (
                "modelo",
                RandomForestRegressor(
                    random_state=SEMILLA_PRINCIPAL,
                    n_jobs=n_jobs_modelo,
                ),
            ),
        ]
    )


def grid_desde_manifiesto(manifiesto_bloque):
    """Convierte las configuraciones de un bloque a grids unitarios."""
    grids = []
    for texto in manifiesto_bloque["parametros_json"]:
        parametros = json.loads(texto)
        grids.append({nombre: [valor] for nombre, valor in parametros.items()})
    return grids


def ruta_resultado_bloque(numero):
    """Devuelve el CSV final de un bloque."""
    return RESULTADOS_BLOQUES / f"resultados_cv_bloque_{numero:02d}.csv"


def ruta_estado_bloque(numero):
    """Devuelve el JSON de control de un bloque."""
    return RESULTADOS_BLOQUES / f"estado_bloque_{numero:02d}.json"


def guardar_csv_atomico(datos, ruta):
    """Guarda un CSV completo sin dejar una salida parcial."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(str(ruta) + ".tmp")
    datos.to_csv(temporal, index=False, encoding="utf-8-sig")
    temporal.replace(ruta)


def guardar_json_atomico(datos, ruta):
    """Guarda JSON completo sin dejar una salida parcial."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = Path(str(ruta) + ".tmp")
    temporal.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporal.replace(ruta)


def cargar_json(ruta):
    """Lee un archivo JSON."""
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def verificar_codigo_original(metadatos):
    """Impide mezclar bloques producidos con versiones distintas del código."""
    comprobaciones = {
        "sha256_funciones_busqueda": hash_archivo(ARCHIVO_FUNCIONES_ORIGINAL),
        "sha256_busqueda_monolitica": hash_archivo(ARCHIVO_BUSQUEDA_ORIGINAL),
        "version_sklearn": SKLEARN_VERSION,
    }
    for nombre, valor_actual in comprobaciones.items():
        if metadatos[nombre] != valor_actual:
            raise RuntimeError(
                f"Cambió {nombre}. No se pueden mezclar resultados de versiones "
                "diferentes."
            )


def verificar_datos(metadatos, entrenamiento, validacion, test_reservado):
    """Impide mezclar bloques ejecutados con datasets diferentes."""
    conjuntos = {
        "entrenamiento": entrenamiento,
        "validacion": validacion,
        "test": test_reservado,
    }
    for nombre, datos in conjuntos.items():
        filas_esperadas = metadatos[f"filas_{nombre}"]
        huella_esperada = metadatos[f"sha256_{nombre}"]
        if len(datos) != filas_esperadas:
            raise RuntimeError(f"Cambió la cantidad de filas de {nombre}.")
        if huella_dataframe(datos) != huella_esperada:
            raise RuntimeError(f"Cambió el contenido o el orden de {nombre}.")


def cargar_plan():
    """Carga y valida los tres archivos de control de la búsqueda."""
    faltantes = [
        ruta for ruta in [MANIFIESTO, ASIGNACION_PLIEGUES, METADATOS]
        if not ruta.exists()
    ]
    if faltantes:
        nombres = ", ".join(str(ruta) for ruta in faltantes)
        raise RuntimeError(
            "Primero ejecute preparar_busqueda_por_bloques.py. Faltan: " + nombres
        )
    manifiesto = pd.read_csv(
        MANIFIESTO,
        dtype={
            "id_configuracion": str,
            "parametros_json": str,
        },
    )
    asignacion = pd.read_csv(ASIGNACION_PLIEGUES)
    metadatos = cargar_json(METADATOS)

    if len(manifiesto) != TOTAL_CONFIGURACIONES:
        raise RuntimeError("El manifiesto no contiene 384 configuraciones.")
    if manifiesto["id_configuracion"].nunique() != TOTAL_CONFIGURACIONES:
        raise RuntimeError("El manifiesto contiene identificadores duplicados.")
    if hash_grid(manifiesto) != metadatos["sha256_grid"]:
        raise RuntimeError("El manifiesto no coincide con su huella guardada.")
    verificar_codigo_original(metadatos)
    return manifiesto, asignacion, metadatos


def validar_resultado_bloque(resultado, manifiesto_bloque, numero):
    """Comprueba integridad, métricas y pertenencia de un bloque."""
    if len(resultado) != CONFIGURACIONES_POR_BLOQUE:
        raise RuntimeError(f"El resultado del bloque {numero} no tiene 48 filas.")
    if resultado["id_configuracion"].nunique() != CONFIGURACIONES_POR_BLOQUE:
        raise RuntimeError(f"El bloque {numero} contiene resultados duplicados.")
    esperadas = set(manifiesto_bloque["id_configuracion"])
    obtenidas = set(resultado["id_configuracion"])
    if obtenidas != esperadas:
        raise RuntimeError(f"El bloque {numero} no coincide con su manifiesto.")
    columnas_metricas = [
        f"mean_test_{metrica}" for metrica in SCORING
    ]
    if resultado[columnas_metricas].isna().any().any():
        raise RuntimeError(f"El bloque {numero} contiene métricas nulas.")
