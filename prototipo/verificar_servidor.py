"""Verifica que el servidor tenga lo necesario para la pestaña "Estimación lechera (PAA)".

No instala ni modifica nada: informa versiones, revisa que app.py tenga las
funciones que usa la pestaña y, si CatBoost está instalado, carga el modelo y
estima perfiles. Deja una copia del resultado en logs/ como evidencia.

Uso, desde la terminal de JupyterLab:
    python /app/PAA/prototipo/verificar_servidor.py
    python /app/PAA/prototipo/verificar_servidor.py --app /otra/ruta/app.py
"""

import argparse
import ast
import importlib
import platform
import sys
from datetime import datetime
from pathlib import Path

CARPETA_PROTOTIPO = Path(__file__).resolve().parent
RAIZ = CARPETA_PROTOTIPO.parents[1]
# La misma carpeta que app.py agrega al inicio de sys.path en el servidor.
LIBRERIAS = Path("/app/librerias")
STREAMLIT_MINIMO = (1, 49)
PAQUETES = ["streamlit", "pandas", "numpy", "plotly", "scipy", "six", "catboost"]
ICONOS = {"layers", "milk", "sprout", "shield", "map", "clock"}
# Funciones de app.py que la pestaña recibe como argumento, con los parámetros que usa.
FUNCIONES = {
    "metric_card": ["label", "value", "description", "source", "icon", "detail"],
    "chart_context": ["question", "source"],
    "insight_box": ["title", "text"],
    "add_plot_style": ["fig", "height", "legend_title", "title", "subtitle"],
}

lineas = []
problemas = []


def informar(texto=""):
    print(texto)
    lineas.append(texto)


def problema(texto):
    informar(f"  [PROBLEMA] {texto}")
    problemas.append(texto)


def version_tupla(texto):
    partes = []
    for parte in texto.split(".")[:2]:
        digitos = "".join(c for c in parte if c.isdigit())
        partes.append(int(digitos or 0))
    return tuple(partes)


def revisar_paquetes():
    informar("1. Paquetes de Python")
    versiones = {}
    for nombre in PAQUETES:
        try:
            modulo = importlib.import_module(nombre)
        except Exception as error:  # noqa: BLE001 - se informa cualquier fallo de importación
            versiones[nombre] = None
            informar(f"  {nombre:<10} FALTA ({type(error).__name__}: {error})")
            continue
        versiones[nombre] = getattr(modulo, "__version__", "?")
        origen = Path(getattr(modulo, "__file__", "") or "").parent
        informar(f"  {nombre:<10} {versiones[nombre]:<10} {origen}")
    for nombre in PAQUETES:
        if versiones[nombre] is None:
            problema(f"Falta el paquete {nombre}.")
    if versiones.get("streamlit") and version_tupla(versiones["streamlit"]) < STREAMLIT_MINIMO:
        problema(f"Streamlit {versiones['streamlit']} es anterior a "
                 f"{'.'.join(map(str, STREAMLIT_MINIMO))}; la pestaña usa width='stretch'.")
    return versiones


def revisar_archivos(ruta_app):
    informar("2. Archivos")
    carpeta_app = ruta_app.parent
    requeridos = [
        ruta_app,
        carpeta_app / "tab_estimacion_lechera.py",
        carpeta_app / "tablas.py",
        CARPETA_PROTOTIPO / "inferencia_lecheria.py",
        CARPETA_PROTOTIPO / "artefactos" / "modelo_catboost_top12.cbm",
        CARPETA_PROTOTIPO / "artefactos" / "metadatos_prototipo.json",
        CARPETA_PROTOTIPO / "artefactos" / "dataset_prototipo.csv",
        CARPETA_PROTOTIPO / "artefactos" / "metricas_prueba.csv",
    ]
    for ruta in requeridos:
        if ruta.exists():
            informar(f"  ok  {ruta} ({ruta.stat().st_size:,} bytes)")
        else:
            problema(f"No existe {ruta}.")
    # La pestaña busca el prototipo en <carpeta de app.py>/../PAA/prototipo.
    esperado = carpeta_app.parent / "PAA" / "prototipo"
    if esperado.resolve() != CARPETA_PROTOTIPO:
        problema(f"La pestaña buscará el prototipo en {esperado}, pero este script está en {CARPETA_PROTOTIPO}.")
    for ruta in (carpeta_app / "tab_estimacion_lechera.py", carpeta_app / "tablas.py"):
        if ruta.exists():
            try:
                compile(ruta.read_text(encoding="utf-8"), str(ruta), "exec")
            except SyntaxError as error:
                problema(f"{ruta.name} no compila con este Python: {error}")


def revisar_app(ruta_app):
    informar("3. Funciones de app.py que usa la pestaña")
    if not ruta_app.exists():
        return
    texto = ruta_app.read_text(encoding="utf-8")
    arbol = ast.parse(texto)
    funciones = {n.name: n for n in ast.walk(arbol) if isinstance(n, ast.FunctionDef)}
    for nombre, parametros in FUNCIONES.items():
        if nombre not in funciones:
            problema(f"app.py no define {nombre}().")
            continue
        actuales = [a.arg for a in funciones[nombre].args.args]
        faltan = [p for p in parametros if p not in actuales]
        if faltan:
            problema(f"{nombre}() no tiene los parámetros {', '.join(faltan)}.")
        else:
            informar(f"  ok  {nombre}({', '.join(actuales)})")

    iconos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign) and any(getattr(t, "id", "") == "ICON_PATHS" for t in nodo.targets):
            if isinstance(nodo.value, ast.Dict):
                iconos = {k.value for k in nodo.value.keys if isinstance(k, ast.Constant)}
    faltan = sorted(ICONOS - iconos)
    if faltan:
        problema(f"ICON_PATHS de app.py no tiene los íconos {', '.join(faltan)}.")
    else:
        informar(f"  ok  ICON_PATHS tiene {', '.join(sorted(ICONOS))}")

    conectada = "tab_estimacion_lechera.render(" in texto
    informar(f"  {'ok ' if conectada else '...'} pestaña {'ya conectada' if conectada else 'todavía no conectada'} en app.py")


def probar_modelo():
    informar("4. Carga del modelo y estimación")
    sys.path.insert(0, str(CARPETA_PROTOTIPO))
    try:
        import inferencia_lecheria as inf
        import pandas as pd

        modelo = inf.cargar_modelo()
        metadatos = inf.cargar_metadatos()
        dataset = inf.cargar_dataset()
        informar(f"  ok  modelo con {len(modelo.feature_names_)} variables; dataset con {len(dataset):,} perfiles")
        salida = inf.predecir(modelo, dataset, metadatos)
        informar(f"  ok  {len(salida):,} perfiles estimados; litros estimados totales "
                 f"{salida['litros_estimados'].sum():,.0f}; bandas "
                 f"{salida['banda_confiabilidad'].value_counts().to_dict()}")
        perfil = pd.DataFrame([{
            "total_bovinos_leche": 300, "toros": 2, "vacas_ordene": 150, "vacas_secas": 30,
            "vaquillonas_mas_2": 30, "vaquillonas_1_2": 40, "terneros_menos_1": 20, "terneras_menos_1": 28,
            "Superficie": 250.0, "UnidadesGanaderas": 260.0, "SuperficieGanadera": 240.0,
        }])
        manual = inf.predecir(modelo, perfil, metadatos).iloc[0]
        informar(f"  ok  perfil de prueba (150 vacas en ordeñe): {manual['litros_estimados']:,.0f} L, "
                 f"confiabilidad {manual['banda_confiabilidad']}")
    except Exception as error:  # noqa: BLE001 - se informa cualquier fallo del pipeline
        problema(f"No se pudo estimar: {type(error).__name__}: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app", type=Path, default=RAIZ / "streamlit" / "app.py",
                        help="Ruta del app.py del dashboard (por defecto /app/streamlit/app.py).")
    args = parser.parse_args()
    ruta_app = args.app.resolve()

    if LIBRERIAS.exists():
        sys.path.insert(0, str(LIBRERIAS))
    informar(f"Verificación del prototipo PAA en el servidor · {datetime.now():%Y-%m-%d %H:%M:%S}")
    informar(f"Python {platform.python_version()} ({sys.executable}) en {platform.platform()}")
    informar(f"Librerías locales {LIBRERIAS}: {'sí' if LIBRERIAS.exists() else 'no existe'}")
    informar()
    versiones = revisar_paquetes()
    informar()
    revisar_archivos(ruta_app)
    informar()
    revisar_app(ruta_app)
    informar()
    if versiones.get("catboost"):
        probar_modelo()
    else:
        informar("4. Carga del modelo y estimación: se omite porque falta CatBoost")
    informar()
    informar("RESULTADO: todo listo" if not problemas else f"RESULTADO: {len(problemas)} problema(s) para resolver")

    carpeta_logs = CARPETA_PROTOTIPO / "logs"
    try:
        carpeta_logs.mkdir(exist_ok=True)
        ruta_log = carpeta_logs / f"verificacion_servidor_{datetime.now():%Y%m%d_%H%M%S}.log"
        ruta_log.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        print(f"Registro guardado en {ruta_log}")
    except OSError as error:
        print(f"No se pudo guardar el registro: {error}")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
