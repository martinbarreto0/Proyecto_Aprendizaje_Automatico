"""Busca hiperparámetros en los modelos MGAP ya evaluados.

No toca resultados_mgap_snig ni PAA/data. El 20% de prueba con semilla 42 se
mantiene reservado; la búsqueda trabaja solo dentro del 80% de desarrollo.
"""
# Herramientas para rutas, combinaciones de parámetros y medición de tiempos.
from pathlib import Path
from itertools import product
from time import perf_counter
# Utilidades estándar para argumentos, memoria, serialización y aleatoriedad.
import argparse
import gc
import hashlib
import json
import os
import random
import sys

# `BASE` es este directorio; `PADRE` contiene los módulos compartidos MGAP.
BASE = Path(__file__).resolve().parent
PADRE = BASE.parent / 'modelos_base'
# Permiten ejecutar este archivo directamente e importar módulos locales.
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(PADRE))

# Define las opciones disponibles al ejecutar el script desde la terminal.
parser = argparse.ArgumentParser(description=__doc__)
# Limita la ejecución a una fuente concreta de datos.
parser.add_argument('--tabla', help='Procesar una sola tabla.')
# Limita la búsqueda a una de las columnas objetivo MGAP.
parser.add_argument('--objetivo', help='Procesar un solo objetivo MGAP.')
# Sustituye cuántas configuraciones iniciales se van a probar.
parser.add_argument('--intentos', type=int, help='Reemplaza el presupuesto inicial por objetivo.')
# Define cuántos finalistas se validan con varias semillas.
parser.add_argument('--top', type=int, help='Cantidad de candidatos para validar con varias semillas.')
# Permite indicar explícitamente las semillas de repetición.
parser.add_argument('--semillas', help='Semillas separadas por coma, por ejemplo 43,44,45.')
# Define la cantidad máxima de hilos de cómputo.
parser.add_argument('--hilos', type=int, default=4)
# Indica que debe recuperarse el CSV parcial ya generado.
parser.add_argument('--reanudar', action='store_true')
# Lee y valida los argumentos recibidos.
args = parser.parse_args()
# Limita el paralelismo de las bibliotecas de aprendizaje automático.
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

# Pandas maneja tablas; Scikit-learn aporta modelos y métricas.
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Listas de objetivos y tablas válidas del proyecto.
from configuracion_MGAP_snig import OBJETIVOS, TABLAS
# `cargar_tabla` prepara datos; `codificar` transforma categorías;
# `separar_aleatorio` realiza la división estratificada; `calcular_pesos`
# compensa el desbalance de clases. `SALIDAS` apunta a resultados previos.
from funciones_MGAP_snig import (SALIDAS, SNIG, calcular_pesos, cargar_tabla,
                                 codificar, separar_aleatorio)
from espacios_busqueda_MGAP_snig import (ESPACIOS, EXPONENTES_PESOS,
                                         MARGEN_DOS_FAMILIAS, MEJORA_MINIMA_F1,
                                         presupuesto)

# Carpeta exclusiva para los resultados de esta búsqueda.
RESULTADOS = BASE / 'resultados'


def familia(nombre):
    """Traduce el prefijo de un modelo a la familia de clasificador usada."""
    if nombre.startswith('RF_'):
        # Identificador abreviado para Random Forest.
        return 'RF'
    if nombre.startswith('ExtraTrees_'):
        # Identificador de la familia Extra Trees.
        return 'ExtraTrees'
    if nombre.startswith('HistBoost_'):
        # Identificador de HistGradientBoosting.
        return 'HistBoost'
    raise ValueError(f'Modelo no ajustable: {nombre}')


def parametros_base(nombre, max_iter):
    """Reconstruye los hiperparámetros del modelo actual para compararlo."""
    # Punto de partida: ponderación suavizada de las clases.
    resultado = {'exponente_pesos': 0.5}
    if nombre.startswith('RF_'):
        return resultado | {'n_estimators': 100, 'max_depth': 20,
                            'min_samples_split': 2, 'min_samples_leaf': 3,
                            'max_features': 'sqrt'}
    if nombre.startswith('ExtraTrees_'):
        return resultado | {'n_estimators': 100, 'max_depth': 25,
                            'min_samples_split': 2, 'min_samples_leaf': 3,
                            'max_features': 'sqrt'}
    if nombre.startswith('HistBoost_'):
        return resultado | {'max_iter': max_iter or 250, 'learning_rate': 0.1,
                            'max_leaf_nodes': 31, 'min_samples_leaf': 20,
                            'l2_regularization': 1.0}
    raise ValueError(f'Modelo no ajustable: {nombre}')


def crear_modelo(nombre_familia, parametros, y_ajuste):
    """Instancia la familia solicitada aplicando exactamente sus parámetros."""
    if nombre_familia == 'RF':
        return RandomForestClassifier(random_state=42, n_jobs=args.hilos,
            n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'],
            min_samples_split=parametros['min_samples_split'],
            min_samples_leaf=parametros['min_samples_leaf'],
            max_features=parametros['max_features'])
    if nombre_familia == 'ExtraTrees':
        return ExtraTreesClassifier(random_state=42, n_jobs=args.hilos,
            n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'],
            min_samples_split=parametros['min_samples_split'],
            min_samples_leaf=parametros['min_samples_leaf'],
            max_features=parametros['max_features'])
    if nombre_familia == 'HistBoost':
        # La parada temprana solo es segura con datos suficientes y dos casos por clase.
        detener = len(y_ajuste) > 10_000 and y_ajuste.value_counts().min() >= 2
        return HistGradientBoostingClassifier(random_state=42,
            max_iter=parametros['max_iter'], learning_rate=parametros['learning_rate'],
            max_leaf_nodes=parametros['max_leaf_nodes'],
            min_samples_leaf=parametros['min_samples_leaf'],
            l2_regularization=parametros['l2_regularization'],
            early_stopping=detener)
    raise ValueError(f'Familia desconocida: {nombre_familia}')


def identificador(nombre_familia, parametros):
    """Genera un ID estable para detectar configuraciones ya evaluadas."""
    # Ordenar las claves garantiza el mismo texto aunque el dict se haya creado distinto.
    texto = json.dumps({'familia': nombre_familia, 'parametros': parametros}, sort_keys=True)
    # Se conserva solo un prefijo de 16 caracteres del hash criptográfico.
    return hashlib.sha256(texto.encode('utf-8')).hexdigest()[:16]


def candidatos(nombre_familia, nombre_base, parametros_actuales, cantidad, semilla):
    """Devuelve la base y una muestra reproducible del espacio indicado."""
    # Extrae los nombres de los hiperparámetros configurables para esta familia.
    claves = list(ESPACIOS[nombre_familia])
    # Combina cada valor posible con cada intensidad de ponderación de clases.
    combinaciones = list(product(*(ESPACIOS[nombre_familia][clave] for clave in claves),
                                 EXPONENTES_PESOS))
    # Generador local: la misma semilla produce la misma muestra de candidatos.
    generador = random.Random(semilla)
    generador.shuffle(combinaciones)
    # El modelo actual siempre se mide como línea de base.
    salida = [parametros_actuales]
    if cantidad <= 1:
        return salida
    # Evita incluir dos veces una misma combinación, incluso si coincide con la base.
    vistos = {identificador(nombre_familia, parametros_actuales)}
    for valores in combinaciones:
        # Convierte la tupla de valores en el diccionario que espera Scikit-learn.
        parametros = dict(zip(claves, valores[:-1]))
        parametros['exponente_pesos'] = valores[-1]
        clave = identificador(nombre_familia, parametros)
        if clave not in vistos:
            salida.append(parametros)
            vistos.add(clave)
        if len(salida) >= cantidad:
            break
    return salida


def medir(tabla, objetivo, nombre_familia, parametros, X_desarrollo, y_desarrollo, semilla):
    """Entrena una configuración y devuelve sus métricas sobre validación interna."""
    # Divide solo el 80 % de desarrollo; el test con semilla 42 permanece reservado.
    X_ajuste, X_val, y_ajuste, y_val = separar_aleatorio(X_desarrollo, y_desarrollo, semilla)
    # Convierte las variables categóricas a columnas numéricas compatibles.
    X_ajuste, X_val = codificar(X_ajuste, X_val, tabla)
    modelo = crear_modelo(nombre_familia, parametros, y_ajuste[objetivo])
    # Inicia la medición de tiempo antes del ajuste del modelo.
    inicio = perf_counter()
    # Los pesos compensan el desbalance de las clases según el exponente probado.
    modelo.fit(X_ajuste, y_ajuste[objetivo],
               sample_weight=calcular_pesos(y_ajuste[objetivo], parametros['exponente_pesos']))
    pred = modelo.predict(X_val)
    # Evalúa todas las etiquetas presentes en esta partición de validación.
    etiquetas = sorted(y_val[objetivo].unique())
    fila = {
        'tabla': tabla,
        'objetivo': objetivo,
        'familia': nombre_familia,
        'id_configuracion': identificador(nombre_familia, parametros),
        'parametros': json.dumps(parametros, sort_keys=True),
        'semilla_validacion': semilla,
        'f1_macro': f1_score(y_val[objetivo], pred, labels=etiquetas, average='macro', zero_division=0),
        'accuracy': accuracy_score(y_val[objetivo], pred),
        'filas_ajuste': len(y_ajuste),
        'filas_validacion': len(y_val),
        'segundos': perf_counter() - inicio,
    }
    # Libera objetos grandes antes de continuar con el siguiente candidato.
    del modelo, X_ajuste, X_val, y_ajuste, y_val
    gc.collect()
    return fila


def guardar(ruta, filas):
    """Persiste el avance en CSV, permitiendo continuar después con --reanudar."""
    pd.DataFrame(filas).to_csv(ruta, index=False)


def modelos_para_ajustar(carpeta, objetivo):
    """Elige la mejor familia previa y, si casi empata, también la segunda."""
    comparacion = pd.read_csv(carpeta / 'comparacion.csv')
    opciones = comparacion[(comparacion['objetivo'] == objetivo) &
                           (comparacion['modelo'] != 'Mayoritaria')].sort_values(
                               ['f1_macro', 'accuracy'], ascending=False).reset_index(drop=True)
    # La primera fila ya está ordenada por F1 macro y luego por exactitud.
    mejor = opciones.iloc[0]
    salida = [mejor]
    if len(opciones) > 1 and mejor['f1_macro'] - opciones.iloc[1]['f1_macro'] < MARGEN_DOS_FAMILIAS:
        salida.append(opciones.iloc[1])
    return salida


def resumen_objetivo(filas, parametros_base_id, mejora_minima):
    """Promedia repeticiones y decide si la mejora justifica cambiar la base."""
    datos = pd.DataFrame(filas)
    agrupado = datos.groupby(['id_configuracion', 'familia', 'parametros'], as_index=False).agg(
        f1_macro_media=('f1_macro', 'mean'), f1_macro_desvio=('f1_macro', 'std'),
        accuracy_media=('accuracy', 'mean'), semillas=('semilla_validacion', 'nunique'),
        segundos_medios=('segundos', 'mean'))
    # Una sola semilla no tiene desvío muestral; se representa como cero.
    agrupado['f1_macro_desvio'] = agrupado['f1_macro_desvio'].fillna(0)
    agrupado = agrupado.sort_values(
        ['f1_macro_media', 'f1_macro_desvio', 'accuracy_media'], ascending=[False, True, False]
    ).reset_index(drop=True)
    base = agrupado.loc[agrupado['id_configuracion'].eq(parametros_base_id)].iloc[0]
    mejor = agrupado.iloc[0]
    mejora = float(mejor['f1_macro_media'] - base['f1_macro_media'])
    # No se recomienda un reemplazo sin superar el umbral mínimo de mejora.
    acepta = mejor['id_configuracion'] != parametros_base_id and mejora >= mejora_minima
    elegido = mejor if acepta else base
    return agrupado, base, elegido, mejora, acepta


# Procesa la tabla solicitada o todas las tablas configuradas.
tablas = [args.tabla] if args.tabla else list(TABLAS)
if args.tabla and args.tabla not in TABLAS:
    raise ValueError(f'Tabla desconocida: {args.tabla}')
if args.objetivo and args.objetivo not in OBJETIVOS:
    raise ValueError(f'Objetivo desconocido: {args.objetivo}')

for tabla in tablas:
    # Obtiene un presupuesto independiente, para poder sobrescribirlo sin efectos globales.
    plan = presupuesto(tabla)
    if args.intentos:
        plan['intentos'] = args.intentos
    if args.top:
        plan['top'] = args.top
    if args.semillas:
        plan['semillas'] = [int(valor) for valor in args.semillas.split(',')]
    objetivos = [args.objetivo] if args.objetivo else OBJETIVOS
    # Resultados previos para elegir familia y carpeta nueva para esta exploración.
    carpeta_actual = SALIDAS / tabla
    carpeta_resultados = RESULTADOS / tabla
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    # Carga las variables predictoras `X` y los dos objetivos `y`.
    df, X, y = cargar_tabla(tabla)
    # Descarta aquí el 20 % de test, que se evaluará en el otro script.
    X_desarrollo, _, y_desarrollo, _ = separar_aleatorio(X, y, semilla=42)
    seleccionados = {}

    for objetivo in objetivos:
        print(f'\n=== {tabla} / {objetivo} ===', flush=True)
        ruta_resultados = carpeta_resultados / f'busqueda_{objetivo}.csv'
        # Reutiliza los intentos guardados solo cuando se pidió reanudar.
        filas = (pd.read_csv(ruta_resultados).to_dict('records')
                 if args.reanudar and ruta_resultados.exists() else [])
        # Selecciona una o dos familias competitivas de la comparación previa.
        candidatos_actuales = modelos_para_ajustar(carpeta_actual, objetivo)
        base_fila = candidatos_actuales[0]
        nombre_base = base_fila['modelo']
        parametros_actuales = parametros_base(nombre_base, int(base_fila.get('max_iter', 0)))
        parametros_actuales['exponente_pesos'] = float(base_fila['exponente_pesos'])
        id_base = identificador(familia(nombre_base), parametros_actuales)
        # Conserva cada familia una sola vez, aunque varias filas la representen.
        familias = []
        for candidato in candidatos_actuales:
            nombre = candidato['modelo']
            nombre_familia = familia(nombre)
            if nombre_familia not in familias:
                familias.append(nombre_familia)

        # Reparte los intentos entre familias, garantizando al menos uno por familia.
        por_familia = max(1, plan['intentos'] // len(familias))
        for indice, nombre_familia in enumerate(familias):
            # Recupera la mejor fila previa correspondiente a esta familia.
            fila_actual = next(c for c in candidatos_actuales if familia(c['modelo']) == nombre_familia)
            parametros_iniciales = parametros_base(fila_actual['modelo'], int(fila_actual.get('max_iter', 0)))
            parametros_iniciales['exponente_pesos'] = float(fila_actual['exponente_pesos'])
            for parametros in candidatos(nombre_familia, fila_actual['modelo'], parametros_iniciales,
                                         por_familia, 1000 + indice):
                clave = identificador(nombre_familia, parametros)
                # No repite el ensayo inicial si ya quedó almacenado.
                if any(f['id_configuracion'] == clave and int(f['semilla_validacion']) == 43 for f in filas):
                    continue
                print(f'Inicial: {nombre_familia} {clave}', flush=True)
                filas.append(medir(tabla, objetivo, nombre_familia, parametros,
                                   X_desarrollo, y_desarrollo, 43))
                guardar(ruta_resultados, filas)

        # Clasifica el primer ensayo de cada configuración y extrae finalistas.
        iniciales, _, _, _, _ = resumen_objetivo(filas, id_base, MEJORA_MINIMA_F1)
        ids_repetir = iniciales['id_configuracion'].head(plan['top']).tolist()
        # La línea de base también se repite para hacer una comparación justa.
        if id_base not in ids_repetir:
            ids_repetir.append(id_base)
        por_id = iniciales.set_index('id_configuracion')
        for clave in ids_repetir:
            fila = por_id.loc[clave]
            parametros = json.loads(fila['parametros'])
            for semilla in plan['semillas']:
                # Al reanudar, omite cada pareja configuración-semilla ya ejecutada.
                if any(f['id_configuracion'] == clave and int(f['semilla_validacion']) == semilla for f in filas):
                    continue
                print(f'Repetición semilla {semilla}: {fila["familia"]} {clave}', flush=True)
                filas.append(medir(tabla, objetivo, fila['familia'], parametros,
                                   X_desarrollo, y_desarrollo, semilla))
                guardar(ruta_resultados, filas)

        tabla_resumen, base, elegido, mejora, acepta = resumen_objetivo(
            filas, id_base, MEJORA_MINIMA_F1
        )
        # Exporta el ranking con medias y variabilidad de cada configuración.
        tabla_resumen.to_csv(carpeta_resultados / f'resumen_{objetivo}.csv', index=False)
        seleccionados[objetivo] = {
            'modelo_actual': nombre_base,
            'parametros_actuales': parametros_actuales,
            'f1_macro_base_media': float(base['f1_macro_media']),
            'id_configuracion_elegida': elegido['id_configuracion'],
            'familia_elegida': elegido['familia'],
            'parametros_elegidos': json.loads(elegido['parametros']),
            'f1_macro_elegido_media': float(elegido['f1_macro_media']),
            'f1_macro_elegido_desvio': float(elegido['f1_macro_desvio']),
            'mejora_sobre_base': mejora,
            'recomienda_reemplazo': acepta,
            'umbral_mejora': MEJORA_MINIMA_F1,
        }
        print(json.dumps(seleccionados[objetivo], ensure_ascii=False, indent=2), flush=True)

    # Deja la selección lista para la evaluación definitiva sobre el test reservado.
    (carpeta_resultados / 'mejores_parametros.json').write_text(
        json.dumps({'tabla': tabla, 'objetivos': seleccionados}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    del df, X, y, X_desarrollo, y_desarrollo
    gc.collect()
