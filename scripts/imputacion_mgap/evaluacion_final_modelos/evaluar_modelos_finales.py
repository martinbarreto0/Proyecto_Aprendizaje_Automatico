"""Compara los tres métodos MGAP en un nuevo test aleatorio congelado.

Los modelos ya se eligieron antes de ejecutar este script. Aquí no hay búsqueda
ni validación: para cada tabla y objetivo se mide una vez el modelo original,
la configuración de hiperparámetros y el ensamble previamente seleccionado.
"""
from pathlib import Path
from time import perf_counter
import argparse
import gc
import json
import os
import sys


BASE = Path(__file__).resolve().parent
MGAP = BASE.parent
ENSAMBLE_DIR = MGAP / 'ensamble'
MODELOS_BASE = MGAP / 'modelos_base'
sys.path.insert(0, str(MODELOS_BASE))
sys.path.insert(0, str(ENSAMBLE_DIR))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tabla', help='Evalúa solo una tabla.')
parser.add_argument('--objetivo', help='Evalúa solo un objetivo MGAP.')
parser.add_argument('--hilos', type=int, default=4)
parser.add_argument('--max-knn', type=int, default=200_000,
                    help='Máximo de filas de entrenamiento para KNN en ensambles.')
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from configuracion_MGAP_snig import OBJETIVOS, TABLAS
from funciones_MGAP_snig import calcular_pesos, cargar_tabla, codificar, separar_aleatorio
from funciones_ensamble_MGAP_snig import probabilidades_bases, probabilidades_oof, puntajes_ensamble, entrenar_meta


MANIFIESTO = BASE / 'modelos_congelados.json'
RESULTADOS = BASE / 'resultados_finales'


def crear_modelo(familia, parametros, y_train):
    """Crea los clasificadores individuales con los parámetros congelados."""
    if familia == 'RF':
        return RandomForestClassifier(random_state=42, n_jobs=args.hilos,
            n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'],
            min_samples_split=parametros['min_samples_split'],
            min_samples_leaf=parametros['min_samples_leaf'], max_features=parametros['max_features'])
    if familia == 'ExtraTrees':
        return ExtraTreesClassifier(random_state=42, n_jobs=args.hilos,
            n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'],
            min_samples_split=parametros['min_samples_split'],
            min_samples_leaf=parametros['min_samples_leaf'], max_features=parametros['max_features'])
    if familia == 'HistBoost':
        detener = len(y_train) > 10_000 and y_train.value_counts().min() >= 2
        return HistGradientBoostingClassifier(random_state=42,
            max_iter=parametros['max_iter'], learning_rate=parametros['learning_rate'],
            max_leaf_nodes=parametros['max_leaf_nodes'], min_samples_leaf=parametros['min_samples_leaf'],
            l2_regularization=parametros['l2_regularization'], early_stopping=detener)
    raise ValueError(f'Familia desconocida: {familia}')


def metricas(y_real, prediccion):
    """Calcula las mismas métricas para los tres métodos."""
    etiquetas = sorted(y_real.unique())
    return {
        'accuracy': accuracy_score(y_real, prediccion),
        'f1_macro': f1_score(y_real, prediccion, labels=etiquetas, average='macro', zero_division=0),
    }


def evaluar_individual(nombre, familia, parametros, X_train, X_test, y_train, y_test):
    """Entrena y mide el modelo original o el de hiperparámetros."""
    inicio = perf_counter()
    modelo = crear_modelo(familia, parametros, y_train)
    modelo.fit(X_train, y_train, sample_weight=calcular_pesos(y_train, parametros['exponente_pesos']))
    fila = metricas(y_test, modelo.predict(X_test))
    fila.update(metodo=nombre, familia=familia, parametros=json.dumps(parametros, sort_keys=True),
                segundos=perf_counter() - inicio)
    del modelo
    gc.collect()
    return fila


def evaluar_ensamble(info, X_train, X_test, y_train, y_test):
    """Mide exactamente el ensamble congelado, sin volver a elegirlo."""
    elegido = info['ensamble']
    nombre = elegido['ensamble']
    inicio = perf_counter()
    # Entrena las cuatro bases una vez y reutiliza sus probabilidades.
    probs, clases, segundos_bases = probabilidades_bases(
        X_train, y_train, X_test, tuple(elegido['histboost']), args.hilos, args.max_knn)
    meta = None
    if nombre == 'E4_stacking':
        # Las probabilidades fuera de pliegue entrenan la meta sin mirar el test nuevo.
        oof, _, _ = probabilidades_oof(X_train, y_train, tuple(elegido['histboost']),
                                       args.hilos, args.max_knn, pliegues=3)
        meta = entrenar_meta(oof, y_train)
    prediccion = clases[puntajes_ensamble(nombre, probs, meta).argmax(axis=1)]
    fila = metricas(y_test, prediccion)
    fila.update(metodo='Ensamble_' + nombre, familia='Ensamble',
                parametros=json.dumps(elegido, sort_keys=True),
                segundos=perf_counter() - inicio,
                segundos_bases=sum(segundos_bases.values()))
    gc.collect()
    return fila


if not MANIFIESTO.exists():
    raise FileNotFoundError('Falta modelos_congelados.json. Ejecute primero preparar_manifest_modelos.py.')
manifest = json.loads(MANIFIESTO.read_text(encoding='utf-8'))
tablas = [args.tabla] if args.tabla else list(manifest['tablas'])
if args.tabla and args.tabla not in manifest['tablas']:
    raise ValueError(f'Tabla desconocida o no congelada: {args.tabla}')
if args.objetivo and args.objetivo not in OBJETIVOS:
    raise ValueError(f'Objetivo MGAP desconocido: {args.objetivo}')

RESULTADOS.mkdir(exist_ok=True)
for tabla in tablas:
    salida = RESULTADOS / tabla
    salida.mkdir(exist_ok=True)
    archivo = salida / 'metricas_finales.csv'
    if archivo.exists():
        raise FileExistsError(f'{archivo} ya existe. No se repite el test final.')

    print(f'\n=== {tabla} ===', flush=True)
    _, X, y = cargar_tabla(tabla)
    X_train, X_test, y_train, y_test = separar_aleatorio(X, y, semilla=manifest['semilla_train_test_split'])
    X_train, X_test = codificar(X_train, X_test, tabla)
    filas = []
    objetivos = [args.objetivo] if args.objetivo else list(manifest['tablas'][tabla]['objetivos'])

    for objetivo in objetivos:
        print(f'  {objetivo}', flush=True)
        info = manifest['tablas'][tabla]['objetivos'][objetivo]
        for nombre, familia, parametros in [
            ('Original_' + info['modelo_original'], info['familia_original'], info['parametros_originales']),
            ('Hiperparametros_' + info['familia_hiperparametros'], info['familia_hiperparametros'],
             info['parametros_hiperparametros']),
        ]:
            fila = evaluar_individual(nombre, familia, parametros, X_train, X_test,
                                      y_train[objetivo], y_test[objetivo])
            fila.update(tabla=tabla, objetivo=objetivo, filas_entrenamiento=len(y_train), filas_prueba=len(y_test))
            filas.append(fila)
        fila = evaluar_ensamble(info, X_train, X_test, y_train[objetivo], y_test[objetivo])
        fila.update(tabla=tabla, objetivo=objetivo, filas_entrenamiento=len(y_train), filas_prueba=len(y_test))
        filas.append(fila)

    pd.DataFrame(filas).to_csv(archivo, index=False)
    del X, y, X_train, X_test, y_train, y_test
    gc.collect()

pd.DataFrame([{'tabla': tabla, 'archivo': str((RESULTADOS / tabla / 'metricas_finales.csv').relative_to(BASE))}
              for tabla in tablas]).to_csv(RESULTADOS / 'archivos_generados.csv', index=False)
print(f'\nResultados guardados en: {RESULTADOS}')
