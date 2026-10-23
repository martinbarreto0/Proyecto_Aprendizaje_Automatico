"""Compara en el test reservado el mejor ajuste contra el modelo MGAP actual."""
# Utilidades para rutas, medición de tiempos, argumentos y liberación de memoria.
from pathlib import Path
from time import perf_counter
import argparse
import gc
import json
import os
import sys

# Directorio de resultados y directorio con módulos MGAP compartidos.
BASE = Path(__file__).resolve().parent
PADRE = BASE.parent / 'modelos_base'
sys.path.insert(0, str(PADRE))

# Opciones para limitar la evaluación a una tabla u objetivo y fijar paralelismo.
parser = argparse.ArgumentParser(description=__doc__)
# Restringe la evaluación final a una tabla.
parser.add_argument('--tabla', help='Evaluar una sola tabla.')
# Restringe la evaluación final a un objetivo.
parser.add_argument('--objetivo', help='Evaluar un solo objetivo MGAP.')
# Fija el paralelismo de los clasificadores.
parser.add_argument('--hilos', type=int, default=4)
# Lee las opciones recibidas y limita los hilos de las bibliotecas numéricas.
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

# Pandas guarda resultados; Scikit-learn construye y mide los clasificadores.
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from configuracion_MGAP_snig import OBJETIVOS, TABLAS
# Carga datos, separa en forma estratificada, codifica categorías y calcula
# pesos por fila para compensar las clases poco frecuentes.
from funciones_MGAP_snig import calcular_pesos, cargar_tabla, codificar, separar_aleatorio

# Ubicación donde la búsqueda dejó `mejores_parametros.json`.
RESULTADOS = BASE / 'resultados'


def familia(nombre):
    """Obtiene la familia de Scikit-learn a partir del prefijo del modelo."""
    if nombre.startswith('RF_'):
        return 'RF'
    if nombre.startswith('ExtraTrees_'):
        return 'ExtraTrees'
    if nombre.startswith('HistBoost_'):
        return 'HistBoost'
    raise ValueError(f'Modelo no ajustable: {nombre}')


def crear_modelo(nombre_familia, parametros, y_entrenamiento):
    """Crea el mismo tipo de clasificador y parámetros usados en la búsqueda."""
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
        # Activa parada temprana solo cuando la distribución permite reservar validación interna.
        detener = len(y_entrenamiento) > 10_000 and y_entrenamiento.value_counts().min() >= 2
        return HistGradientBoostingClassifier(random_state=42,
            max_iter=parametros['max_iter'], learning_rate=parametros['learning_rate'],
            max_leaf_nodes=parametros['max_leaf_nodes'],
            min_samples_leaf=parametros['min_samples_leaf'],
            l2_regularization=parametros['l2_regularization'], early_stopping=detener)
    raise ValueError(f'Familia desconocida: {nombre_familia}')


def evaluar(nombre, nombre_familia, parametros, X_train, X_test, y_train, y_test, objetivo):
    """Entrena una configuración y devuelve sus métricas en el test reservado."""
    modelo = crear_modelo(nombre_familia, parametros, y_train[objetivo])
    # Cronometra exclusivamente el ajuste y la predicción de este modelo.
    inicio = perf_counter()
    # Compensa el desbalance de clases según el exponente elegido.
    modelo.fit(X_train, y_train[objetivo],
               sample_weight=calcular_pesos(y_train[objetivo], parametros['exponente_pesos']))
    pred = modelo.predict(X_test)
    # Fija explícitamente las clases presentes al calcular F1 macro.
    etiquetas = sorted(y_test[objetivo].unique())
    fila = {
        'objetivo': objetivo,
        'modelo': nombre,
        'familia': nombre_familia,
        'parametros': json.dumps(parametros, sort_keys=True),
        'accuracy': accuracy_score(y_test[objetivo], pred),
        'f1_macro': f1_score(y_test[objetivo], pred, labels=etiquetas, average='macro', zero_division=0),
        'filas_entrenamiento': len(y_train),
        'filas_prueba': len(y_test),
        'segundos': perf_counter() - inicio,
    }
    # Libera el estimador antes de evaluar el siguiente objetivo.
    del modelo
    gc.collect()
    return fila


# Usa una tabla concreta si se indicó; de lo contrario evalúa todas las configuradas.
tablas = [args.tabla] if args.tabla else list(TABLAS)
if args.tabla and args.tabla not in TABLAS:
    raise ValueError(f'Tabla desconocida: {args.tabla}')
if args.objetivo and args.objetivo not in OBJETIVOS:
    raise ValueError(f'Objetivo desconocido: {args.objetivo}')

for tabla in tablas:
    # Cada tabla tiene su propio JSON generado durante la búsqueda.
    carpeta = RESULTADOS / tabla
    # Carga modelo actual, modelo elegido y parámetros de cada objetivo.
    seleccion = json.loads((carpeta / 'mejores_parametros.json').read_text(encoding='utf-8'))
    objetivos = [args.objetivo] if args.objetivo else list(seleccion['objetivos'])
    # Reconstruye los mismos datos originales y la misma división 80/20 fija.
    df, X, y = cargar_tabla(tabla)
    X_train, X_test, y_train, y_test = separar_aleatorio(X, y, semilla=42)
    X_train, X_test = codificar(X_train, X_test, tabla)
    filas = []
    decisiones = {}

    for objetivo in objetivos:
        # Información que la búsqueda guardó para este objetivo.
        info = seleccion['objetivos'][objetivo]
        print(f'\n=== {tabla} / {objetivo} ===', flush=True)
        # Mide la línea de base y la configuración propuesta en idéntico test.
        actual = evaluar(info['modelo_actual'], familia(info['modelo_actual']),
                          info['parametros_actuales'], X_train, X_test, y_train, y_test, objetivo)
        elegido = evaluar('Busqueda_' + info['familia_elegida'], info['familia_elegida'],
                          info['parametros_elegidos'], X_train, X_test, y_train, y_test, objetivo)
        actual['rol'] = 'actual'
        elegido['rol'] = 'busqueda'
        # Diferencia positiva significa que la búsqueda superó la base en datos inéditos.
        delta = elegido['f1_macro'] - actual['f1_macro']
        decisiones[objetivo] = {
            'f1_macro_actual_test': actual['f1_macro'],
            'f1_macro_busqueda_test': elegido['f1_macro'],
            'delta_f1_macro_test': delta,
            # Exige tanto la evidencia interna como una mejora real en el test reservado.
            'recomienda_reemplazo': bool(info['recomienda_reemplazo'] and delta > 0),
        }
        filas.extend([actual, elegido])
        print(json.dumps(decisiones[objetivo], ensure_ascii=False, indent=2), flush=True)

    # Exporta las métricas de ambos modelos para auditoría.
    pd.DataFrame(filas).to_csv(carpeta / 'evaluacion_test_hiperparametros.csv', index=False)
    # Exporta la decisión resumida que indica si corresponde reemplazar el modelo.
    (carpeta / 'decision_final_test.json').write_text(
        json.dumps({'tabla': tabla, 'objetivos': decisiones}, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    del df, X, y, X_train, X_test, y_train, y_test
    gc.collect()
