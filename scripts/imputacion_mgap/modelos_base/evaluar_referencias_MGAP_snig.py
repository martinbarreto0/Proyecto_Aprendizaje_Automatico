"""Evalúa el modelo elegido y dos referencias sobre el 20% aleatorio final."""
from time import perf_counter
import argparse
import json
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tabla', help='Por defecto evalúa las seis tablas.')
parser.add_argument('--hilos', type=int, default=4)
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

import pandas as pd
from configuracion_MGAP_snig import TABLAS, OBJETIVOS
from funciones_MGAP_snig import (cargar_tabla, codificar, calcular_pesos, crear_modelo,
                                 evaluar, separar_aleatorio, SALIDAS)

for tabla in ([args.tabla] if args.tabla else list(TABLAS)):
    df, X, y = cargar_tabla(tabla)
    elegidos_finales = []
    referencias = []
    carpeta = SALIDAS / tabla
    elegidos = json.loads((carpeta / 'modelos_elegidos.json').read_text(encoding='utf-8'))
    X_train, X_test, y_train, y_test = separar_aleatorio(X, y, semilla=42)
    X_train, X_test = codificar(X_train, X_test, tabla)
    etapa = 'prueba_aleatoria'
    for objetivo in OBJETIVOS:
        info = elegidos[objetivo]
        pruebas = [('Elegido', info['modelo'], info['exponente_pesos']),
                   ('Referencia', 'Mayoritaria', 0), ('Referencia', 'RF_balanceado', 1)]
        ya_evaluados = {}
        for rol, nombre, exponente in pruebas:
            clave = (nombre, exponente)
            if clave in ya_evaluados:
                fila = dict(ya_evaluados[clave])
            else:
                print(f'{tabla}: {etapa}, {objetivo}, {nombre}', flush=True)
                iteraciones = info['max_iter'] if nombre.startswith('HistBoost_') else 250
                modelo = crear_modelo(nombre, y_train[objetivo], args.hilos, iteraciones)
                inicio = perf_counter()
                modelo.fit(X_train, y_train[objetivo],
                           sample_weight=calcular_pesos(y_train[objetivo], exponente))
                pred = modelo.predict(X_test)
                fila = evaluar(tabla, etapa, nombre, objetivo, y_train[objetivo],
                               y_test[objetivo], pred, perf_counter()-inicio)
                ya_evaluados[clave] = dict(fila)
                del modelo
            fila['rol'] = rol
            (elegidos_finales if rol == 'Elegido' else referencias).append(fila)
            pd.DataFrame(elegidos_finales).to_csv(carpeta / 'evaluacion_elegidos_aleatorio.csv', index=False)
            pd.DataFrame(referencias).to_csv(carpeta / 'evaluacion_referencias.csv', index=False)
