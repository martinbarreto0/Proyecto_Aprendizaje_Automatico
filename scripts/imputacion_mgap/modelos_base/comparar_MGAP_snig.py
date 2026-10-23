"""Compara modelos con una división aleatoria, sin usar los años."""
from pathlib import Path
from time import perf_counter
import argparse
import gc
import hashlib
import json
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tabla', help='Comparar solo una tabla; por defecto compara las seis.')
parser.add_argument('--hilos', type=int, default=4)
parser.add_argument('--reanudar', action='store_true', help='Conservar candidatos ya evaluados.')
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

import pandas as pd
from configuracion_MGAP_snig import TABLAS, OBJETIVOS
from funciones_MGAP_snig import (cargar_tabla, codificar, calcular_pesos, crear_modelo,
                                 evaluar, separar_aleatorio, SALIDAS, SNIG)

candidatos = [('Mayoritaria', 0), ('RF_balanceado', 1), ('RF_suavizado', 0.5),
              ('ExtraTrees_suavizado', 0.5), ('HistBoost_suavizado', 0.5),
              ('HistBoost_leve', 0.25), ('HistBoost_sin_pesos', 0)]
tablas = [args.tabla] if args.tabla else list(TABLAS)
METODO = 'train_test_split_64_16_20_semillas_42_43_pareja_mgap_v1'

for tabla in tablas:
    print(f'\n=== {tabla} ===', flush=True)
    carpeta = SALIDAS / tabla
    huella = hashlib.sha256((SNIG / f'{tabla}.csv').read_bytes()).hexdigest()
    registro = carpeta / 'datos_comparacion_sha256.txt'
    registro_metodo = carpeta / 'metodo_comparacion.txt'
    if args.reanudar and (carpeta / 'comparacion.csv').exists():
        if not registro.exists() or registro.read_text(encoding='utf-8').strip() != huella:
            raise ValueError(f'{tabla}: la comparación guardada no corresponde al CSV actual. Ejecutar sin --reanudar.')
        if not registro_metodo.exists() or registro_metodo.read_text(encoding='utf-8').strip() != METODO:
            raise ValueError(f'{tabla}: la comparación guardada usa otra partición. Ejecutar sin --reanudar.')
    carpeta.mkdir(parents=True, exist_ok=True)
    registro.write_text(huella, encoding='utf-8')
    registro_metodo.write_text(METODO, encoding='utf-8')
    df, X, y = cargar_tabla(tabla)
    # 20% queda intacto para la prueba final. Dentro del 80% restante se reserva
    # otro 20% para elegir el modelo: 64% ajuste, 16% validación y 20% prueba.
    X_desarrollo, _, y_desarrollo, _ = separar_aleatorio(X, y, semilla=42)
    X_ajuste, X_val, y_ajuste, y_val = separar_aleatorio(X_desarrollo, y_desarrollo, semilla=43)
    X_ajuste, X_val = codificar(X_ajuste, X_val, tabla)
    resultados = (pd.read_csv(carpeta / 'comparacion.csv', keep_default_na=False).to_dict('records')
                  if args.reanudar and (carpeta / 'comparacion.csv').exists() else [])
    for fila in resultados:
        fila.setdefault('max_iter', 250 if fila['modelo'].startswith('HistBoost_') else 0)
    elegidos = {}
    carpeta.mkdir(parents=True, exist_ok=True)

    for objetivo in OBJETIVOS:
        for nombre, exponente in candidatos:
            # En tablas grandes, tipo de producción compara una sola variante de boosting.
            if tabla == 'DatosAnimalesDetallados' and objetivo == OBJETIVOS[1] \
                    and nombre in ['HistBoost_leve', 'HistBoost_sin_pesos']:
                continue
            if any(r['objetivo'] == objetivo and r['modelo'] == nombre for r in resultados):
                continue
            print(f'Entrenando {objetivo}: {nombre}', flush=True)
            iteraciones = 100 if tabla == 'DatosAnimalesDetallados' else 250
            modelo = crear_modelo(nombre, y_ajuste[objetivo], args.hilos, iteraciones)
            inicio = perf_counter()
            modelo.fit(X_ajuste, y_ajuste[objetivo],
                       sample_weight=calcular_pesos(y_ajuste[objetivo], exponente))
            pred = modelo.predict(X_val)
            metricas = evaluar(tabla, 'validacion_aleatoria', nombre, objetivo,
                y_ajuste[objetivo], y_val[objetivo], pred, perf_counter()-inicio)
            metricas['exponente_pesos'] = exponente
            metricas['iteraciones'] = int(getattr(modelo, 'n_iter_', getattr(modelo, 'n_estimators', 0)))
            metricas['max_iter'] = int(getattr(modelo, 'max_iter', 0))
            resultados.append(metricas)
            pd.DataFrame(resultados).to_csv(carpeta / 'comparacion.csv', index=False)
            # Liberar cada bosque antes de entrenar el siguiente, especialmente en Detallados.
            del modelo
            gc.collect()

        opciones = [r for r in resultados if r['objetivo'] == objetivo and r['modelo'] != 'Mayoritaria']
        mejor = max(opciones, key=lambda r: (r['f1_macro'], r['accuracy']))
        elegidos[objetivo] = dict(modelo=mejor['modelo'], exponente_pesos=mejor['exponente_pesos'],
                                 f1_macro_validacion_aleatoria=mejor['f1_macro'], max_iter=mejor['max_iter'])
    pd.DataFrame(resultados).to_csv(carpeta / 'comparacion.csv', index=False)
    (carpeta / 'modelos_elegidos.json').write_text(json.dumps(elegidos, indent=2), encoding='utf-8')
    print('ELEGIDOS:', tabla, elegidos, flush=True)
    del df, X, y, X_desarrollo, y_desarrollo, X_ajuste, X_val, y_ajuste, y_val
    gc.collect()
