"""Prueba temporal: entrenar con 2022–2024 y medir en 2025, sin filas gemelas en la medición más estricta.

La división aleatoria deja el mismo establecimiento (en distintos años) en
entrenamiento y prueba. Aquí se mide cada base y cada ensamble en 2025 sobre:
- todas: todas las filas de 2025;
- con_gemelo: filas de 2025 con una fila idéntica (mismas variables) en 2022–2024;
- sin_gemelo: el resto, la medida más estricta.
Solo se leen los CSV: no se modifica ningún dataset.
"""
import argparse
import gc
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tabla', help='Comparar solo una tabla; por defecto las siete.')
parser.add_argument('--hilos', type=int, default=4)
parser.add_argument('--max-knn', type=int, default=200_000)
parser.add_argument('--anio-prueba', type=int, default=2025)
parser.add_argument('--sin-stacking', action='store_true')
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from funciones_ensamble_MGAP_snig import (fm, BASE, BASES, ENSAMBLES, variante_histboost,
    probabilidades_bases, probabilidades_oof, entrenar_meta, puntajes_ensamble)
from configuracion_MGAP_snig import TABLAS, OBJETIVOS

SALIDAS = BASE / 'resultados_por_anio'
fm.SALIDAS = SALIDAS  # los reportes por clase de evaluar() quedan en esta carpeta


def cargar_anios(tabla, indice):
    # Misma lectura y mismo filtro que cargar_tabla(), solo para recuperar el año de cada fila.
    df = pd.read_csv(fm.SNIG / f'{tabla}.csv', sep=';', usecols=['Anio'] + OBJETIVOS,
                     na_values=['-', ' ', 'NULL', 'null'], low_memory=False)
    anios = df.dropna(subset=OBJETIVOS).Anio.astype(int)
    assert anios.index.equals(indice), f'{tabla}: el índice no coincide con cargar_tabla()'
    return anios


def metricas(real, pred):
    etiquetas = sorted(real.unique())
    return dict(accuracy=accuracy_score(real, pred), filas=len(real),
                f1_macro=f1_score(real, pred, labels=etiquetas, average='macro', zero_division=0))


ensambles = [e for e in ENSAMBLES if not (args.sin_stacking and e == 'E4_stacking')]

for tabla in ([args.tabla] if args.tabla else list(TABLAS)):
    print(f'\n=== {tabla} ===', flush=True)
    carpeta = SALIDAS / tabla
    carpeta.mkdir(parents=True, exist_ok=True)
    df, X, y = fm.cargar_tabla(tabla)
    anios = cargar_anios(tabla, X.index)
    del df
    prueba = anios.eq(args.anio_prueba)
    X_train, X_test, y_train, y_test = X[~prueba], X[prueba], y[~prueba], y[prueba]
    anios_train = anios[~prueba].to_numpy()

    # Fila gemela: las mismas variables de entrada (antes de codificar) en algún año de entrenamiento.
    clave = lambda d: pd.util.hash_pandas_object(d.astype(str), index=False)
    gemelo = clave(X_test).isin(set(clave(X_train))).to_numpy()
    subconjuntos = {'todas': slice(None), 'con_gemelo': gemelo, 'sin_gemelo': ~gemelo}
    print(f'{prueba.sum()} filas de prueba; {gemelo.mean():.1%} con gemelo en entrenamiento', flush=True)
    del X, y
    X_train, X_test = fm.codificar(X_train, X_test, tabla)

    filas = []
    for objetivo in OBJETIVOS:
        print(f'Objetivo {objetivo}', flush=True)
        hb = variante_histboost(tabla, objetivo)
        y_fit, real = y_train[objetivo], y_test[objetivo]
        probs, clases, segundos = probabilidades_bases(X_train, y_fit, X_test, hb, args.hilos, args.max_knn)
        predicciones = {nombre: clases[probs[nombre].argmax(axis=1)] for nombre in BASES}
        for nombre in ensambles:
            meta = None
            if nombre == 'E4_stacking':
                # Los pliegues dejan afuera un año entero, igual que la prueba.
                oof, _, _ = probabilidades_oof(X_train, y_fit, hb, args.hilos, args.max_knn, grupos=anios_train)
                meta = entrenar_meta(oof, y_fit)
            predicciones[nombre] = clases[puntajes_ensamble(nombre, probs, meta).argmax(axis=1)]
        for nombre, pred in predicciones.items():
            rol = 'Base' if nombre in BASES else 'Ensamble'
            fm.evaluar(tabla, f'prueba_{args.anio_prueba}', nombre, objetivo, y_fit, real, pred,
                       segundos.get(nombre, 0))
            for sub, mascara in subconjuntos.items():
                filas.append(dict(tabla=tabla, objetivo=objetivo, modelo=nombre, rol=rol,
                                  histboost=hb[0], subconjunto=sub,
                                  **metricas(real[mascara], pred[mascara])))
        pd.DataFrame(filas).to_csv(carpeta / 'evaluacion_por_anio.csv', index=False)
        gc.collect()
    del X_train, X_test, y_train, y_test
    gc.collect()
