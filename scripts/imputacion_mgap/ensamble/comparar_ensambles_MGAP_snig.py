"""Compara ensambles de KNN + árboles con la misma división 64/16/20 del pipeline vigente.

1. Validación: bases entrenadas en el 64% y medidas en el 16%; se elige el ensamble
   con mayor F1 macro.
2. Prueba: bases reentrenadas en el 80% y medidas una sola vez en el mismo 20% final
   que usaron los modelos elegidos anteriormente.
"""
import argparse
import gc
import json
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tabla', help='Comparar solo una tabla; por defecto las siete.')
parser.add_argument('--hilos', type=int, default=4)
parser.add_argument('--max-knn', type=int, default=200_000,
                    help='Máximo de filas de entrenamiento para KNN.')
parser.add_argument('--sin-stacking', action='store_true',
                    help='Omitir E4 (recomendado en DatosAnimalesDetallados por el tiempo).')
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

import pandas as pd
from funciones_ensamble_MGAP_snig import (fm, BASES, ENSAMBLES, SALIDAS, variante_histboost,
    probabilidades_bases, probabilidades_oof, entrenar_meta, puntajes_ensamble)
from configuracion_MGAP_snig import TABLAS, OBJETIVOS


def registrar(filas, tabla, etapa, nombre, rol, objetivo, y_fit, real, pred, segundos, hb):
    fila = fm.evaluar(tabla, etapa, nombre, objetivo, y_fit, real, pred, segundos)
    fila.update(rol=rol, miembros=' + '.join(ENSAMBLES.get(nombre, [nombre])), histboost=hb[0])
    filas.append(fila)
    return fila


def evaluar_etapa(tabla, etapa, X_fit, y_fit, X_eval, y_eval, objetivo, hb, ensambles, filas):
    """Entrena las bases una vez y evalúa bases y ensambles con las mismas probabilidades."""
    probs, clases, segundos = probabilidades_bases(X_fit, y_fit, X_eval, hb, args.hilos, args.max_knn)
    for nombre in BASES:
        registrar(filas, tabla, etapa, nombre, 'Base', objetivo, y_fit, y_eval,
                  clases[probs[nombre].argmax(axis=1)], segundos[nombre], hb)
    for nombre in ensambles:
        meta, extra = None, 0
        if nombre == 'E4_stacking':
            oof, _, extra = probabilidades_oof(X_fit, y_fit, hb, args.hilos, args.max_knn)
            meta = entrenar_meta(oof, y_fit)
        pred = clases[puntajes_ensamble(nombre, probs, meta).argmax(axis=1)]
        tiempo = sum(segundos[m] for m in ENSAMBLES[nombre]) + extra
        registrar(filas, tabla, etapa, nombre, 'Ensamble', objetivo, y_fit, y_eval, pred, tiempo, hb)


ensambles_validacion = [e for e in ENSAMBLES if not (args.sin_stacking and e == 'E4_stacking')]

for tabla in ([args.tabla] if args.tabla else list(TABLAS)):
    print(f'\n=== {tabla} ===', flush=True)
    carpeta = SALIDAS / tabla
    carpeta.mkdir(parents=True, exist_ok=True)
    df, X, y = fm.cargar_tabla(tabla)
    X_dev, X_test, y_dev, y_test = fm.separar_aleatorio(X, y, semilla=42)
    del df, X, y

    # Validación: 64% ajuste / 16% validación, igual que comparar_MGAP_snig.py.
    X_aj, X_val, y_aj, y_val = fm.separar_aleatorio(X_dev, y_dev, semilla=43)
    X_aj, X_val = fm.codificar(X_aj, X_val, tabla)
    validacion, elegidos = [], {}
    for objetivo in OBJETIVOS:
        print(f'Validación {objetivo}', flush=True)
        hb = variante_histboost(tabla, objetivo)
        evaluar_etapa(tabla, 'validacion_ensamble', X_aj, y_aj[objetivo], X_val, y_val[objetivo],
                      objetivo, hb, ensambles_validacion, validacion)
        opciones = [f for f in validacion if f['objetivo'] == objetivo and f['rol'] == 'Ensamble']
        mejor = max(opciones, key=lambda f: (f['f1_macro'], f['accuracy']))
        elegidos[objetivo] = dict(ensamble=mejor['modelo'], miembros=ENSAMBLES[mejor['modelo']],
                                  histboost=list(hb), f1_macro_validacion=mejor['f1_macro'])
        pd.DataFrame(validacion).to_csv(carpeta / 'comparacion_ensambles.csv', index=False)
    (carpeta / 'ensambles_elegidos.json').write_text(json.dumps(elegidos, indent=2), encoding='utf-8')
    print('ELEGIDOS:', tabla, elegidos, flush=True)
    del X_aj, X_val, y_aj, y_val
    gc.collect()

    # Prueba: 80% / 20%, solo con el ensamble elegido en validación.
    X_dev, X_test = fm.codificar(X_dev, X_test, tabla)
    prueba = []
    for objetivo in OBJETIVOS:
        print(f'Prueba {objetivo}', flush=True)
        info = elegidos[objetivo]
        evaluar_etapa(tabla, 'prueba_ensamble', X_dev, y_dev[objetivo], X_test, y_test[objetivo],
                      objetivo, tuple(info['histboost']), [info['ensamble']], prueba)
        pd.DataFrame(prueba).to_csv(carpeta / 'evaluacion_ensambles.csv', index=False)
    del X_dev, X_test, y_dev, y_test
    gc.collect()
