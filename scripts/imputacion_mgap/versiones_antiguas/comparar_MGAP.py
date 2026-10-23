"""Comparación simple. Selecciona en 2023 y reserva 2022 para la prueba final."""
from pathlib import Path
from time import perf_counter
import argparse
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE = Path(__file__).resolve().parent
RUTA = BASE.parents[2] / 'data/processed/SNIG/DatosAnimales.csv'
SALIDAS = BASE / 'resultados_mgap'
SALIDAS.mkdir(exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ajustar-pesos', action='store_true', help='Probar solo dos variantes de pesos para tipo.')
args = parser.parse_args()
archivo_metricas = 'ajuste_pesos.csv' if args.ajustar_pesos else 'comparacion.csv'
df = pd.read_csv(RUTA, sep=';', na_values=['-', ' ', 'NULL', 'null'])
objetivos = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']
df = df.dropna(subset=objetivos).copy()
cat_originales = ['GiroCodigo', 'EspecieCodigo']
cat_ampliadas = cat_originales + ['DepartamentoCodigo', 'NaturalezaJuridicaCodigo']
numericas = ['EstratoCodigo', 'AnimalesPorEspecieConsumoPD_AD',
             'AnimalesPorEspecieNacimientosPD_AD', 'AnimalesPorEspecieMortandadPD_AD',
             'CantidadTenedoresPorEspecie', 'AnimalesPorEspeciePD_AD', 'AnimalesPorEspeciePD_PF']
for col in cat_ampliadas:
    df[col] = df[col].fillna('SIN_DATO').astype(str)
for col in numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

X_original = pd.get_dummies(df[cat_originales + numericas], columns=cat_originales, drop_first=True)
X_ampliada = df[cat_ampliadas + numericas].copy()
# Denominador al menos 1 para evitar divisiones por cero.
stock = X_ampliada['AnimalesPorEspeciePD_AD'].clip(lower=1)
for col in ['AnimalesPorEspecieConsumoPD_AD', 'AnimalesPorEspecieNacimientosPD_AD',
            'AnimalesPorEspecieMortandadPD_AD', 'AnimalesPorEspeciePD_PF']:
    X_ampliada[col + '_Proporcion'] = X_ampliada[col] / stock
X_ampliada['AnimalesPorTenedor'] = stock / X_ampliada['CantidadTenedoresPorEspecie'].clip(lower=1)
X_dummy = pd.get_dummies(X_ampliada, columns=cat_ampliadas)
entrenar = df['Anio'].isin([2024, 2025])
validar = df['Anio'].eq(2023)
resultados = []

def evaluar(nombre, objetivo, real, pred, segundos, etapa):
    fila = dict(etapa=etapa, modelo=nombre, objetivo=objetivo,
                accuracy=accuracy_score(real, pred),
                f1_macro=f1_score(real, pred, average='macro', zero_division=0),
                f1_weighted=f1_score(real, pred, average='weighted', zero_division=0),
                segundos=segundos)
    resultados.append(fila)
    print(fila, flush=True)
    reporte = classification_report(real, pred, digits=4, zero_division=0)
    (SALIDAS / f'{etapa}_{nombre}_{objetivo}.txt').write_text(reporte, encoding='utf-8')
    pd.DataFrame(resultados).to_csv(SALIDAS / archivo_metricas, index=False)

# Misma validación para todos. F1 macro pesa igual a cada clase.
for objetivo in objetivos:
    if args.ajustar_pesos and objetivo != 'TipoProduccionMGAPCodigo':
        continue
    y = df[objetivo].astype(int)
    conteos = y[entrenar].value_counts()
    pesos = {clase: float((len(y[entrenar]) / (len(conteos) * n)) ** 0.5)
             for clase, n in conteos.items()}
    candidatos = [
        ('RF_original', RandomForestClassifier(n_estimators=150, max_depth=20,
          min_samples_leaf=3, class_weight=pesos, random_state=42, n_jobs=4), X_original),
        ('ExtraTrees_ampliado', ExtraTreesClassifier(n_estimators=150, min_samples_leaf=3,
          class_weight=pesos, random_state=42, n_jobs=4), X_dummy),
        ('HistBoost_ampliado', HistGradientBoostingClassifier(max_iter=250,
          max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1,
          random_state=42), X_dummy),
        ('CatBoost_original', CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
          class_weights=pesos, cat_features=cat_originales, random_seed=42,
          thread_count=4, verbose=False, allow_writing_files=False), df[cat_originales + numericas]),
        ('CatBoost_ampliado', CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
          class_weights=pesos, cat_features=cat_ampliadas, random_seed=42,
          thread_count=4, verbose=False, allow_writing_files=False), X_ampliada),
    ]
    if objetivo == 'TipoProduccionMGAPCodigo':
        candidatos += [
            ('HistBoost_sin_pesos', HistGradientBoostingClassifier(max_iter=250,
             max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1, random_state=42), X_dummy),
            ('HistBoost_pesos_leves', HistGradientBoostingClassifier(max_iter=250,
             max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1, random_state=42), X_dummy),
        ]
    for nombre, modelo, X in candidatos:
        if args.ajustar_pesos and nombre not in ['HistBoost_sin_pesos', 'HistBoost_pesos_leves']:
            continue
        print(f'Entrenando {nombre}: {objetivo}', flush=True)
        inicio = perf_counter()
        if nombre == 'HistBoost_ampliado':
            modelo.fit(X[entrenar], y[entrenar], sample_weight=y[entrenar].map(pesos))
        elif nombre == 'HistBoost_pesos_leves':
            pesos_leves = {clase: peso**0.5 for clase, peso in pesos.items()}
            modelo.fit(X[entrenar], y[entrenar], sample_weight=y[entrenar].map(pesos_leves))
        else:
            modelo.fit(X[entrenar], y[entrenar])
        pred = np.asarray(modelo.predict(X[validar])).ravel()
        evaluar(nombre, objetivo, y[validar], pred, perf_counter()-inicio, 'validacion_2023')

print('\nGanadores por F1 macro (sin usar 2022):', flush=True)
tabla = pd.DataFrame(resultados)
print(tabla.loc[tabla.groupby('objetivo')['f1_macro'].idxmax()].to_string(index=False), flush=True)
