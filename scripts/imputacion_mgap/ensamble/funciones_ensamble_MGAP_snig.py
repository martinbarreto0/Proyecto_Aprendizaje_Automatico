"""Funciones para combinar los modelos ya usados en la imputación MGAP.

No se agregan modelos nuevos: se combinan KNN (primera versión) con los árboles
de la versión actual (Random Forest, Extra Trees y HistGradientBoosting).
"""
import gc
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'modelos_base'))

import funciones_MGAP_snig as fm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ANTERIORES = BASE.parent / 'resultados_mgap_snig'
SALIDAS = BASE / 'resultados_ensamble'
# evaluar() guarda sus reportes en fm.SALIDAS: se redirige para no tocar los resultados anteriores.
fm.SALIDAS = SALIDAS

BASES = ['KNN', 'RF_suavizado', 'ExtraTrees_suavizado', 'HistBoost']
ENSAMBLES = {
    'E1_voto_duro': BASES,
    'E2_voto_blando': BASES,
    'E3_voto_blando_arboles': ['RF_suavizado', 'ExtraTrees_suavizado', 'HistBoost'],
    'E4_stacking': BASES,
}


def variante_histboost(tabla, objetivo):
    """La variante de HistBoost con mejor F1 macro en la comparación anterior."""
    comparacion = pd.read_csv(ANTERIORES / tabla / 'comparacion.csv')
    hb = comparacion[comparacion.objetivo.eq(objetivo) & comparacion.modelo.str.startswith('HistBoost_')]
    mejor = hb.sort_values(['f1_macro', 'accuracy'], ascending=False).iloc[0]
    return mejor.modelo, float(mejor.exponente_pesos), int(mejor.max_iter)


def muestra_por_clase(y, maximo):
    # KNN es lento para predecir en tablas grandes: se entrena con una muestra
    # proporcional que conserva al menos una fila de cada clase.
    if len(y) <= maximo:
        return y.index
    fraccion = maximo / len(y)
    return y.groupby(y, group_keys=False).apply(
        lambda s: s.sample(max(1, round(len(s) * fraccion)), random_state=42)).index


def entrenar_base(nombre, X, y, hb, hilos, max_knn):
    if nombre == 'KNN':
        # Igual que la primera versión: escalado y 5 vecinos ponderados por distancia.
        filas = muestra_por_clase(y, max_knn)
        modelo = make_pipeline(StandardScaler(),
                               KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=hilos))
        return modelo.fit(X.loc[filas], y.loc[filas])
    if nombre == 'HistBoost':
        nombre_hb, exponente, iteraciones = hb
        modelo = fm.crear_modelo(nombre_hb, y, hilos, iteraciones)
    else:
        modelo, exponente = fm.crear_modelo(nombre, y, hilos), 0.5
    return modelo.fit(X, y, sample_weight=fm.calcular_pesos(y, exponente))


def probabilidades_bases(X_fit, y_fit, X_eval, hb, hilos, max_knn):
    """Entrena cada base una sola vez y devuelve sus probabilidades sobre X_eval."""
    probs, segundos, clases = {}, {}, None
    for nombre in BASES:
        print(f'  base {nombre} ({len(y_fit)} filas)', flush=True)
        inicio = perf_counter()
        modelo = entrenar_base(nombre, X_fit, y_fit, hb, hilos, max_knn)
        probs[nombre] = modelo.predict_proba(X_eval)
        segundos[nombre] = perf_counter() - inicio
        if clases is None:
            clases = modelo.classes_
        assert np.array_equal(clases, modelo.classes_), f'{nombre}: clases distintas'
        del modelo
        gc.collect()
    return probs, clases, segundos


def voto_blando(probs, miembros):
    return np.mean([probs[m] for m in miembros], axis=0)


def voto_duro(probs, miembros):
    k = probs[miembros[0]].shape[1]
    votos = sum(np.eye(k)[probs[m].argmax(axis=1)] for m in miembros)
    # Los empates se resuelven con el promedio de probabilidades.
    return votos + 1e-6 * voto_blando(probs, miembros)


def probabilidades_oof(X, y, hb, hilos, max_knn, pliegues=3, grupos=None):
    """Probabilidades fuera de pliegue para entrenar el stacking sin fuga de información.

    Con `grupos` (por ejemplo, el año) cada pliegue deja afuera un grupo completo.
    """
    clases = np.sort(y.unique())
    oof = {nombre: np.zeros((len(y), len(clases))) for nombre in BASES}
    inicio = perf_counter()
    if grupos is None:
        divisor = StratifiedKFold(pliegues, shuffle=True, random_state=42).split(X, y)
    else:
        divisor = LeaveOneGroupOut().split(X, y, grupos)
        pliegues = len(np.unique(grupos))
    for numero, (tr, va) in enumerate(divisor, start=1):
        print(f' pliegue {numero}/{pliegues}', flush=True)
        probs, clases_pliegue, _ = probabilidades_bases(X.iloc[tr], y.iloc[tr], X.iloc[va], hb, hilos, max_knn)
        # Una clase muy rara puede faltar en un pliegue: se ubica cada columna en su clase.
        columnas = np.searchsorted(clases, clases_pliegue)
        for nombre in BASES:
            oof[nombre][np.ix_(va, columnas)] = probs[nombre]
    return oof, clases, perf_counter() - inicio


def entrenar_meta(oof, y):
    Z = np.hstack([oof[m] for m in BASES])
    meta = LogisticRegression(max_iter=1000)
    return meta.fit(Z, y, sample_weight=fm.calcular_pesos(y, 0.5))


def puntajes_ensamble(nombre, probs, meta=None):
    miembros = ENSAMBLES[nombre]
    if nombre == 'E1_voto_duro':
        return voto_duro(probs, miembros)
    if nombre == 'E4_stacking':
        return meta.predict_proba(np.hstack([probs[m] for m in miembros]))
    return voto_blando(probs, miembros)
