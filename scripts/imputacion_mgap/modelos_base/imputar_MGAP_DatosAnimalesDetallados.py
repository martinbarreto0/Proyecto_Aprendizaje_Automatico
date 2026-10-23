"""Evalúa los códigos MGAP de DatosAnimalesDetallados con un train_test_split 80/20."""
from pathlib import Path
from time import perf_counter
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--hilos', type=int, default=4)
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE = Path(__file__).resolve().parent
TABLA = 'DatosAnimalesDetallados'
RUTA = BASE.parents[3] / 'data/processed/SNIG' / f'{TABLA}.csv'
SALIDAS = BASE.parent / 'resultados_mgap_snig' / TABLA
SALIDAS.mkdir(parents=True, exist_ok=True)
objetivos = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']
cat = ['GiroCodigo', 'DepartamentoCodigo', 'NaturalezaJuridicaCodigo', 'EspecieCodigo', 'CategoriaCodigo']
num = ['EstratoCodigo', 'AnimalesPorCategoriaPD_AD', 'AnimalesPorCategoriaPD_PF']
proporciones = {'RelacionPF_AD': ['AnimalesPorCategoriaPD_PF', 'AnimalesPorCategoriaPD_AD']}
nombres = {'EspecializacionMGAPCodigo': 'HistBoost_suavizado',
 'TipoProduccionMGAPCodigo': 'HistBoost_suavizado'}
exponentes = {'EspecializacionMGAPCodigo': 0.5, 'TipoProduccionMGAPCodigo': 0.5}
max_iteraciones = {'EspecializacionMGAPCodigo': 100, 'TipoProduccionMGAPCodigo': 100}

# 1. Cargar solamente las columnas de esta tabla. El año no se lee ni se usa.
df = pd.read_csv(RUTA, sep=';', usecols=cat + num + objetivos,
                 na_values=['-', ' ', 'NULL', 'null'], low_memory=False)
df = df.dropna(subset=objetivos).copy()
for col in cat:
    df[col] = df[col].fillna('SIN_DATO').astype(str)
for col in num:
    convertido = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    if (df[col].notna() & convertido.isna()).any():
        raise ValueError(f'Hay valores no numéricos en {col}')
    df[col] = convertido.fillna(0)

# 2. Crear las proporciones que permite esta tabla.
X = df[cat + num].copy()
for nueva, (numerador, denominador) in proporciones.items():
    X[nueva] = X[numerador] / X[denominador].clip(lower=1)
y = df[objetivos].astype(int)

# 3. Mezclar y separar 80% para entrenamiento y 20% para prueba.
# Se estratifica por la pareja de códigos; las parejas con una sola fila quedan en entrenamiento.
estrato = y.astype(str).agg('|'.join, axis=1)
conteos_pareja = estrato.value_counts()
raras = estrato.map(conteos_pareja).lt(2)
train, test = train_test_split(y.index[~raras], test_size=0.2, random_state=42,
                              stratify=estrato.loc[~raras])
train = train.append(y.index[raras])
X_train, X_test = X.loc[train], X.loc[test]
y_train, y_test = y.loc[train], y.loc[test]
X_train = pd.get_dummies(X_train, columns=cat)
X_test = pd.get_dummies(X_test, columns=cat).reindex(columns=X_train.columns, fill_value=0)
print(f'{TABLA}: entrenamiento={len(y_train)}, prueba={len(y_test)}.', flush=True)

# 4. Entrenar el modelo elegido para cada objetivo.
resultados = []
for objetivo in objetivos:
    nombre = nombres[objetivo]
    conteos = y_train[objetivo].value_counts()
    pesos_clase = (len(y_train) / (len(conteos) * conteos)) ** exponentes[objetivo]
    if nombre.startswith('RF_'):
        modelo = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=3,
                                       random_state=42, n_jobs=args.hilos)
    elif nombre.startswith('ExtraTrees_'):
        modelo = ExtraTreesClassifier(n_estimators=100, max_depth=25, min_samples_leaf=3,
                                     random_state=42, n_jobs=args.hilos)
    else:
        detener = len(y_train) > 10_000 and conteos.min() >= 2
        modelo = HistGradientBoostingClassifier(max_iter=max_iteraciones[objetivo], max_leaf_nodes=31,
            learning_rate=0.1, l2_regularization=1, early_stopping=detener, random_state=42)
    print(f'Entrenando {objetivo}: {nombre}', flush=True)
    inicio = perf_counter()
    modelo.fit(X_train, y_train[objetivo],
               sample_weight=y_train[objetivo].map(pesos_clase).to_numpy())
    pred = modelo.predict(X_test)
    etiquetas = sorted(y_test[objetivo].unique())
    fila = dict(tabla=TABLA, etapa='prueba_aleatoria', modelo=nombre, rol='Elegido', objetivo=objetivo,
        accuracy=accuracy_score(y_test[objetivo], pred),
        f1_macro=f1_score(y_test[objetivo], pred, labels=etiquetas, average='macro', zero_division=0),
        f1_weighted=f1_score(y_test[objetivo], pred, labels=etiquetas, average='weighted', zero_division=0),
        filas_train=len(y_train), filas_prueba=len(y_test), segundos=perf_counter()-inicio,
        clases_prueba=' '.join(map(str, etiquetas)), clases_sin_entrenamiento='')
    reporte = (f"Acierto: {fila['accuracy']:.4f}\nF1 macro: {fila['f1_macro']:.4f}\n\n"
               + classification_report(y_test[objetivo], pred, labels=etiquetas,
                                       digits=4, zero_division=0))
    print(f'\n--- {objetivo} ---\n{reporte}', flush=True)
    (SALIDAS / f'prueba_aleatoria_elegido_{objetivo}.txt').write_text(reporte, encoding='utf-8')
    resultados.append(fila)
    pd.DataFrame(resultados).to_csv(SALIDAS / 'evaluacion_elegidos_aleatorio.csv', index=False)
