"""Funciones pequeñas compartidas por la comparación de tablas SNIG."""
from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from configuracion_MGAP_snig import TABLAS, OBJETIVOS

BASE = Path(__file__).resolve().parent
SNIG = BASE.parents[2] / 'data/processed/SNIG'
SALIDAS = BASE / 'resultados_mgap_snig'

def cargar_tabla(tabla):
    config = TABLAS[tabla]
    cat = config['categoricas']
    num = config['numericas']
    requeridas = cat + num + OBJETIVOS
    # La validación de columnas es estricta: una columna ausente produce error.
    df = pd.read_csv(SNIG / f'{tabla}.csv', sep=';', usecols=requeridas,
                     na_values=['-', ' ', 'NULL', 'null'], low_memory=False)
    df = df.dropna(subset=OBJETIVOS).copy()
    for col in cat:
        df[col] = df[col].fillna('SIN_DATO').astype(str)
    for col in num:
        # Importante para UnidadesGanaderas y SuperficieGanadera: usan coma decimal.
        convertido = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        if (df[col].notna() & convertido.isna()).any():
            raise ValueError(f'{tabla}: hay valores no numéricos en {col}')
        df[col] = convertido.fillna(0)
    X = df[cat + num].copy()
    for nueva, (numerador, denominador) in config['proporciones'].items():
        X[nueva] = X[numerador] / X[denominador].clip(lower=1)
    y = df[OBJETIVOS].astype(int)
    return df, X, y

def codificar(X_train, X_test, tabla):
    cat = TABLAS[tabla]['categoricas']
    X_train = pd.get_dummies(X_train, columns=cat)
    X_test = pd.get_dummies(X_test, columns=cat).reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test

def calcular_pesos(y, exponente):
    # 1 = balanceado; 0.5 = suavizado; 0.25 = leve; 0 = sin pesos.
    conteos = y.value_counts()
    pesos = (len(y) / (len(conteos) * conteos)) ** exponente
    return y.map(pesos).to_numpy()

def separar_aleatorio(X, y, semilla=42):
    # Estratificar por la pareja conserva mejor los dos objetivos.
    # Una pareja con una sola fila queda en entrenamiento porque no se puede dividir.
    estrato = y.astype(str).agg('|'.join, axis=1)
    conteos = estrato.value_counts()
    raras = estrato.map(conteos).lt(2)
    train, test = train_test_split(y.index[~raras], test_size=0.2,
                                  random_state=semilla, stratify=estrato.loc[~raras])
    train = train.append(y.index[raras])
    return X.loc[train], X.loc[test], y.loc[train], y.loc[test]

def crear_modelo(nombre, y_train, hilos=4, iteraciones=250):
    if nombre == 'Mayoritaria':
        return DummyClassifier(strategy='most_frequent')
    if nombre.startswith('RF_'):
        return RandomForestClassifier(n_estimators=100, max_depth=20,
            min_samples_leaf=3, random_state=42, n_jobs=hilos)
    if nombre.startswith('ExtraTrees_'):
        return ExtraTreesClassifier(n_estimators=100, max_depth=25,
            min_samples_leaf=3, random_state=42, n_jobs=hilos)
    if nombre.startswith('HistBoost_'):
        # La partición interna estratificada falla si una clase tiene una sola fila.
        detener = len(y_train) > 10_000 and y_train.value_counts().min() >= 2
        return HistGradientBoostingClassifier(max_iter=iteraciones, max_leaf_nodes=31,
            learning_rate=0.1, l2_regularization=1, early_stopping=detener, random_state=42)
    raise ValueError(f'Modelo desconocido: {nombre}')

def evaluar(tabla, etapa, nombre, objetivo, y_train, real, pred, segundos):
    # El mismo conjunto de clases para todos los candidatos: las presentes en la prueba.
    etiquetas = sorted(real.unique())
    ausentes = sorted(set(etiquetas) - set(y_train.unique()))
    metricas = dict(tabla=tabla, etapa=etapa, modelo=nombre, objetivo=objetivo,
        accuracy=accuracy_score(real, pred),
        f1_macro=f1_score(real, pred, labels=etiquetas, average='macro', zero_division=0),
        f1_weighted=f1_score(real, pred, labels=etiquetas, average='weighted', zero_division=0),
        filas_train=len(y_train), filas_prueba=len(real), segundos=segundos,
        clases_prueba=' '.join(map(str, etiquetas)), clases_sin_entrenamiento=' '.join(map(str, ausentes)))
    reporte = (f"Acierto: {metricas['accuracy']:.4f}\n"
               f"F1 macro (clases con soporte en prueba): {metricas['f1_macro']:.4f}\n"
               f"Clases sin ejemplos en entrenamiento: {ausentes}\n\n"
               + classification_report(real, pred, labels=etiquetas, digits=4, zero_division=0))
    carpeta = SALIDAS / tabla
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f'{etapa}_{nombre}_{objetivo}.txt').write_text(reporte, encoding='utf-8')
    print(metricas, flush=True)
    return metricas
