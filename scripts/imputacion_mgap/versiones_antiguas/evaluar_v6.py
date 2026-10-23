"""Ejecuta la v6 original con sus mismos parámetros en dos particiones."""
from pathlib import Path
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE = Path(__file__).resolve().parent
SALIDAS = BASE / 'resultados_mgap'
SALIDAS.mkdir(exist_ok=True)
df = pd.read_csv(BASE.parents[2] / 'data/processed/SNIG/DatosAnimales.csv', sep=';',
                 na_values=['-', ' ', 'NULL', 'null'])
objetivos = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']
df = df.dropna(subset=objetivos).copy()
cat = ['GiroCodigo', 'EspecieCodigo']
num = ['EstratoCodigo', 'AnimalesPorEspecieConsumoPD_AD',
       'AnimalesPorEspecieNacimientosPD_AD', 'AnimalesPorEspecieMortandadPD_AD',
       'CantidadTenedoresPorEspecie', 'AnimalesPorEspeciePD_AD', 'AnimalesPorEspeciePD_PF']
for col in cat:
    df[col] = df[col].astype(str)
for col in num:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
X = pd.get_dummies(df[cat + num], columns=cat, drop_first=True)
y = df[objetivos]
resultados = []

for etapa in ['aleatorio', 'historico_2022']:
    if etapa == 'aleatorio':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y[objetivos[1]])
    else:
        train = df.Anio.gt(2022)
        test = df.Anio.eq(2022)
        X_train, X_test, y_train, y_test = X[train].copy(), X[test].copy(), y[train], y[test]
    print(f'Entrenando v6: {etapa}', flush=True)
    rf_esp = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=3,
                                   class_weight='balanced_subsample', random_state=42, n_jobs=4)
    rf_esp.fit(X_train, y_train[objetivos[0]])
    pred_esp = rf_esp.predict(X_test)
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()
    X_train_enc['Especializacion_Predicha'] = rf_esp.predict(X_train)
    X_test_enc['Especializacion_Predicha'] = pred_esp
    counts = y_train[objetivos[1]].value_counts()
    pesos = {clase: float((len(y_train) / (len(counts)*n))**0.5) for clase, n in counts.items()}
    rf_tipo = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_leaf=4,
                                    class_weight=pesos, random_state=42, n_jobs=4)
    rf_tipo.fit(X_train_enc, y_train[objetivos[1]])
    for objetivo, pred in zip(objetivos, [pred_esp, rf_tipo.predict(X_test_enc)]):
        reporte = classification_report(y_test[objetivo], pred, digits=4, zero_division=0)
        print(objetivo, '\n', reporte, flush=True)
        (SALIDAS / f'{etapa}_v6_{objetivo}.txt').write_text(reporte, encoding='utf-8')
        resultados.append(dict(etapa=etapa, modelo='v6', objetivo=objetivo,
            accuracy=accuracy_score(y_test[objetivo], pred),
            f1_macro=f1_score(y_test[objetivo], pred, average='macro'),
            f1_weighted=f1_score(y_test[objetivo], pred, average='weighted')))
    pd.DataFrame(resultados).to_csv(SALIDAS / 'evaluacion_v6.csv', index=False)
