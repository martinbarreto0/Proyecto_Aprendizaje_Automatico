from __future__ import annotations

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" # Evita el UserWarning en Windows

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SNIG_DIR = ROOT / "data" / "processed" / "SNIG"
RUTA = SNIG_DIR / "DatosAnimales.csv"

# 1. Cargar CSV reconociendo representaciones de nulos
df_completo = pd.read_csv(RUTA, sep=";", na_values=['-', ' ', 'NULL', 'null'])

# Categorización de variables
cols_cat = ['GiroCodigo', 'EspecieCodigo']
cols_num = [
    'EstratoCodigo',
    'AnimalesPorEspecieConsumoPD_AD',
    'AnimalesPorEspecieNacimientosPD_AD',
    'AnimalesPorEspecieMortandadPD_AD',
    'CantidadTenedoresPorEspecie',
    'AnimalesPorEspeciePD_AD',
    'AnimalesPorEspeciePD_PF'
]

cols_x = cols_cat + cols_num
cols_y = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']

# 2. Filtrar registros donde falte la variable objetivo (Y)
df = df_completo.dropna(subset=cols_y).copy()

# 3. Limpieza y formateo de features (X)
for col in cols_cat:
    df[col] = df[col].astype(str)

for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# One-Hot Encoding para las categóricas
X = pd.get_dummies(df[cols_x], columns=cols_cat, drop_first=True)
y = df[cols_y]

# 4. Split estratificado para preservar la distribución de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['TipoProduccionMGAPCodigo']
)

# 5. Definir y entrenar modelos
# Tip: Si 'class_weight="balanced"' sigue distorsionando las clases mayoritarias,
# puedes probar cambiarlo a 'balanced_subsample' o quitarlo (None).
rf_especializacion = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced',
    random_state=42, 
    n_jobs=-1
)

rf_tipo_prod = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1
)

rf_especializacion.fit(X_train, y_train['EspecializacionMGAPCodigo'])
rf_tipo_prod.fit(X_train, y_train['TipoProduccionMGAPCodigo'])

# 6. Predicción y Evaluación
pred_esp = rf_especializacion.predict(X_test)
pred_tipo = rf_tipo_prod.predict(X_test)

print("--- Evaluación Random Forest: EspecializacionMGAPCodigo ---")
print(classification_report(y_test['EspecializacionMGAPCodigo'], pred_esp))

print("\n--- Evaluación Random Forest: TipoProduccionMGAPCodigo ---")
print(classification_report(y_test['TipoProduccionMGAPCodigo'], pred_tipo))

importances = pd.Series(rf_tipo_prod.feature_importances_, index=X.columns)
print("\n--- Featura Importance Random Forest: TipoProduccionMGAPCodigo ---")
print(importances.sort_values(ascending=False).head(10))
