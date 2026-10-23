from __future__ import annotations

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" # Evita el UserWarning en Windows

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SNIG_DIR = ROOT / "data" / "processed" / "SNIG"
RUTA = SNIG_DIR / "DatosAnimales.csv"

# 1. Cargar CSV
df_completo = pd.read_csv(RUTA, sep=";", na_values=['-', ' ', 'NULL', 'null'])

cols_x = ['GiroCodigo', 'EstratoCodigo', 'EspecieCodigo']
cols_y = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']

# 2. Filtrar nulos
df = df_completo.dropna(subset=cols_x + cols_y).copy()

# 3. One-Hot Encoding para las categóricas
df['GiroCodigo'] = df['GiroCodigo'].astype(str)
df['EspecieCodigo'] = df['EspecieCodigo'].astype(str)

X = pd.get_dummies(df[cols_x], columns=['GiroCodigo', 'EspecieCodigo'], drop_first=True)
y = df[cols_y]

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Separar en dos modelos independientes
rf_especializacion = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', # Obliga al modelo a prestar atención a clases pequeñas
    random_state=42, 
    n_jobs=-1
)

rf_tipo_prod = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1
)

# 2. Entrenar por separado (puedes usar X sin escalar)
rf_especializacion.fit(X_train, y_train['EspecializacionMGAPCodigo'])
rf_tipo_prod.fit(X_train, y_train['TipoProduccionMGAPCodigo'])

# 3. Predecir por separado
pred_esp = rf_especializacion.predict(X_test)
pred_tipo = rf_tipo_prod.predict(X_test)

# 4. Evaluar
print("--- Evaluación Random Forest: EspecializacionMGAPCodigo ---")
print(classification_report(y_test['EspecializacionMGAPCodigo'], pred_esp))

print("\n--- Evaluación Random Forest: TipoProduccionMGAPCodigo ---")
print(classification_report(y_test['TipoProduccionMGAPCodigo'], pred_tipo))
