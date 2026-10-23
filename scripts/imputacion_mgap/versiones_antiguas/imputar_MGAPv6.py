from __future__ import annotations

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

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

# 1. Cargar dataset con imputación de tipos
df_completo = pd.read_csv(RUTA, sep=";", na_values=['-', ' ', 'NULL', 'null'])

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

df = df_completo.dropna(subset=cols_y).copy()

for col in cols_cat:
    df[col] = df[col].astype(str)

for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# One-Hot Encoding
X = pd.get_dummies(df[cols_x], columns=cols_cat, drop_first=True)
y = df[cols_y]

# Train / Test Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['TipoProduccionMGAPCodigo']
)

# -------------------------------------------------------------------
# MODELO 1: Especialización
# -------------------------------------------------------------------
rf_esp = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=3,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
rf_esp.fit(X_train, y_train['EspecializacionMGAPCodigo'])

pred_esp_train = rf_esp.predict(X_train)
pred_esp_test = rf_esp.predict(X_test)

# -------------------------------------------------------------------
# MODELO 2: Tipo de Producción (Incorpora predicción de Especialización)
# -------------------------------------------------------------------
X_train_enc = X_train.copy()
X_train_enc['Especializacion_Predicha'] = pred_esp_train

X_test_enc = X_test.copy()
X_test_enc['Especializacion_Predicha'] = pred_esp_test

# Cálculo de pesos suavizados por raíz cuadrada: (N_total / (N_clases * N_clase)) ^ 0.5
counts = y_train['TipoProduccionMGAPCodigo'].value_counts()
total_samples = len(y_train)
n_classes = len(counts)

pesos_suavizados = {
    cls: float((total_samples / (n_classes * count)) ** 0.5) 
    for cls, count in counts.items()
}

rf_tipo = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=4,
    class_weight=pesos_suavizados,
    random_state=42,
    n_jobs=-1
)

rf_tipo.fit(X_train_enc, y_train['TipoProduccionMGAPCodigo'])
pred_tipo = rf_tipo.predict(X_test_enc)

# -------------------------------------------------------------------
# EVALUACIÓN
# -------------------------------------------------------------------
print("--- Evaluación Random Forest: EspecializacionMGAPCodigo ---")
print(classification_report(y_test['EspecializacionMGAPCodigo'], pred_esp_test))

print("\n--- Evaluación Random Forest: TipoProduccionMGAPCodigo ---")
print(classification_report(y_test['TipoProduccionMGAPCodigo'], pred_tipo))
