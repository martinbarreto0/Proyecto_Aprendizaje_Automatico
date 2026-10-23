from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # O el número de núcleos lógicos de tu procesador
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SNIG_DIR = ROOT / "data" / "processed" / "SNIG"
RUTA = SNIG_DIR / "DatosAnimales.csv"

# 1. Leer CSV reemplazando '-' por NaN directamente
df_completo = pd.read_csv(RUTA, sep=";", na_values=['-', ' ', 'NULL', 'null'])

cols_x = ['GiroCodigo', 'EstratoCodigo', 'EspecieCodigo']
cols_y = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']

# 2. Eliminar filas con nulos tanto en predictores como en objetivos
df = df_completo.dropna(subset=cols_x + cols_y).copy()

# 3. Aplicar One-Hot Encoding a las variables categóricas nominales
# Convertimos GiroCodigo y EspecieCodigo a texto antes de get_dummies
df['GiroCodigo'] = df['GiroCodigo'].astype(str)
df['EspecieCodigo'] = df['EspecieCodigo'].astype(str)

X = pd.get_dummies(df[cols_x], columns=['GiroCodigo', 'EspecieCodigo'], drop_first=True)
y = df[cols_y]

# 4. Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Entrenar KNN
knn = KNeighborsClassifier(n_neighbors=5,weights="distance", n_jobs=-1) # n_jobs=-1 usa todos los núcleos para acelerar
knn.fit(X_train_scaled, y_train)

# 7. Evaluar
y_pred = knn.predict(X_test_scaled)
y_pred_df = pd.DataFrame(y_pred, columns=y.columns, index=y_test.index)

print("--- Evaluación para EspecializacionMGAPCodigo ---")
print(classification_report(y_test['EspecializacionMGAPCodigo'], y_pred_df['EspecializacionMGAPCodigo']))

print("\n--- Evaluación para TipoProduccionMGAPCodigo ---")
print(classification_report(y_test['TipoProduccionMGAPCodigo'], y_pred_df['TipoProduccionMGAPCodigo']))
