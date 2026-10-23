"""Espacios acotados para ajustar los modelos MGAP ya comparados."""

# Valores candidatos para cada hiperparámetro, agrupados por familia de modelo.
ESPACIOS = {
    # Random Forest: cantidad de árboles, profundidad, condiciones de división y variables por división.
    'RF': {
        'n_estimators': [100, 200, 400],
        'max_depth': [12, 16, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 5, 8],
        'max_features': ['sqrt', 0.5, 1.0],
    },
    # Extra Trees: mismos tipos de controles, con un punto de partida de profundidad distinto.
    'ExtraTrees': {
        'n_estimators': [100, 200, 400],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 5, 8],
        'max_features': ['sqrt', 0.5, 1.0],
    },
    # Gradient Boosting por histogramas: iteraciones, tasa de aprendizaje, hojas y regularización.
    'HistBoost': {
        'max_iter': [100, 200, 250, 350],
        'learning_rate': [0.03, 0.05, 0.08, 0.10, 0.15],
        'max_leaf_nodes': [15, 31, 63],
        'min_samples_leaf': [10, 20, 30, 50],
        'l2_regularization': [0.0, 0.1, 0.5, 1.0, 3.0],
    },
}

# El exponente ya existe en el pipeline actual: 0 no pesa las clases y 1 usa
# el peso balanceado completo. Los demás valores hacen la transición gradual.
EXPONENTES_PESOS = [0.0, 0.25, 0.40, 0.50, 0.65, 0.80, 1.0]

# Cantidad de configuraciones iniciales por objetivo. Los detalles animales
# usan menos porque cada entrenamiento tarda bastante más que el resto.
# Presupuesto adaptado al costo computacional de las tablas más grandes.
PRESUPUESTO_POR_TABLA = {
    'DatosAnimalesDetallados': {'intentos': 12, 'top': 3, 'semillas': [43, 44, 45]},
    'DatosProduccionLeche': {'intentos': 40, 'top': 5, 'semillas': [43, 44, 45, 46, 47]},
    'DatosProduccionLecheEnEstablecimiento': {'intentos': 40, 'top': 5, 'semillas': [43, 44, 45, 46, 47]},
}
# Presupuesto para cualquier tabla que no tenga una regla específica arriba.
PRESUPUESTO_PREDETERMINADO = {'intentos': 24, 'top': 5, 'semillas': [43, 44, 45]}

# Si el segundo modelo quedó a menos de este margen, se ajustan ambas familias.
MARGEN_DOS_FAMILIAS = 0.01
# Mejora mínima promedio de F1 macro necesaria para sustituir al modelo actual.
MEJORA_MINIMA_F1 = 0.005


def presupuesto(tabla):
    """Devuelve una copia del presupuesto de la tabla o el predeterminado."""
    # `.copy()` impide que las opciones de consola modifiquen las constantes globales.
    return PRESUPUESTO_POR_TABLA.get(tabla, PRESUPUESTO_PREDETERMINADO).copy()
