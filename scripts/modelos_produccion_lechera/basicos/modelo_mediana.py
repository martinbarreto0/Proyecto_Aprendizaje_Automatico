"""Entrena la referencia más sencilla: siempre predice la mediana de litros."""

# NumPy permite crear una columna ficticia para DummyRegressor.
import numpy as np

# Joblib permite guardar el modelo entrenado en un archivo.
from joblib import dump

# DummyRegressor implementa una predicción constante y sirve como referencia mínima.
from sklearn.dummy import DummyRegressor

# Se importan las funciones comunes de carga, división, métricas y guardado.
from datos_lecheria import (
    CARPETA_RESULTADOS,
    OBJETIVO,
    calcular_metricas,
    construir_dataset,
    dividir_dataset,
    guardar_resultados,
)


def main():
    """Construye los datos, entrena la mediana y evalúa solo en validación."""

    # Se construye el dataset usando todos los ejercicios disponibles desde 2016.
    dataset = construir_dataset()

    # Se crean las particiones aleatorias 70/15/15.
    entrenamiento, validacion, _prueba_reservada = dividir_dataset(dataset)

    # DummyRegressor necesita recibir una X aunque ignore completamente sus valores.
    x_entrenamiento = np.zeros((len(entrenamiento), 1))

    # Se crea la misma columna ficticia para las filas de validación.
    x_validacion = np.zeros((len(validacion), 1))

    # Se extraen los litros reales usados para aprender la mediana.
    y_entrenamiento = entrenamiento[OBJETIVO]

    # Se extraen los litros reales que se usarán solamente para medir el resultado.
    y_validacion = validacion[OBJETIVO]

    # Se crea el modelo que siempre devolverá la mediana del entrenamiento.
    modelo = DummyRegressor(strategy="median")

    # Se calcula y guarda la mediana a partir del conjunto de entrenamiento.
    modelo.fit(x_entrenamiento, y_entrenamiento)

    # Se genera una predicción para cada fila de validación.
    predicciones = modelo.predict(x_validacion)

    # Se calculan MAE, RMSE, R² y WAPE sin tocar el test reservado.
    metricas = calcular_metricas("Mediana", y_validacion, predicciones)

    # Se guardan tanto las métricas como las predicciones fila por fila.
    ruta_metricas, ruta_predicciones = guardar_resultados(
        "mediana",
        validacion,
        predicciones,
        metricas,
    )

    # Se crea la carpeta de resultados si todavía no existe.
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Se define dónde quedará guardado el modelo de referencia.
    ruta_modelo = CARPETA_RESULTADOS / "modelo_mediana.joblib"

    # Se serializa el modelo para poder volver a usarlo sin reentrenar.
    dump(modelo, ruta_modelo)

    # Se muestran las métricas principales en la terminal.
    print("\nMétricas de validación:")

    # Se imprime cada métrica en una línea separada.
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor}")

    # Se informa la ubicación del CSV de métricas.
    print(f"\nMétricas guardadas en: {ruta_metricas}")

    # Se informa la ubicación del CSV de predicciones.
    print(f"Predicciones guardadas en: {ruta_predicciones}")

    # Se informa la ubicación del modelo serializado.
    print(f"Modelo guardado en: {ruta_modelo}")


# Este bloque evita que el entrenamiento se ejecute al importar el archivo.
if __name__ == "__main__":
    # El modelo se entrena únicamente cuando se ejecuta este script directamente.
    main()
