# Predicción de producción lechera con datos SNIG

Este proyecto completa dos códigos MGAP faltantes en siete tablas SNIG y utiliza cuatro de esas tablas para estimar la producción anual de leche (`Litros`) por perfil productivo. Este README describe los **datos y resultados finales publicados**; los archivos de experimentos anteriores siguen disponibles para reproducir las comparaciones.

## Contenido del repositorio

| Ruta | Contenido |
|---|---|
| [`data/`](data/) | Siete CSV con los códigos MGAP completados y dos columnas de confianza; [`data/metadatos_modelos_finales/`](data/metadatos_modelos_finales/) registra la aplicación por tabla. |
| [`scripts/imputacion_mgap/`](scripts/imputacion_mgap/) | Preparación, comparación y aplicación de los clasificadores MGAP. |
| [`scripts/modelos_produccion_lechera/`](scripts/modelos_produccion_lechera/) | Construcción del dataset, modelos básicos, selección de variables, búsquedas de hiperparámetros y evaluación final. |
| [`reports/`](reports/) | Métricas, predicciones, artefactos entrenados y metadatos de los experimentos. |
| [`diagnostico_datos/`](diagnostico_datos/) | Diagnósticos exploratorios de los datos. |
| [`registro_uso_IA.md`](registro_uso_IA.md) | Declaración y registro del uso de IA. |

Los CSV originales de `data/processed/SNIG` **no forman parte de este repositorio**. Los siete archivos de `data/` son las tablas finales; para repetir la imputación desde cero hacen falta los originales. Los archivos grandes del repositorio usan Git LFS.

## Imputación MGAP

Para cada tabla y objetivo se compararon modelos individuales, ajustes de hiperparámetros y ensambles. La elección final se hizo por separado para `EspecializacionMGAPCodigo` y `TipoProduccionMGAPCodigo`, sobre una partición aleatoria común 80/20. Los modelos aplicados y sus métricas F1 macro se consultan en [`resumen_comparacion_final.csv`](scripts/imputacion_mgap/evaluacion_final_modelos/resultados_finales/resumen_comparacion_final.csv); los detalles de cada tabla están en [`data/metadatos_modelos_finales/`](data/metadatos_modelos_finales/). Esa comparación reutiliza datos históricos de desarrollo y no equivale a un test ciego independiente.

El [aplicador final](scripts/imputacion_mgap/evaluacion_final_modelos/aplicar_modelos_finales_MGAP_snig.py) conservó los códigos observados y completó solo las celdas faltantes de cada objetivo. Agregó `EspecializacionMGAPConfianza` y `TipoProduccionMGAPConfianza`, vacías cuando el código era conocido. Son puntajes del modelo **sin calibración probabilística**; no deben interpretarse como una probabilidad garantizada de acierto.

Las siete tablas finales suman **4.620.729 filas** (no perfiles únicos). Se completaron **2.565.161 valores de cada código MGAP**. Estas cifras provienen de los [metadatos de aplicación](data/metadatos_modelos_finales/).

## Dataset y modelos de producción lechera

[`construir_dataset()`](scripts/modelos_produccion_lechera/basicos/datos_lecheria.py) lee `DatosProduccionLeche`, `DatosAnimales`, `DatosAnimalesDetallados` y `DatosGenerales`. Filtra los perfiles lecheros y, donde corresponde, los bovinos de leche; suma litros y existencias por ocho claves de perfil y ejercicio, transforma las siete categorías animales en columnas, une las cuatro tablas y calcula la carga ganadera. Conserva filas con `Litros > 0`. Las otras tres tablas MGAP no entran directamente en esta regresión. Los códigos MGAP se usan para **filtrar**, no como predictores; las confianzas tampoco son entradas del modelo.

El resultado actual tiene **16.204 filas y 21 columnas** para 2016–2025: ocho claves, doce medidas numéricas y `Litros`. Los modelos completos usan 16 predictores (cuatro categóricos y doce numéricos). [`dividir_dataset()`](scripts/modelos_produccion_lechera/basicos/datos_lecheria.py) mezcla los registros mediante `train_test_split` y conserva **11.342** para entrenamiento, **2.431** para validación y **2.431** para test (70/15/15). La partición no separa años.

Se compararon en **validación** 13 variantes: mediana, Ridge, Random Forest y CatBoost en versiones básicas, con selección de variables o con búsqueda exhaustiva de hiperparámetros según el caso. Ridge usa codificación one-hot y normalización de numéricas; Random Forest usa one-hot pero no normalización; CatBoost maneja las categorías y faltantes numéricos sin esas transformaciones. Los CSV de resultados están en [`reports/modelos_basicos/`](reports/modelos_basicos/), [`reports/modelos_seleccion_variables/`](reports/modelos_seleccion_variables/), [`reports/hiperparametros_modelos_basicos/`](reports/hiperparametros_modelos_basicos/) y [`reports/modelos_catboost/`](reports/modelos_catboost/).

El equipo eligió **CatBoost básico con las 12 variables numéricas seleccionadas**, guardado en [`modelo_top_12.cbm`](reports/modelos_catboost/seleccion_variables/modelo_top_12.cbm). Sus entradas y orden están en [`variables_top_12.csv`](reports/modelos_catboost/seleccion_variables/variables_top_12.csv). Este modelo no es el CatBoost top 12 ajustado mediante grid search; tampoco necesita one-hot ni normalización. Aunque el Random Forest ajustado obtuvo menor MAE en validación, el CatBoost elegido redujo el RMSE y el tamaño del artefacto. Frente al CatBoost completo básico, aumentó ligeramente el MAE y redujo el RMSE.

| Partición del CatBoost elegido | Filas | MAE (litros) | RMSE (litros) | R² | WAPE |
|---|---:|---:|---:|---:|---:|
| Validación | 2.431 | 250.079,99 | 575.768,85 | 0,97583 | 21,9606 % |
| Test final | 2.431 | 310.103,50 | 2.103.442,54 | 0,83542 | 22,8453 % |

Fuentes: [`metricas_top_12.csv`](reports/modelos_catboost/seleccion_variables/metricas_top_12.csv), [`metricas_test.csv`](reports/modelos_catboost/evaluacion_final_test/metricas_test.csv) y [`predicciones_test.csv`](reports/modelos_catboost/evaluacion_final_test/predicciones_test.csv). **Solo el modelo elegido fue evaluado en test**. Unos pocos errores extremos elevan mucho el RMSE de test; las métricas deben leerse conjuntamente.

## Preparación y comprobación

Se requiere Python 3.11 y Git LFS, además de espacio para descargar los artefactos grandes. En PowerShell:

```powershell
git clone https://github.com/martinbarreto0/Proyecto_Aprendizaje_Automatico.git
Set-Location Proyecto_Aprendizaje_Automatico
git lfs install
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r scripts/modelos_produccion_lechera/catboost/requirements.txt
.\.venv\Scripts\python.exe scripts/modelos_produccion_lechera/evaluacion_final_test/evaluar_test_top_12.py
```

El último comando **no reentrena ni vuelve a abrir el test**: comprueba las huellas SHA-256 de seis insumos, reconstruye el dataset y reproduce las métricas de validación del artefacto publicado. Sin la bandera de test, el script muestra al final «Test no evaluado»; ese mensaje se refiere solo a **esta ejecución de control**, no al estado del proyecto. Las métricas y predicciones de test ya están en [`reports/modelos_catboost/evaluacion_final_test/`](reports/modelos_catboost/evaluacion_final_test/). El modo de evaluación de test se usó una sola vez; para consultar los resultados publicados no hace falta activarlo. Ejecutar scripts de entrenamiento puede sobrescribir sus salidas, por lo que la comprobación anterior es el punto de entrada recomendado para revisar el estado publicado.

## Alcance de los resultados

La división aleatoria mezcla ejercicios: el test evalúa registros del período 2016–2025 y **no mide pronóstico de años futuros**. La imputación MGAP introduce incertidumbre en el filtro de perfiles lecheros; en `DatosProduccionLeche` emplea `Litros` para estimar códigos que luego intervienen en ese filtro. Por ello, esta evaluación no demuestra aislamiento de información de extremo a extremo. El modelo produjo algunas predicciones negativas en test y no debe utilizarse sin controles de plausibilidad para decisiones operativas.
