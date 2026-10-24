# Registro de uso de inteligencia artificial

## Declaración

El equipo utilizó herramientas de IA generativa como apoyo para programar, ejecutar y revisar experimentos, interpretar salidas, organizar archivos y redactar documentación. Las personas integrantes definieron los objetivos, eligieron el modelo final y son responsables de comprobar los datos, resultados y afirmaciones del proyecto. Las herramientas pudieron editar archivos y ejecutar comandos bajo instrucciones del equipo; por eso sería incorrecto afirmar que la IA no modificó datos o no generó métricas durante el trabajo.

## Herramientas y actividades declaradas

Se conservan las declaraciones sobre el trabajo de las compañeras con Claude Code. Los nombres de los modelos y períodos de esas sesiones son los que ellas aportaron; este registro no incluye todavía las transcripciones completas de sus prompts y respuestas.

| Herramienta | Modelos declarados | Período declarado | Actividades |
|---|---|---|---|
| Claude Code (Anthropic) | Claude Opus 5.5 y Claude Sonnet 5.5 | Septiembre-octubre de 2026 | Apoyo en preparación de datos, scripts de imputación y modelado, organización de carpetas, explicación de métricas, documentación y bibliografía. |
| Claude Code (Anthropic) | Claude Opus 5.5 y Claude Sonnet 5.5 | Octubre de 2026 | Adecuación del informe al formato institucional, scripts y figuras de `PAA/informe` y redacción del Entregable 3. |
| Claude Code (Anthropic) | Claude Sonnet 5.5 | 23 oct. 2026 | Programación y prueba de la interfaz del prototipo (simulador, filtros y descarga CSV en tablas, orden de pestañas), configuración Docker del dashboard y actualización del informe y documentación. |
| Codex (OpenAI) | Versión exacta de modelo no consolidada en este registro | Durante el desarrollo de 2026 | Asistencia interactiva en imputación MGAP, modelos de lechería, análisis de resultados, ejecución de scripts, redacción técnica y preparación de archivos para GitHub y el servidor. |

## Solicitudes y respuestas relevantes de las sesiones de Codex

Los fragmentos entre comillas son solicitudes del equipo registradas en la conversación. La columna «resultado» resume la respuesta y su aplicación, **no pretende ser una cita textual del asistente**. Las rutas se interpretan desde la raíz de `PAA`; los CSV/JSON y scripts indicados permiten comprobar el resultado técnico. Se consignan ejemplos que influyeron en código, datos, evaluación o decisiones, no cada intercambio de la conversación.

| Solicitud del equipo (fragmento) | Resultado de la asistencia y uso | Comprobación |
|---|---|---|
| «divide el conjunto de datos con traintestsplit (que tambien los mezcla)» | Se modificó la evaluación de clasificadores MGAP para comparar candidatos con partición aleatoria; después se congelaron los candidatos y se los comparó en una partición común 80/20. | `scripts/imputacion_mgap/evaluacion_final_modelos/evaluar_modelos_finales.py`; `scripts/imputacion_mgap/evaluacion_final_modelos/resultados_finales/resumen_comparacion_final.csv`. |
| «el .py que completa las tablas originales en las columnas MGAP con el modelo establecido para cada tabla-columna» | Se preparó y ejecutó la aplicación por tabla y objetivo. Los códigos observados quedaron preservados y las tablas finales incorporaron dos columnas de confianza. | `scripts/imputacion_mgap/evaluacion_final_modelos/aplicar_modelos_finales_MGAP_snig.py`; `data/metadatos_modelos_finales/`; `data/*.csv`. |
| «recuerda mantener el mismo dataset ... y la separacion ... test no lo vamos a tocar hasta el final» | Se conservaron la construcción del dataset y la partición 70/15/15 para comparar modelos básicos, selección de variables y búsquedas de hiperparámetros en validación. | `scripts/modelos_produccion_lechera/basicos/datos_lecheria.py`; `reports/modelos_basicos/`; `reports/modelos_catboost/`; `reports/hiperparametros_modelos_basicos/`. |
| «Finalmente elgimos que el modelo final sea el CatBoost top 12» | Se documentó la selección del CatBoost **básico** top 12 frente al Random Forest ajustado y al CatBoost completo, atendiendo MAE, RMSE, R² y tamaño del artefacto. La decisión fue del equipo. | `reports/modelos_catboost/seleccion_variables/metricas_top_12.csv`; `reports/modelos_catboost/seleccion_variables/modelo_top_12.cbm`; métricas de los candidatos en `reports/`. |
| «la prueba con el test (no quiero que la hagas, solo explica como hacerlo ...)» | Primero se preparó el procedimiento sin ejecutar test. Una evaluación posterior, realizada por el equipo, produjo las métricas y predicciones finales del modelo elegido. | `scripts/modelos_produccion_lechera/evaluacion_final_test/evaluar_test_top_12.py`; `reports/modelos_catboost/evaluacion_final_test/`. |
| «actualices el entregable_3.docx ... revisa cualquier dato que vayas a poner y no inventes nada» | Se contrastaron cifras con los CSV/JSON y se asistió la redacción del informe. El equipo debe revisar el documento final y conservar solo afirmaciones respaldadas. | `Entregable_3.docx`; `reports/`; `data/metadatos_modelos_finales/`. |

## Verificación, límites y pendientes del registro

Las cifras vigentes deben verificarse en los archivos de salida, no inferirse del texto de una respuesta de IA. En particular, las siete tablas finales suman 4.620.729 filas, con 2.565.161 imputaciones por código MGAP; el dataset lechero tiene 16.204 filas; y el CatBoost elegido fue evaluado sobre 2.431 filas de test. Las fuentes son `data/metadatos_modelos_finales/`, `scripts/modelos_produccion_lechera/basicos/datos_lecheria.py` y `reports/modelos_catboost/evaluacion_final_test/metricas_test.csv`.

Las filas MGAP imputadas son estimaciones, no observaciones originales. Sus confianzas no están calibradas. La partición aleatoria de la regresión no equivale a pronosticar años futuros. Además, en una de las tablas el código MGAP se estimó usando `Litros`, y ese código interviene luego en el filtro del dataset lechero; la evaluación de test del regresor no resuelve por sí sola ese posible sesgo de selección.

Queda por incorporar, con ayuda de quienes realizaron esas sesiones, una selección de **prompts y respuestas relevantes de Claude Code** y cómo se aceptaron, modificaron, descartaron o verificaron. No se inventan aquí intercambios que no constan en el material disponible. También conviene contrastar la tabla de herramientas del `Entregable_3.docx` con este registro antes de la entrega, pues aquí se explicita el uso adicional de Codex. Ese documento y `PAA/informe` se conservan en la carpeta de trabajo del equipo; no forman parte de este repositorio de GitHub.
