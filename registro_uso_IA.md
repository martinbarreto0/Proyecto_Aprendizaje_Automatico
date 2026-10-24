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
| Codex (OpenAI) | Versión exacta de modelo no consolidada en este registro | Durante el desarrollo de 2026 | Asistencia interactiva en imputación MGAP, modelos de lechería, redacción técnica y preparación de archivos para GitHub y el servidor. |

## Solicitudes y respuestas relevantes de las sesiones de Codex

Los fragmentos entre comillas son solicitudes del equipo registradas en la conversación. La columna «resultado» resume la respuesta y su aplicación, **no pretende ser una cita textual del asistente**. Las rutas se interpretan desde la raíz de `PAA`; los CSV/JSON y scripts indicados permiten comprobar el resultado técnico. Se consignan ejemplos que influyeron en código, datos, evaluación o decisiones, no cada intercambio de la conversación.

| Solicitud del equipo (fragmento) | Resultado de la asistencia y uso | Comprobación |
|---|---|---|
| «divide el conjunto de datos con traintestsplit (que tambien los mezcla)» | Se modificó la evaluación de clasificadores MGAP para comparar candidatos con partición aleatoria; después se congelaron los candidatos y se los comparó en una partición común 80/20. | `scripts/imputacion_mgap/evaluacion_final_modelos/evaluar_modelos_finales.py`; `scripts/imputacion_mgap/evaluacion_final_modelos/resultados_finales/resumen_comparacion_final.csv`. |
| «el .py que completa las tablas originales en las columnas MGAP con el modelo establecido para cada tabla-columna» | Se preparó y ejecutó la aplicación por tabla y objetivo. Los códigos observados quedaron preservados y las tablas finales incorporaron dos columnas de confianza. | `scripts/imputacion_mgap/evaluacion_final_modelos/aplicar_modelos_finales_MGAP_snig.py`; `data/metadatos_modelos_finales/`; `data/*.csv`. |
| «la prueba con el test (no quiero que la hagas, solo explica como hacerlo ...)» | Primero se preparó el procedimiento sin ejecutar test. Una evaluación posterior, realizada por el equipo, produjo las métricas y predicciones finales del modelo elegido. | `scripts/modelos_produccion_lechera/evaluacion_final_test/evaluar_test_top_12.py`; `reports/modelos_catboost/evaluacion_final_test/`. |

## Verificación, límites y pendientes del registro

Las cifras vigentes deben verificarse en los archivos de salida, no inferirse del texto de una respuesta de IA. En particular, las siete tablas finales suman 4.620.729 filas, con 2.565.161 imputaciones por código MGAP; el dataset lechero tiene 16.204 filas; y el CatBoost elegido fue evaluado sobre 2.431 filas de test. Las fuentes son `data/metadatos_modelos_finales/`, `scripts/modelos_produccion_lechera/basicos/datos_lecheria.py` y `reports/modelos_catboost/evaluacion_final_test/metricas_test.csv`.

Las filas MGAP imputadas son estimaciones, no observaciones originales. Sus confianzas no están calibradas. La partición aleatoria de la regresión no equivale a pronosticar años futuros.
