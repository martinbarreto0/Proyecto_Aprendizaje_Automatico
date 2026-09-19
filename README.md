# Predicción de producción lechera

Proyecto de aprendizaje automático orientado a estimar la producción de leche, medida en litros, a partir de información productiva, ganadera, territorial y administrativa. El trabajo emplea registros desde 2016 y construye modelos de referencia y modelos supervisados para comparar su capacidad predictiva.

## Contenido del repositorio

```text
data/                                       Datasets SNIG con códigos MGAP completados
data/metadatos/                             Metadatos de la imputación por tabla
scripts/imputacion_mgap/                    Código de aplicación de la imputación MGAP
scripts/modelos_produccion_lechera/basicos/ Modelos base de producción lechera
reports/modelos_basicos/                    Métricas y artefactos generados por los modelos
registro_uso_IA.md                          Declaración de uso de inteligencia artificial
```

## Datos

La carpeta `data` contiene siete tablas con información de producción, existencias animales, categorías animales, características generales del establecimiento, producción en establecimiento, tenencia y uso de la tierra:

- `DatosProduccionLeche.csv`
- `DatosAnimales.csv`
- `DatosAnimalesDetallados.csv`
- `DatosGenerales.csv`
- `DatosProduccionLecheEnEstablecimiento.csv`
- `DatosTenenciasTierra.csv`
- `DatosUsosTierra.csv`

Cada tabla incluye las columnas `EspecializacionMGAPCodigo` y `TipoProduccionMGAPCodigo`. Cuando ambos códigos estaban ausentes en un registro, fueron estimados mediante modelos de clasificación seleccionados para cada tabla y objetivo. Las columnas `EspecializacionMGAPConfianza` y `TipoProduccionMGAPConfianza` indican la probabilidad máxima asociada a la clase estimada; quedan vacías cuando el código MGAP era observado.

Los detalles de los conteos de filas y de la imputación se encuentran en [data/README.md](data/README.md). Los modelos de producción no utilizan las columnas de confianza.

## Construcción del dataset de producción lechera

Los modelos básicos construyen un dataset analítico a partir de cuatro tablas:

- producción de leche;
- animales por especie;
- animales por categoría;
- variables generales del establecimiento.

Se conservan registros desde 2016 que corresponden a especialización lechera (`EspecializacionMGAPCodigo = 2`) y tipo de producción lechera (`TipoProduccionMGAPCodigo = 3`). Para las tablas que contienen especies se selecciona la especie bovina de leche (`EspecieCodigo = 11`).

Los litros se agregan por perfil productivo y ejercicio. Las tablas se unen mediante las variables de identificación territorial, administrativa y de estratificación disponibles: año, departamento, seccional policial, área de supervisión, área de enumeración, giro, naturaleza jurídica y estrato. El conjunto final incorpora existencias bovinas totales y por categoría, superficie, unidades ganaderas, superficie ganadera y carga ganadera. Se excluyen filas sin producción válida o con litros menores o iguales a cero.

La división de datos es aleatoria y reproducible: 70 % para entrenamiento, 15 % para validación y 15 % para prueba final. El conjunto de prueba se reserva y no interviene en los resultados presentados a continuación.

## Modelos básicos

La carpeta `scripts/modelos_produccion_lechera/basicos` contiene tres modelos sin búsqueda de hiperparámetros:

- **Mediana:** referencia mínima que asigna a todos los casos la mediana de litros observada en entrenamiento.
- **Ridge:** regresión lineal regularizada, útil como referencia explicable y para revisar los coeficientes asociados a las variables transformadas.
- **Random Forest:** conjunto de árboles de decisión que representa relaciones no lineales e interacciones entre las variables.

Los tres scripts usan la misma construcción de datos, variables predictoras y división aleatoria. Los valores faltantes numéricos se imputan dentro del flujo de cada modelo usando únicamente el conjunto de entrenamiento.

## Resultados iniciales de validación

Las siguientes métricas corresponden a 2.418 registros de validación. Un menor MAE, RMSE y WAPE indica menor error; un R² más cercano a 1 indica mayor capacidad explicativa.

| Modelo | MAE (litros) | RMSE (litros) | R² | WAPE |
|---|---:|---:|---:|---:|
| Mediana | 992.717 | 3.640.720 | -0,0348 | 81,88 % |
| Ridge | 455.756 | 1.023.140 | 0,9183 | 37,59 % |
| Random Forest | 231.318 | 473.957 | 0,9825 | 19,08 % |

En esta etapa inicial, Random Forest obtuvo el menor error y el mayor R². Estos resultados son de validación; la evaluación definitiva deberá realizarse una única vez sobre el conjunto de prueba reservado después de definir los ajustes de hiperparámetros.

## Ejecución

Los comandos deben ejecutarse desde la raíz del repositorio. Se recomienda utilizar un entorno virtual independiente:

```powershell
py -m venv .venv_modelos
.\.venv_modelos\Scripts\Activate.ps1
python -m pip install -r scripts/modelos_produccion_lechera/basicos/requirements.txt
```

Para entrenar y evaluar cada modelo básico:

```powershell
python scripts/modelos_produccion_lechera/basicos/modelo_mediana.py
python scripts/modelos_produccion_lechera/basicos/modelo_ridge.py
python scripts/modelos_produccion_lechera/basicos/modelo_random_forest.py
```

Cada ejecución muestra las métricas en la terminal y guarda en `reports/modelos_basicos` un CSV de métricas, un CSV con las predicciones de validación y un archivo `.joblib` con el modelo entrenado. Ridge también genera `coeficientes_ridge.csv` para facilitar la interpretación de sus variables.

## Próximos pasos

- realizar búsqueda de hiperparámetros sobre los modelos seleccionados;
- incorporar y evaluar un modelo CatBoost;
- comparar los candidatos con validación y conservar el conjunto de prueba para la evaluación final;
- analizar errores por tipo de establecimiento, volumen de producción y territorio.

## Uso de inteligencia artificial

El uso de inteligencia artificial generativa en tareas de apoyo técnico y documental se detalla en [registro_uso_IA.md](registro_uso_IA.md). La validación de datos, ejecución de scripts, análisis de resultados y decisiones metodológicas son responsabilidad del equipo del proyecto.
