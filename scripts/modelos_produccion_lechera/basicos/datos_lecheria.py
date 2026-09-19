"""Funciones sencillas para preparar los datos de los modelos básicos."""

# Path permite construir rutas que funcionan aunque el script se ejecute desde otra carpeta.
from pathlib import Path

# NumPy se usa para cálculos numéricos pequeños, como la raíz del error cuadrático.
import numpy as np

# Pandas se usa para leer, unir y transformar los archivos CSV.
import pandas as pd

# Estas funciones calculan las métricas de regresión que se guardarán al final.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Esta función permite crear las divisiones aleatorias de los datos.
from sklearn.model_selection import train_test_split


# Se obtiene la ruta del archivo actual.
RUTA_SCRIPT = Path(__file__).resolve()

# La carpeta PAA está tres niveles por encima de este archivo.
CARPETA_PAA = RUTA_SCRIPT.parents[3]

# Los CSV completos e imputados están dentro de PAA/data.
CARPETA_DATOS = CARPETA_PAA / "data"

# Los resultados de estos modelos se guardarán en una carpeta separada.
CARPETA_RESULTADOS = CARPETA_PAA / "reports" / "modelos_basicos"

# El código 2 identifica la especialización lechera utilizada en el proyecto.
CODIGO_ESPECIALIZACION_LECHERA = 2

# El código 3 identifica el tipo de producción lechera utilizado en el proyecto.
CODIGO_TIPO_PRODUCCION_LECHERA = 3

# El código 11 identifica la especie bovina de leche.
CODIGO_ESPECIE_LECHERA = 11

# Los datos anteriores a 2016 quedan fuera del alcance actual.
ANIO_INICIAL = 2016

# Esta semilla hace que la primera división aleatoria siempre sea igual.
SEMILLA_PRINCIPAL = 42

# Esta segunda semilla hace reproducible la separación entre entrenamiento y validación.
SEMILLA_VALIDACION = 43

# El objetivo que queremos predecir es la cantidad total de litros.
OBJETIVO = "Litros"

# Estas columnas definen un perfil productivo dentro de un ejercicio.
CLAVES_UNION = [
    "Anio",
    "DepartamentoCodigo",
    "SeccionalPolicialCodigo",
    "AreaSupervision",
    "AreaEnumeracion",
    "GiroCodigo",
    "NaturalezaJuridicaCodigo",
    "EstratoCodigo",
]

# Estas variables se tratarán como categorías en Random Forest.
VARIABLES_CATEGORICAS = [
    "DepartamentoCodigo",
    "GiroCodigo",
    "NaturalezaJuridicaCodigo",
    "EstratoCodigo",
]

# Estas variables se tratarán como cantidades numéricas.
VARIABLES_NUMERICAS = [
    "total_bovinos_leche",
    "toros",
    "vacas_ordene",
    "vacas_secas",
    "vaquillonas_mas_2",
    "vaquillonas_1_2",
    "terneros_menos_1",
    "terneras_menos_1",
    "Superficie",
    "UnidadesGanaderas",
    "SuperficieGanadera",
    "carga_ganadera",
]

# Esta lista reúne todas las variables que reciben los modelos.
VARIABLES_MODELO = VARIABLES_CATEGORICAS + VARIABLES_NUMERICAS

# Cada código de categoría animal se convierte en una columna fácil de interpretar.
NOMBRES_CATEGORIAS = {
    1: "toros",
    2: "vacas_ordene",
    3: "vacas_secas",
    4: "vaquillonas_mas_2",
    5: "vaquillonas_1_2",
    6: "terneros_menos_1",
    7: "terneras_menos_1",
}

# Estos textos se interpretan como valores faltantes al leer los CSV.
VALORES_FALTANTES = ["-", "", " ", "NULL", "null", "NaN", "nan", "********"]

# Las columnas necesarias se declaran para no cargar información que no usa el modelo.
COLUMNAS_NECESARIAS = {
    "DatosProduccionLeche.csv": CLAVES_UNION
    + [
        "EspecieCodigo",
        "EspecializacionMGAPCodigo",
        "TipoProduccionMGAPCodigo",
        "Litros",
    ],
    "DatosAnimales.csv": CLAVES_UNION
    + [
        "EspecieCodigo",
        "EspecializacionMGAPCodigo",
        "TipoProduccionMGAPCodigo",
        "AnimalesPorEspeciePD_AD",
    ],
    "DatosAnimalesDetallados.csv": CLAVES_UNION
    + [
        "EspecieCodigo",
        "CategoriaCodigo",
        "EspecializacionMGAPCodigo",
        "TipoProduccionMGAPCodigo",
        "AnimalesPorCategoriaPD_AD",
    ],
    "DatosGenerales.csv": CLAVES_UNION
    + [
        "EspecializacionMGAPCodigo",
        "TipoProduccionMGAPCodigo",
        "Superficie",
        "UnidadesGanaderas",
        "SuperficieGanadera",
    ],
}

# Estos códigos se leen como texto para conservar su representación original.
COLUMNAS_CODIGO_TEXTO = [
    "DepartamentoCodigo",
    "SeccionalPolicialCodigo",
    "AreaSupervision",
    "AreaEnumeracion",
    "GiroCodigo",
    "NaturalezaJuridicaCodigo",
    "EstratoCodigo",
]


def convertir_a_numero(datos, columnas):
    """Convierte las columnas indicadas a números y deja NaN si algo no es válido."""

    # Se recorre una columna por vez para que el proceso sea fácil de seguir.
    for columna in columnas:
        # Se convierte temporalmente a texto para aceptar una coma decimal residual.
        texto = datos[columna].astype("string").str.replace(",", ".", regex=False)

        # Los valores que no se puedan convertir quedan marcados como faltantes.
        datos[columna] = pd.to_numeric(texto, errors="coerce")


def leer_tabla(nombre_archivo):
    """Lee una tabla y excluye las columnas de confianza de la imputación MGAP."""

    # Se construye la ruta completa del CSV solicitado.
    ruta = CARPETA_DATOS / nombre_archivo

    # Se detiene el proceso con un mensaje claro si falta algún archivo.
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    # Primero se lee solamente la cabecera para revisar las columnas disponibles.
    cabecera = pd.read_csv(ruta, sep=";", encoding="utf-8-sig", nrows=0).columns.tolist()

    # Se identifican explícitamente todas las columnas cuyo nombre contiene confianza.
    columnas_confianza = [columna for columna in cabecera if "confianza" in columna.lower()]

    # Se obtiene la lista exacta de columnas que necesita esta tabla.
    columnas_a_leer = COLUMNAS_NECESARIAS[nombre_archivo]

    # Se comprueba que el archivo tenga todas las columnas necesarias.
    columnas_faltantes = sorted(set(columnas_a_leer) - set(cabecera))

    # Se informa el problema antes de intentar entrenar un modelo incompleto.
    if columnas_faltantes:
        raise ValueError(f"{nombre_archivo} no contiene: {columnas_faltantes}")

    # Se crea un tipo de dato de texto solamente para los códigos presentes en la tabla.
    tipos = {
        columna: "string"
        for columna in COLUMNAS_CODIGO_TEXTO
        if columna in columnas_a_leer
    }

    # Se lee únicamente lo necesario; las columnas de confianza no entran al DataFrame.
    datos = pd.read_csv(
        ruta,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
        na_values=VALORES_FALTANTES,
        keep_default_na=True,
        low_memory=False,
        usecols=columnas_a_leer,
        dtype=tipos,
    )

    # Se muestran las columnas de confianza excluidas para que la decisión quede visible.
    print(f"[carga] {nombre_archivo}: confianzas excluidas={columnas_confianza}")

    # Estas columnas se necesitan como números para aplicar los filtros del proyecto.
    columnas_selectoras = [
        "Anio",
        "EspecieCodigo",
        "CategoriaCodigo",
        "EspecializacionMGAPCodigo",
        "TipoProduccionMGAPCodigo",
    ]

    # Se convierten solamente las columnas selectoras que existen en la tabla actual.
    for columna in columnas_selectoras:
        if columna in datos.columns:
            datos[columna] = pd.to_numeric(datos[columna], errors="coerce").astype("Int64")

    # Se devuelve la tabla lista para ser filtrada.
    return datos


def filtrar_lecheria(datos, filtrar_especie=False):
    """Conserva los registros lecheros desde 2016."""

    # Se conservan todos los ejercicios disponibles desde 2016 en adelante.
    resultado = datos.loc[datos["Anio"].ge(ANIO_INICIAL)].copy()

    # Se seleccionan los registros clasificados con especialización lechera.
    resultado = resultado.loc[
        resultado["EspecializacionMGAPCodigo"].eq(CODIGO_ESPECIALIZACION_LECHERA)
    ].copy()

    # Se seleccionan los registros cuyo tipo de producción MGAP es lechero.
    resultado = resultado.loc[
        resultado["TipoProduccionMGAPCodigo"].eq(CODIGO_TIPO_PRODUCCION_LECHERA)
    ].copy()

    # Algunas tablas contienen varias especies y necesitan un filtro adicional.
    if filtrar_especie:
        resultado = resultado.loc[
            resultado["EspecieCodigo"].eq(CODIGO_ESPECIE_LECHERA)
        ].copy()

    # Se devuelve una copia para evitar modificar accidentalmente la tabla original.
    return resultado


def construir_dataset():
    """Une las cuatro tablas utilizadas para predecir litros."""

    # Se carga la tabla que contiene la cantidad de litros.
    produccion = leer_tabla("DatosProduccionLeche.csv")

    # Se carga la tabla que contiene el total de animales por especie.
    animales = leer_tabla("DatosAnimales.csv")

    # Se carga la tabla que separa los animales por categoría.
    detalle = leer_tabla("DatosAnimalesDetallados.csv")

    # Se carga la tabla que contiene superficie y unidades ganaderas.
    generales = leer_tabla("DatosGenerales.csv")

    # Se aplican los filtros MGAP y de especie a la tabla de producción.
    produccion = filtrar_lecheria(produccion, filtrar_especie=True)

    # Se aplican los mismos filtros a la tabla de animales.
    animales = filtrar_lecheria(animales, filtrar_especie=True)

    # Se aplican los mismos filtros a la tabla de animales detallados.
    detalle = filtrar_lecheria(detalle, filtrar_especie=True)

    # La tabla general no tiene especie, por eso solo usa los filtros MGAP.
    generales = filtrar_lecheria(generales, filtrar_especie=False)

    # La variable objetivo se convierte a número.
    convertir_a_numero(produccion, [OBJETIVO])

    # Los litros se suman para obtener un único total por perfil y ejercicio.
    objetivo = (
        produccion.groupby(CLAVES_UNION, dropna=False)[OBJETIVO]
        .sum(min_count=1)
        .reset_index()
    )

    # La cantidad total de bovinos lecheros se convierte a número.
    convertir_a_numero(animales, ["AnimalesPorEspeciePD_AD"])

    # Los animales se suman para obtener un total por perfil y ejercicio.
    animales = (
        animales.groupby(CLAVES_UNION, dropna=False)["AnimalesPorEspeciePD_AD"]
        .sum(min_count=1)
        .reset_index(name="total_bovinos_leche")
    )

    # La cantidad de animales por categoría se convierte a número.
    convertir_a_numero(detalle, ["AnimalesPorCategoriaPD_AD"])

    # Se conservan únicamente las siete categorías bovinas contempladas.
    detalle = detalle.loc[detalle["CategoriaCodigo"].isin(NOMBRES_CATEGORIAS)].copy()

    # Se suma cada categoría dentro de cada perfil.
    detalle = (
        detalle.groupby(CLAVES_UNION + ["CategoriaCodigo"], dropna=False)[
            "AnimalesPorCategoriaPD_AD"
        ]
        .sum(min_count=1)
        .reset_index()
    )

    # Las categorías que estaban en filas se transforman en columnas numéricas.
    detalle = detalle.pivot(
        index=CLAVES_UNION,
        columns="CategoriaCodigo",
        values="AnimalesPorCategoriaPD_AD",
    )

    # Se crean también las categorías que eventualmente no aparezcan en los datos.
    detalle = detalle.reindex(columns=list(NOMBRES_CATEGORIAS), fill_value=0)

    # Si una categoría no aparece en un perfil existente, su cantidad correcta es cero.
    detalle = detalle.fillna(0)

    # Los códigos de categoría se reemplazan por nombres comprensibles.
    detalle = detalle.rename(columns=NOMBRES_CATEGORIAS).reset_index()

    # Se quita el nombre técnico que Pandas deja sobre las columnas del pivote.
    detalle.columns.name = None

    # Las tres medidas generales se convierten a valores numéricos.
    convertir_a_numero(
        generales,
        ["Superficie", "UnidadesGanaderas", "SuperficieGanadera"],
    )

    # Se conserva una sola fila general por perfil.
    generales = generales[
        CLAVES_UNION + ["Superficie", "UnidadesGanaderas", "SuperficieGanadera"]
    ].drop_duplicates(subset=CLAVES_UNION, keep="first")

    # Se unen los litros con el total de animales.
    dataset = objetivo.merge(animales, on=CLAVES_UNION, how="inner", validate="one_to_one")

    # Se agregan las cantidades por categoría animal.
    dataset = dataset.merge(detalle, on=CLAVES_UNION, how="inner", validate="one_to_one")

    # Se agregan las variables de superficie y unidades ganaderas.
    dataset = dataset.merge(generales, on=CLAVES_UNION, how="inner", validate="one_to_one")

    # La carga ganadera solo se calcula cuando la superficie ganadera es positiva.
    dataset["carga_ganadera"] = np.where(
        dataset["SuperficieGanadera"].gt(0),
        dataset["UnidadesGanaderas"] / dataset["SuperficieGanadera"],
        np.nan,
    )

    # Las categorías faltantes reciben una etiqueta explícita.
    for columna in VARIABLES_CATEGORICAS:
        dataset[columna] = dataset[columna].fillna("SIN_DATO").astype(str)

    # Las variables numéricas se convierten otra vez para asegurar un formato uniforme.
    convertir_a_numero(dataset, VARIABLES_NUMERICAS + [OBJETIVO])

    # Las categorías ya están dentro de las claves, por eso no se agregan una segunda vez.
    columnas_salida = CLAVES_UNION + VARIABLES_NUMERICAS + [OBJETIVO]

    # Se conservan las claves para identificar predicciones, las variables y el objetivo.
    dataset = dataset[columnas_salida].copy()

    # Se eliminan posibles copias exactas del mismo perfil.
    dataset = dataset.drop_duplicates(subset=CLAVES_UNION, keep="first")

    # Se eliminan filas sin objetivo porque no sirven para aprendizaje supervisado.
    dataset = dataset.dropna(subset=[OBJETIVO])

    # Se eliminan valores nulos o negativos de litros porque no son objetivos válidos.
    dataset = dataset.loc[dataset[OBJETIVO].gt(0)].copy()

    # Se ordenan las filas para que el archivo de predicciones sea fácil de revisar.
    dataset = dataset.sort_values(CLAVES_UNION).reset_index(drop=True)

    # Se detiene el proceso si las uniones no produjeron ninguna fila.
    if dataset.empty:
        raise ValueError("El dataset final quedó vacío después de aplicar los filtros.")

    # Se muestra el rango de años realmente presente después de todas las uniones.
    print(
        f"[dataset] filas={len(dataset)}, "
        f"años={int(dataset['Anio'].min())}-{int(dataset['Anio'].max())}"
    )

    # Se devuelve el dataset analítico completo.
    return dataset


def dividir_dataset(dataset):
    """Crea una división aleatoria 70/15/15 y mantiene el test sin usar."""

    # Primero se aparta el 15 % que quedará reservado como test final.
    desarrollo, prueba = train_test_split(
        dataset,
        test_size=0.15,
        random_state=SEMILLA_PRINCIPAL,
        shuffle=True,
    )

    # Este valor convierte el 15 % total en una fracción del 85 % restante.
    proporcion_validacion = 0.15 / 0.85

    # El 85 % restante se divide en 70 % de entrenamiento y 15 % de validación.
    entrenamiento, validacion = train_test_split(
        desarrollo,
        test_size=proporcion_validacion,
        random_state=SEMILLA_VALIDACION,
        shuffle=True,
    )

    # Se informa el tamaño de cada partición para poder comprobar la división.
    print(
        f"[división] entrenamiento={len(entrenamiento)}, "
        f"validación={len(validacion)}, test_reservado={len(prueba)}"
    )

    # Se reinician los índices sin cambiar las filas seleccionadas.
    entrenamiento = entrenamiento.reset_index(drop=True)

    # Se reinician también los índices de validación.
    validacion = validacion.reset_index(drop=True)

    # Se reinician los índices del test reservado.
    prueba = prueba.reset_index(drop=True)

    # Se devuelven las tres particiones; los scripts básicos no evaluarán el test.
    return entrenamiento, validacion, prueba


def calcular_metricas(nombre_modelo, valores_reales, predicciones):
    """Calcula un grupo pequeño de métricas fáciles de interpretar."""

    # Los valores reales se convierten a un arreglo numérico.
    reales = np.asarray(valores_reales, dtype=float)

    # Las predicciones se convierten al mismo formato.
    predichos = np.asarray(predicciones, dtype=float)

    # MAE expresa el error absoluto promedio directamente en litros.
    mae = mean_absolute_error(reales, predichos)

    # RMSE da mayor peso a los errores grandes.
    rmse = np.sqrt(mean_squared_error(reales, predichos))

    # R² resume qué proporción de la variabilidad logra explicar el modelo.
    r2 = r2_score(reales, predichos)

    # WAPE compara la suma de errores absolutos con la producción total observada.
    wape = np.abs(reales - predichos).sum() / reales.sum() * 100

    # Se devuelve una fila lista para convertirse en CSV.
    return {
        "modelo": nombre_modelo,
        "particion": "validacion",
        "filas": len(reales),
        "mae_litros": mae,
        "rmse_litros": rmse,
        "r2": r2,
        "wape_porcentaje": wape,
    }


def guardar_resultados(nombre_archivo, validacion, predicciones, metricas):
    """Guarda las métricas y las predicciones del conjunto de validación."""

    # La carpeta se crea solamente si todavía no existe.
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Se construye la ruta del archivo de métricas.
    ruta_metricas = CARPETA_RESULTADOS / f"metricas_{nombre_archivo}.csv"

    # Una lista con un diccionario se convierte en una tabla de una fila.
    pd.DataFrame([metricas]).to_csv(ruta_metricas, index=False, encoding="utf-8-sig")

    # Las claves del perfil permiten identificar cada predicción guardada.
    tabla_predicciones = validacion[CLAVES_UNION].copy()

    # Se agrega el valor real para poder comparar.
    tabla_predicciones["Litros_reales"] = validacion[OBJETIVO].to_numpy()

    # Se agrega la predicción producida por el modelo.
    tabla_predicciones["Litros_predichos"] = predicciones

    # Se calcula el error con signo para detectar sobreestimaciones y subestimaciones.
    tabla_predicciones["Error"] = (
        tabla_predicciones["Litros_predichos"] - tabla_predicciones["Litros_reales"]
    )

    # Se construye la ruta del archivo de predicciones.
    ruta_predicciones = CARPETA_RESULTADOS / f"predicciones_{nombre_archivo}.csv"

    # Se guarda una fila por registro de validación.
    tabla_predicciones.to_csv(ruta_predicciones, index=False, encoding="utf-8-sig")

    # Se devuelven las rutas para mostrarlas al terminar.
    return ruta_metricas, ruta_predicciones
