"""Diagnóstico descriptivo de PAA, sin modificar el proyecto de ingeniería.

Lee observaciones EXCLUSIVAMENTE de data/processed. Los CSV de data/raw se
usan como documentación, no como observaciones adicionales. SNIG se filtra
solo por los departamentos del catálogo de estaciones INIA: no se filtran
años, especies ni áreas de enumeración (se conserva expresamente el código 0).

No integra tablas, no agrega a la unidad de estudio, no imputa y no entrena.
Dependencia: pandas. Ejemplo desde la raíz del proyecto:
    python PAA/diagnostico_datos/diagnostico.py
Para repetir sin sobrescribir una ejecución anterior:
    python PAA/diagnostico_datos/diagnostico.py --salida resultados_2
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PROCESSED = ROOT / "data/processed"
META_SNIG = ROOT / "data/raw/SNIG/2025/metadatos/Metadatos"
CENTRALES = ("DatosGenerales", "DatosAnimales", "DatosAnimalesDetallados")

# Solo los catálogos que permiten interpretar códigos de las tres centrales.
# El valor es el nombre REAL del archivo de metadatos, que no siempre coincide.
CATALOGOS = {
    "Departamentos": "departamentos.csv",
    "Actividades": "actividades.csv",
    "Giros": "giros.csv",
    "NaturalezasJuridicas": "naturalezasjuridicas.csv",
    "Estratos": "estratos.csv",
    "EspecializacionesMGAP": "especializacionesMGAP.csv",
    "TiposProduccionMGAP": "tiposproduccionesMGAP.csv",
    "Especies": "especies.csv",
    "Categorias": "categorias.csv",
}

# Leer como texto evita que pandas convierta automáticamente códigos en NaN.
# Un guion o un cero NO pertenecen a esta lista y se cuentan por separado.
MARCADORES_NULOS = {"na", "n/a", "nan", "null", "none", "<na>"}
METRICAS = ("n_registros", "n_vacios", "n_marcadores_nulo", "n_ceros", "n_guiones")
CLAVE_DETALLE = ("fuente", "tabla", "Anio", "DepartamentoCodigo", "EstacionAgr", "columna")


def relativo(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def leer_csv(path: Path, **kwargs):
    """No normaliza ni cambia los datos fuente; preserva los valores textuales."""
    sep = ";" if "SNIG" in path.parts else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str,
                       na_filter=False, keep_default_na=False, **kwargs)


def huella(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for bloque in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def texto_limpio(texto: str) -> str:
    """Solo para documentación de salida: unifica espacios y saltos escapados."""
    texto = texto.replace("\\r\\n", " ")
    return " ".join(unicodedata.normalize("NFC", texto).split())


def reparar_codificacion(texto: str) -> str:
    """Algunos metadatos mezclan filas CP850 y CP1252 en el MISMO archivo.

    Se lee primero CP1252 y solo se recodifican celdas con señales inequívocas
    de CP850 mal interpretado (por ejemplo 'C¢digo'). No se aplica al dataset.
    Una celda correcta CP1252 ('En función...') se conserva. No se corrigen
    errores semánticos ni de redacción del proveedor.
    """
    sospechosos = set("¢£µ¤‚\u00a0")
    # El byte CP850 de í aparece como ¡ dentro de palabras en CP1252.
    def puntuacion(s: str) -> int:
        return sum(c in sospechosos for c in s) + len(re.findall(r"\w¡\w", s))

    if puntuacion(texto):
        try:
            candidato = texto.encode("cp1252").decode("cp850")
            if puntuacion(candidato) < puntuacion(texto) and not any(
                "\u2500" <= c <= "\u259f" for c in candidato
            ):
                texto = candidato
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return texto_limpio(texto)


def metadatos_snig(path: Path) -> dict[str, dict]:
    contenido = path.read_bytes()
    try:
        contenido = contenido.decode("utf-8-sig")
    except UnicodeDecodeError:
        contenido = contenido.decode("cp1252")
    filas = csv.DictReader(io.StringIO(contenido, newline=None), delimiter=";")
    resultado = {}
    for fila in filas:
        fila = {k.strip(): reparar_codificacion(v or "") for k, v in fila.items() if k}
        variable = fila.get("VARIABLE", "")
        if variable:
            assert variable not in resultado, (path, variable)
            resultado[variable] = fila
    return resultado


def metadatos_inia(paths: list[Path]) -> dict[str, str]:
    """El pie alterna nombre y definición; csv.reader respeta campos citados.

    Se comprueban los seis pies completos, no se asume que son iguales.
    Cualquier diferencia detiene el script para evitar asignar una definición
    de una estación a otra sin revisar su alcance.
    """
    referencias = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            filas = list(csv.reader(stream))
        inicio = next(i for i, f in enumerate(filas) if f and
                      texto_limpio(f[0]) == "Definición de las variables listadas")
        celdas = []
        for fila in filas[inicio + 1:]:
            if not fila or not any(c.strip() for c in fila):
                continue
            assert len(fila) == 1, f"Pie INIA inesperado: {path}"
            celdas.append(texto_limpio(fila[0]))
        assert len(celdas) % 2 == 0, f"Definición incompleta: {path}"
        definiciones = dict(zip(celdas[0::2], celdas[1::2]))
        assert len(definiciones) * 2 == len(celdas), "Variables repetidas en pie INIA"
        referencias.append(definiciones)
    assert referencias and all(x == referencias[0] for x in referencias), "Pies INIA diferentes"
    return referencias[0]


def mapa_nombres_inia() -> dict[str, str]:
    """Lee el literal del ETL sin importarlo ni ejecutar cargas de bases de datos."""
    arbol = ast.parse((ROOT / "scripts/etl_inia.py").read_text(encoding="utf-8-sig"))
    for nodo in arbol.body:
        if isinstance(nodo, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "RENOMBRAR_COLUMNAS" for t in nodo.targets
        ):
            mapa = ast.literal_eval(nodo.value)
            assert len(set(mapa.values())) == len(mapa)
            return {destino: texto_limpio(origen) for origen, destino in mapa.items()}
    raise ValueError("No se encontró RENOMBRAR_COLUMNAS en el ETL de INIA")


def interpretacion(fuente: str, columna: str) -> str:
    if fuente == "INIA" and columna.startswith("GradosMenos"):
        return ("Vacío potencialmente estructural: el metadato indica que no se calcula "
                "si la media diaria es inferior a la base. El CSV no permite atribuir "
                "cada vacío a esa causa. No imputado.")
    if fuente == "SNIG" and columna == "AreaEnumeracion":
        return "El código 0 se conserva y no se cuenta como faltante; no se presupone que sea un agregado."
    if fuente == "SNIG" and columna in {"EspecializacionMGAPCodigo", "TipoProduccionMGAPCodigo"}:
        return "Vacíos contabilizados; el metadato 2025 no documenta su disponibilidad en cada ejercicio."
    if fuente == "SNIG" and columna == "AnimalesPorEspecieNacimientosPD_AD":
        return "El metadato indica disponibilidad solo para bovinos; cero en otras especies no demuestra ausencia de nacimientos."
    if fuente == "INIA" and columna in {"HeladaAgromet", "HeladaMeteo", "Llovio", "TMin15"}:
        return "Indicador 1=sí, 0=no; el cero es un dato presente, no un nulo."
    return "Vacíos y marcadores explícitos contabilizados; no se asigna una causa no documentada."


def unidad_inia(nombre: str) -> str:
    if "S:1/N:0" in nombre:
        return "Indicador: 1=sí; 0=no"
    if nombre.startswith("Grados Dias"):
        return "Grados-día sobre la base positiva indicada (encabezado fuente: ºC)"
    if "cal/cm²" in nombre:
        return "cal/cm²"
    if "km/dia" in nombre:
        return "km/día"
    if any(x in nombre for x in ("(Horas)", "(Hrs)", "(Hrs.)")):
        return "Horas"
    if "(ºC)" in nombre:
        return "ºC"
    if "(mm)" in nombre:
        return "mm"
    if "(%)" in nombre:
        return "%"
    if "Richardson" in nombre:
        return "Unidades de frío Richardson"
    if "Arroz" in nombre:
        return "Unidades térmicas de arroz (base 10 ºC y topes según definición)"
    raise ValueError(f"Unidad INIA sin documentar: {nombre}")


def construir_diccionario(tablas: list[tuple[str, str, Path]], pies: dict,
                           raw_inia: list[Path]) -> pd.DataFrame:
    """Una fila por columna REAL de cada tabla; falla si alguna queda sin definir."""
    renombres = mapa_nombres_inia()
    filas = []
    for fuente, tabla, path in tablas:
        columnas = leer_csv(path, nrows=0).columns
        es_snig = fuente == "SNIG"
        meta_path = META_SNIG / (CATALOGOS[tabla] if tabla in CATALOGOS else tabla.lower() + ".csv") if es_snig else None
        meta = metadatos_snig(meta_path) if meta_path else {}
        for posicion, columna in enumerate(columnas, 1):
            r = dict(fuente=fuente, tabla=tabla, archivo_datos=relativo(path),
                     orden_columna=posicion, columna=columna, variable_original=columna,
                     representa="", unidad_o_codificacion="", origen="",
                     limitantes_metadato="", notas_metadato="", observacion_paa="",
                     fuente_definicion="", estado_definicion="Documentada en metadatos")
            if es_snig and columna != "ID":
                original = "Ejercicio" if columna == "Anio" else columna
                m = meta[original]  # KeyError intencional si falta documentación.
                r.update(variable_original=original, representa=m["DESCRIPCION"],
                         origen=m["ORIGEN"], limitantes_metadato=m["LIMITANTES CONOCIDAS"],
                         notas_metadato=m["NOTAS"], fuente_definicion=relativo(meta_path))
                if columna == "Anio":
                    r["observacion_paa"] = "Ejercicio ganadero, renombrado Anio por el ETL; no asumir año calendario climático."
                    r["fuente_definicion"] += " | scripts/etl_snig_datos.py (renombre)"
                elif columna == "AreaEnumeracion":
                    r["observacion_paa"] = interpretacion("SNIG", columna)
                elif columna in {"AnimalesPorCategoriaPD_AD", "AnimalesPorEspeciePD_AD"}:
                    r["observacion_paa"] = "Propios dentro + ajenos dentro; no es un momento temporal."
                elif columna in {"AnimalesPorEspeciePD_PF", "AnimalesPorCategoriaPD_PF"}:
                    r["observacion_paa"] = "Propios dentro + propios fuera; no es un momento temporal."
                if columna.endswith("Codigo") or columna in {"AreaEnumeracion", "AreaSupervision"}:
                    r["unidad_o_codificacion"] = "Código; no es una magnitud continua"
                elif columna in {"Superficie", "SuperficieGanadera"}:
                    r["unidad_o_codificacion"] = "Hectáreas (SuperficieGanadera remite a Superficie)"
                elif columna == "UnidadesGanaderas":
                    r["unidad_o_codificacion"] = "Unidades ganaderas"
                elif columna == "CategoriaPonderacionUG":
                    r["unidad_o_codificacion"] = "Factor de conversión a unidades ganaderas"
                elif columna.startswith("AnimalesPor"):
                    r["unidad_o_codificacion"] = "Cabezas / animales"
                elif columna.startswith("CantidadTenedores"):
                    r["unidad_o_codificacion"] = "Cantidad de tenedores/productores según definición"
                elif columna == "Anio":
                    r["unidad_o_codificacion"] = "Ejercicio ganadero"
                else:
                    r["unidad_o_codificacion"] = "Texto descriptivo"
            elif es_snig:  # ID añadido por el proyecto anterior, no por el metadato.
                r.update(representa="Identificador técnico de la fila generado por el ETL; no identifica por sí solo un establecimiento.",
                         unidad_o_codificacion="Identificador", origen="ETL del proyecto de ingeniería",
                         fuente_definicion="scripts/etl_snig_datos.py", estado_definicion="Derivada; documentada en código")
            elif tabla == "INIA" and columna in renombres:
                original = renombres[columna]
                r.update(variable_original=original, representa=pies[original],
                         unidad_o_codificacion=unidad_inia(original), origen="INIA: pie del CSV de cada estación",
                         fuente_definicion=" | ".join(relativo(p) for p in raw_inia) + " | scripts/etl_inia.py (renombre)",
                         observacion_paa=interpretacion("INIA", columna))
                if columna.startswith("GradosMenos"):
                    r["observacion_paa"] += " El nombre processed puede confundir: son grados-día sobre una base POSITIVA, no grados por debajo de cero."
                elif columna == "TempAireMedia":
                    r["observacion_paa"] += " El pie no detalla su fórmula; no equiparar sin comprobación con TMedia."
            else:
                r.update(descripcion_auxiliar(tabla, columna))
            assert r["representa"] and r["fuente_definicion"], (tabla, columna)
            filas.append(r)
    resultado = pd.DataFrame(filas)
    assert not resultado.duplicated(["fuente", "tabla", "columna"]).any()
    return resultado


def descripcion_auxiliar(tabla: str, columna: str) -> dict:
    """Proveniencia explícita para columnas sin ficha de metadatos propia.

    PAD se incluye solo en el diccionario por formar parte de la propuesta.
    No se inventa su definición física ni se ejecuta aquí un diagnóstico PAD.
    """
    etl = "scripts/etl_inia.py" if tabla in {"INIA", "EstacionAgr"} else "scripts/etl_pad.py"
    r = dict(origen="Transformación existente del proyecto de ingeniería", fuente_definicion=etl,
             estado_definicion="Derivada; documentada en código")
    if tabla == "INIA" and columna in {"Anio", "Mes", "Dia"}:
        r.update(variable_original="Fecha", representa={"Anio": "Año calendario", "Mes": "Mes", "Dia": "Día del mes"}[columna] + " de la fecha de observación diaria de INIA.", unidad_o_codificacion="Componente de fecha")
    elif tabla == "PAD" and columna in {"Anio", "Mes"}:
        r.update(variable_original="YYYY_MM0", representa=("Año" if columna == "Anio" else "Mes") + " de la observación mensual PAD, extraído del nombre del campo JSON.", unidad_o_codificacion="Componente de fecha")
    elif columna == "EstacionAgr" and tabla in {"INIA", "EstacionAgr"}:
        r.update(variable_original="Nombre del archivo INIA", representa="Nombre de la estación agrometeorológica; vincula cada observación con el catálogo EstacionAgr.", unidad_o_codificacion="Nombre/clave de estación", observacion_paa="Se conservan las grafías del catálogo processed, incluidos sus acentos y erratas.")
    elif tabla == "EstacionAgr" and columna == "DepartamentoCodigo":
        r.update(representa="Código del departamento asignado a la estación en el catálogo processed; define el filtro geográfico de SNIG.", unidad_o_codificacion="Código del catálogo Departamentos", fuente_definicion="data/processed/INIA/EstacionAgr.csv | data/processed/SNIG/Catalogos/Departamentos.csv", estado_definicion="Relación explícita en los catálogos processed")
    elif columna == "UbicacionCodigo" and tabla in {"PAD", "Ubicacion"}:
        r.update(representa="Identificador de ubicación generado por el ETL a partir de pares de coordenadas; permite relacionar PAD con Ubicacion.", unidad_o_codificacion="Identificador", observacion_paa="No es un código de área de enumeración SNIG.")
    elif tabla == "Ubicacion" and columna == "DepartamentoCodigo":
        r.update(representa="Departamento asignado por el ETL a la ubicación PAD mediante cruce geográfico; incluye resolución por proximidad de algunos casos no intersectados.", unidad_o_codificacion="Código del catálogo Departamentos", observacion_paa="Revisar el procedimiento de asignación geográfica antes de interpretar puntos próximos a límites.")
    elif tabla == "Ubicacion" and columna in {"Latitud", "Longitud"}:
        r.update(representa=columna + " de la ubicación PAD procedente del JSON original.", unidad_o_codificacion="Grados; ETL utiliza EPSG:4326")
    elif tabla == "PAD" and columna == "CapProm":
        r.update(variable_original="YYYY_MM0", representa="Promedio mensual del indicador hídrico PAD de una ubicación; procede del campo mensual terminado en 0, no de los campos decadales 1, 2 y 3.", unidad_o_codificacion="Unidad física no explicitada en los metadatos aportados; ETL valida rango 0–100", estado_definicion="Transformación documentada; definición física/unidad pendiente", observacion_paa="No se presenta como porcentaje confirmado solo por el rango. El ETL promedia claves mensuales duplicadas.")
    else:
        raise ValueError(f"Columna auxiliar sin definición: {tabla}.{columna}")
    return r


def agregar_perfil(df: pd.DataFrame, fuente: str, tabla: str, detalle: dict,
                   estaciones: dict[str, str]) -> None:
    """Acumula conteos por año y departamento (SNIG) o año y estación (INIA).

    Cada fila original tiene peso 1. No se deduplica ni se agregan animales:
    n_registros es cantidad de filas, no de tenedores ni de unidades de estudio.
    Los indicadores usan texto recortado solo para la comparación en memoria.
    """
    if df.empty:
        return
    departamento = df["DepartamentoCodigo"] if fuente == "SNIG" else df["EstacionAgr"].map(estaciones)
    estacion = pd.Series("", index=df.index) if fuente == "SNIG" else df["EstacionAgr"]
    assert departamento.notna().all(), "Hay estaciones sin departamento en el catálogo"
    for columna in df.columns:
        s = df[columna].str.strip()
        indicadores = pd.DataFrame({
            "Anio": df["Anio"], "DepartamentoCodigo": departamento, "EstacionAgr": estacion,
            "n_registros": 1,
            "n_vacios": s.eq("").astype("int64"),
            "n_marcadores_nulo": s.str.casefold().isin(MARCADORES_NULOS).astype("int64"),
            "n_ceros": s.str.fullmatch(r"[+-]?0+(?:[.,]0+)?").astype("int64"),
            "n_guiones": s.eq("-").astype("int64"),
        })
        grupos = indicadores.groupby(["Anio", "DepartamentoCodigo", "EstacionAgr"], sort=False)[list(METRICAS)].sum()
        for grupo, valores in grupos.iterrows():
            clave = (fuente, tabla, *grupo, columna)
            if clave not in detalle:
                detalle[clave] = Counter()
            detalle[clave].update({k: int(v) for k, v in valores.items()})


def completar_perfil(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_faltantes_csv"] = df["n_vacios"] + df["n_marcadores_nulo"]
    df["n_presentes"] = df["n_registros"] - df["n_faltantes_csv"]
    df["porcentaje_faltantes"] = (100 * df["n_faltantes_csv"] / df["n_registros"]).round(4)
    assert (df["n_presentes"] >= 0).all()
    assert (df["n_ceros"] + df["n_guiones"] <= df["n_presentes"]).all()
    return df


def tabla_md(df: pd.DataFrame) -> str:
    """Markdown sin depender de tabulate. Formato legible para el informe."""
    def celda(x):
        if isinstance(x, float):
            return f"{x:.2f}".replace(".", ",")
        return str(x).replace("|", "/").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join("---" for _ in df.columns) + " |",
        *("| " + " | ".join(celda(x) for x in fila) + " |" for fila in df.itertuples(index=False, name=None)),
    ])


def informe(snig: pd.DataFrame, inia: pd.DataFrame, detalle: pd.DataFrame,
            cobertura: pd.DataFrame, resumen: dict, diccionario: pd.DataFrame) -> str:
    partes = ["# PAA — Diagnóstico de calidad, adecuación, cobertura y limitaciones", "",
        "## 1. Alcance y criterio de conteo", "",
        "Se examinaron las tablas centrales DatosGenerales, DatosAnimales y DatosAnimalesDetallados de SNIG, "
        "leyendo data/processed y reteniendo únicamente los departamentos con estaciones en el catálogo "
        "INIA: Canelones (2), Colonia (4), Salto (15), Tacuarembó (18) y Treinta y Tres (19). "
        "No se filtró por año, especie ni área. AreaEnumeracion=0 se conserva como código y no se cuenta como faltante.", "",
        "La unidad de estudio prevista es **año–departamento–área de enumeración**. Sin embargo, los conteos "
        "de este diagnóstico son filas originales de cada tabla: una misma unidad puede aparecer varias veces "
        "por otras dimensiones, especies o categorías. No son cantidades de establecimientos ni de unidades únicas. "
        "No se suman las tres tablas como si fueran observaciones independientes.", "",
        "Para INIA se analizan todas las columnas y filas del CSV processed, sin restringir el período. "
        "PAD y los catálogos auxiliares se incluyen en el diccionario, no en un nuevo diagnóstico de faltantes.", "",
        "Faltante CSV = celda vacía (incluidos espacios) o marcador textual explícito, sin distinguir mayúsculas: "
        + ", ".join(f"`{x}`" for x in sorted(MARCADORES_NULOS)) + ". Los ceros y los guiones se contabilizan por separado y no se "
        "recodifican como nulos. El porcentaje usa las filas de la tabla después del filtro departamental "
        "para SNIG y todas las filas para INIA. En grados-día, un vacío puede ser estructural, por lo que "
        "el porcentaje no equivale automáticamente a pérdida de observaciones.", "",
        "## 2. Cobertura SNIG: registros por departamento", ""]
    pivot = cobertura.pivot(index="departamento", columns="tabla", values="n_registros").reindex(columns=CENTRALES).reset_index()
    partes += [tabla_md(pivot), "", "Control del filtro y conservación del código de área 0:", ""]
    control = pd.DataFrame([dict(tabla=t, registros_originales=v["originales"], registros_retenidos=v["retenidos"],
                                registros_excluidos=v["excluidos"], area_0_conservada=v["area_0_conservada"])
                            for t, v in resumen["snig"].items()])
    partes += [tabla_md(control), "",
        "Los volúmenes describen la cobertura de las tablas disponibles, no la cobertura porcentual del "
        "universo ganadero: no se dispone aquí de un denominador poblacional independiente. Las diferencias "
        "entre tablas también responden a sus dimensiones (por especie o categoría).", "",
        "## 3. SNIG: revisión de cada columna", ""]
    for tabla in CENTRALES:
        filas = snig[snig.tabla.eq(tabla)]
        partes += [f"### {tabla}", "", tabla_md(filas[["columna", "n_registros", "n_faltantes_csv", "porcentaje_faltantes", "n_ceros", "n_guiones"]]), ""]
        con_faltantes = filas[filas.n_faltantes_csv.gt(0)]
        if con_faltantes.empty:
            partes += ["No se encontraron celdas vacías ni los marcadores de nulo definidos. Esto no certifica exactitud semántica.", ""]
        else:
            partes += ["Columnas con faltantes: " + "; ".join(
                f"{r.columna}: {r.n_faltantes_csv} ({r.porcentaje_faltantes:.2f} %)" for r in con_faltantes.itertuples()) + ".", ""]
    mgap = detalle[detalle.fuente.eq("SNIG") & detalle.columna.isin(["EspecializacionMGAPCodigo", "TipoProduccionMGAPCodigo"])]
    mgap = mgap.groupby(["tabla", "Anio", "columna"], as_index=False)[["n_registros", "n_faltantes_csv"]].sum()
    mgap["porcentaje_faltantes"] = (100 * mgap.n_faltantes_csv / mgap.n_registros).round(4)
    patrones = []
    for (tabla, columna), grupo in mgap.groupby(["tabla", "columna"], sort=False):
        patrones.append(dict(tabla=tabla, columna=columna,
            ejercicios_totalmente_vacios=", ".join(grupo.loc[grupo.n_faltantes_csv.eq(grupo.n_registros), "Anio"]),
            ejercicios_parcialmente_vacios=", ".join(grupo.loc[grupo.n_faltantes_csv.gt(0) & grupo.n_faltantes_csv.lt(grupo.n_registros), "Anio"]),
            ejercicios_sin_vacios=", ".join(grupo.loc[grupo.n_faltantes_csv.eq(0), "Anio"])))
    partes += ["Un código presente no garantiza información interpretable: en ActividadCodigo, el guion "
               "está documentado como PRODUCTOR en el catálogo Actividades. En cambio, el catálogo Giros "
               "no contiene una definición para el guion observado en GiroCodigo. Esos casos quedan "
               "cuantificados en n_guiones, sin convertirlos a nulo ni asignarles un significado supuesto. "
               "Por tanto, «sin vacíos» no equivale a «sin limitaciones de información».", "",
               "Patrón temporal de las dos clasificaciones MGAP (observado en los archivos; "
               "no se presupone una causa ni una fecha oficial de incorporación):", "", tabla_md(pd.DataFrame(patrones)), "",
        "## 4. INIA: revisión de cada columna", "",
        tabla_md(inia[["columna", "n_registros", "n_faltantes_csv", "porcentaje_faltantes", "n_ceros"]]), "",
        "Las definiciones se contrastaron entre los pies de los seis CSV raw: coinciden. "
        "Hay 53 variables climáticas documentadas allí; processed conserva 47, más fecha descompuesta "
        "en tres columnas y nombre de estación (51 columnas en total). Las seis temperaturas a 20 cm "
        "no están en processed; no se reincorporaron ni se incluyen como columnas utilizables.", "",
        "En GradosMenos4p5, GradosMenos6, GradosMenos7, GradosMenos8, GradosMenos9, GradosMenos10, "
        "GradosMenos10p5 y GradosMenos12p8, el metadato establece que el valor queda vacío cuando "
        "la media diaria es inferior a la base positiva correspondiente. Se conservan los vacíos tal como "
        "están: los datos disponibles no identifican inequívocamente la causa de cada uno. El nombre "
        "GradosMenos no significa temperatura negativa ni grados por debajo de la base.", "",
        "Los ceros en Llovio, HeladaAgromet, HeladaMeteo y TMin15 significan que el evento no ocurrió. "
        "En UnTermicasArroz el metadato prescribe 0 por debajo de la base; no debe trasladarse ese criterio "
        "automáticamente a las columnas de grados-día, cuya convención es distinta.", "",
        "Los faltantes de temperatura de suelo no están distribuidos uniformemente entre estaciones. "
        "La siguiente tabla muestra el menor y el mayor porcentaje de vacíos entre las doce columnas "
        "de temperatura de suelo de cada estación (no es un porcentaje de filas con cualquier faltante):", ""]
    suelo = detalle[detalle.fuente.eq("INIA") & detalle.columna.str.startswith(("TempSC", "TempSD"))]
    suelo = suelo.groupby(["EstacionAgr", "columna"], as_index=False)[["n_registros", "n_faltantes_csv"]].sum()
    suelo["porcentaje"] = 100 * suelo.n_faltantes_csv / suelo.n_registros
    suelo = suelo.groupby("EstacionAgr").porcentaje.agg(["min", "max"]).reset_index()
    suelo = suelo.rename(columns={"min": "menor_porcentaje_vacios", "max": "mayor_porcentaje_vacios"})
    partes += [tabla_md(suelo), "",
        "Los metadatos describen estas temperaturas, pero no explican la causa específica de los vacíos "
        "de cada estación. No se supone una falla de sensor ni falta de instrumental sin documentación.", "",
        "Resumen de disponibilidad por estación (los faltantes por columna, estación y año están en detalle_faltantes.csv):", ""]
    fechas = pd.DataFrame(resumen["inia_estaciones"])
    partes += [tabla_md(fechas), "", "Registros por año en INIA:", "",
               tabla_md(pd.DataFrame(resumen["inia_anios"])), "",
        "La presencia de 2026 no se ocultó ni se eliminó. El período deberá alinearse con los ejercicios "
        "SNIG en una fase posterior; un año calendario de clima no equivale automáticamente a un ejercicio ganadero. "
        "Además, ausencia de una fila diaria y nulo dentro de una fila son problemas distintos: aquí se cuentan "
        "los nulos en filas existentes; no se inventan filas para días no presentes.", "",
        "## 5. Adecuación y limitaciones para la propuesta", "",
        "- SNIG aporta existencias y flujos ganaderos; INIA aporta observaciones climáticas de estaciones. "
        "La restricción departamental hace coincidir el ámbito geográfico disponible, pero una estación no "
        "representa necesariamente todas las áreas de enumeración de su departamento. Tacuarembó tiene dos estaciones.",
        "- DepartamentoCodigo de SNIG refiere al departamento de registro del tenedor. El metadato advierte "
        "que puede diferir del departamento físico del establecimiento o del área de enumeración. El código de "
        "área 0 permanece válido como código en este diagnóstico; no permite asumir una localización precisa.",
        "- Las tablas SNIG no están todavía a la unidad año–departamento–área de enumeración. Unirlas directamente "
        "sin tratar sus dimensiones podría multiplicar registros; no se hizo esa unión en esta entrega.",
        "- PD_AD = propios dentro + ajenos dentro; PD_PF = propios dentro + propios fuera. Son criterios de "
        "tenencia/localización, no stocks de inicio y fin de año. Su promedio no constituye un stock temporal.",
        "- Nacimientos está documentado solo para bovinos. Las existencias de suinos, caprinos y yeguarizos se "
        "incluyen como referencia y el metadato advierte que no se han realizado controles de calidad sobre ellas.",
        "- El estrato 0 y los casos de animales sin superficie declarada están contemplados en los metadatos. "
        "No se eliminaron ceros de superficie, animales ni códigos mediante reglas generales.",
        "- En Las Brujas cambió la obtención de heliofanía desde el 21/12/2020: se pasó de lectura de banda "
        "a estimación con un modelo basado en radiación de la estación automática. Puede afectar comparabilidad temporal.",
        "- RadSolar se calcula a partir de heliofanía mediante la fórmula de Angstrom. TMedia utiliza máxima "
        "y mínima, mientras que el metadato de TempAireMedia no especifica la fórmula; no se consideran "
        "automáticamente mediciones independientes o intercambiables.",
        "- PAD es mensual y tiene ubicaciones propias; UbicacionCodigo no es AreaEnumeracion. Su transformación "
        "se documentó a partir del ETL, pero la definición física y unidad de CapProm requieren una ficha fuente adicional.",
        "- El análisis de completitud no constituye validación de rangos físicos, exactitud, duplicados, consistencia "
        "de balances ni representatividad estadística. Esas comprobaciones no fueron solicitadas en este recorte.", "",
        "## 6. Preparación realizada y decisiones pendientes", "",
        "La única preparación de observaciones fue el filtro departamental de SNIG en memoria. Se "
        "conservaron todas las demás filas y valores, sin rellenar nulos, eliminar áreas, seleccionar especies, "
        "renombrar columnas fuente ni construir variables de modelado. La normalización de codificación y espacios "
        "se aplicó exclusivamente al texto de metadatos copiado al diccionario. No se exportó una base integrada.", "",
        "Antes del modelado corresponde acordar el ejercicio climático que se asociará a cada declaración, "
        "el tratamiento de variables MGAP incompletas y de los vacíos de INIA, y la agregación a la unidad de "
        "estudio. Este diagnóstico no toma esas decisiones de manera implícita.", "",
        "## 7. Cómo incorporarlo al entregable 2", "",
        "En la sección «Diagnóstico de calidad, adecuación, cobertura y limitaciones» del informe:", "",
        "1. Explicar el alcance y el filtro geográfico de la sección 1.",
        "2. Presentar la tabla de registros por departamento de la sección 2, aclarando que cuenta filas.",
        "3. Resumir los faltantes de las secciones 3 y 4, distinguiendo vacíos potencialmente estructurales de INIA.",
        "4. Incorporar las limitaciones de la sección 5 y delimitar la preparación realizada según la sección 6.",
        "5. Adjuntar los CSV columna por columna y el diccionario como anexos; usar detalle_faltantes.csv para "
        "consultar concentraciones por departamento/estación y ejercicio/año.", "",
        f"El diccionario tiene {len(diccionario)} filas, una por columna de cada una de las "
        f"{diccionario[['fuente', 'tabla']].drop_duplicates().shape[0]} tablas incluidas, "
        "con definición, unidad/codificación, limitaciones, notas y procedencia verificable. "
        "Incluye las tres centrales SNIG, sus nueve catálogos de códigos, INIA, EstacionAgr, PAD y Ubicacion. "
        "Las descripciones técnicas derivadas del ETL se distinguen de las definiciones del proveedor.", ""]
    return "\n".join(partes)


def guardar_csv(df: pd.DataFrame, path: Path) -> None:
    # BOM UTF-8 para tildes en Excel; ; no colisiona con las comas decimales.
    # mode='x' impide sobrescribir incluso si el archivo aparece durante la ejecución.
    df.to_csv(path, index=False, sep=";", decimal=",", encoding="utf-8-sig", mode="x")
    recuperado = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, na_filter=False)
    assert list(recuperado.columns) == list(df.columns) and len(recuperado) == len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", default="resultados", help="Nueva carpeta relativa al directorio de este script")
    parser.add_argument("--tamano-bloque", type=int, default=100000, help="Filas SNIG por bloque; afecta memoria, no resultados")
    args = parser.parse_args()
    salida = (BASE / args.salida).resolve()
    if not salida.is_relative_to(BASE) or salida == BASE:
        parser.error("La salida debe ser una subcarpeta de PAA/diagnostico_datos")
    if salida.exists():
        parser.error("La salida ya existe. Use --salida con otro nombre: no se sobrescriben archivos.")
    if args.tamano_bloque <= 0:
        parser.error("El tamaño de bloque debe ser positivo")

    tablas = [("SNIG", t, PROCESSED / "SNIG" / f"{t}.csv") for t in CENTRALES]
    tablas += [("SNIG", t, PROCESSED / "SNIG/Catalogos" / f"{t}.csv") for t in CATALOGOS]
    tablas += [(f, t, PROCESSED / f / f"{t}.csv") for f, t in
               [("INIA", "INIA"), ("INIA", "EstacionAgr"), ("PAD", "PAD"), ("PAD", "Ubicacion")]]
    raw_inia = sorted((ROOT / "data/raw/inia").glob("*.csv"))
    entradas = {p for _, _, p in tablas} | set(raw_inia)
    entradas |= {META_SNIG / f"{t.lower()}.csv" for t in CENTRALES}
    entradas |= {META_SNIG / f for f in CATALOGOS.values()}
    entradas |= {ROOT / "scripts" / f"etl_{nombre}.py" for nombre in ["inia", "pad", "snig_datos"]}
    huellas_antes = {relativo(p): huella(p) for p in sorted(entradas)}

    estaciones_df = leer_csv(PROCESSED / "INIA/EstacionAgr.csv")
    assert not estaciones_df.EstacionAgr.duplicated().any()
    estaciones = dict(zip(estaciones_df.EstacionAgr, estaciones_df.DepartamentoCodigo))
    departamentos = set(estaciones.values())
    assert "" not in departamentos
    cat_dept = leer_csv(PROCESSED / "SNIG/Catalogos/Departamentos.csv")
    nombres_dept = dict(zip(cat_dept.DepartamentoCodigo, cat_dept.DepartamentoDescripcion))
    assert departamentos <= nombres_dept.keys()
    # Este documento está redactado para la selección solicitada. Si cambia el
    # catálogo, detenerse evita producir un informe con nombres obsoletos.
    assert departamentos == {"2", "4", "15", "18", "19"}, "Revisar alcance: cambiaron los departamentos INIA"
    assert len(estaciones) == len(raw_inia) == 6
    pies = metadatos_inia(raw_inia)
    diccionario = construir_diccionario(tablas, pies, raw_inia)

    detalle_acumulado = {}
    resumen = {"snig": {}}
    for tabla in CENTRALES:
        print(f"Revisando {tabla}: filtro departamental y todas las columnas...", flush=True)
        originales = retenidos = areas_cero = 0
        ejercicios = set()
        path = PROCESSED / "SNIG" / f"{tabla}.csv"
        for bloque in leer_csv(path, chunksize=args.tamano_bloque):
            originales += len(bloque)
            filtrado = bloque.loc[bloque.DepartamentoCodigo.isin(departamentos)]
            retenidos += len(filtrado)
            areas_cero += int(filtrado.AreaEnumeracion.str.strip().eq("0").sum())
            ejercicios.update(filtrado.Anio.unique())
            agregar_perfil(filtrado, "SNIG", tabla, detalle_acumulado, estaciones)
        resumen["snig"][tabla] = dict(originales=originales, retenidos=retenidos,
            excluidos=originales-retenidos, area_0_conservada=areas_cero, ejercicios=sorted(ejercicios))
        print(f"  {retenidos:,} filas retenidas de {originales:,}; área 0: {areas_cero:,}.", flush=True)

    print("Revisando todas las columnas INIA y sus metadatos...", flush=True)
    clima = leer_csv(PROCESSED / "INIA/INIA.csv")
    assert set(clima.EstacionAgr) <= estaciones.keys()
    agregar_perfil(clima, "INIA", "INIA", detalle_acumulado, estaciones)
    fechas = pd.to_datetime(clima[["Anio", "Mes", "Dia"]].rename(columns={"Anio": "year", "Mes": "month", "Dia": "day"}), errors="raise")
    aux = clima.assign(fecha=fechas)
    resumen["inia_estaciones"] = [dict(estacion=e, departamento=nombres_dept[estaciones[e]],
        registros=len(g), fecha_inicial=g.fecha.min().strftime("%Y-%m-%d"), fecha_final=g.fecha.max().strftime("%Y-%m-%d"))
        for e, g in aux.groupby("EstacionAgr", sort=True)]
    resumen["inia_anios"] = [dict(Anio=a, n_registros=len(g)) for a, g in clima.groupby("Anio", sort=True)]

    detalle = pd.DataFrame([{**dict(zip(CLAVE_DETALLE, k)), **v} for k, v in detalle_acumulado.items()])
    detalle = completar_perfil(detalle).sort_values(list(CLAVE_DETALLE), kind="stable")
    totales = detalle.groupby(["fuente", "tabla", "columna"], as_index=False)[list(METRICAS)].sum()
    totales = completar_perfil(totales)
    # Restaurar orden de columnas de los CSV originales, no orden alfabético.
    orden = diccionario[["fuente", "tabla", "columna", "orden_columna"]]
    totales = totales.merge(orden, on=["fuente", "tabla", "columna"], validate="one_to_one")
    totales = totales.sort_values(["fuente", "tabla", "orden_columna"], kind="stable").drop(columns="orden_columna")
    totales["interpretacion"] = [interpretacion(f, c) for f, c in zip(totales.fuente, totales.columna)]
    snig = totales[totales.fuente.eq("SNIG")].copy()
    inia = totales[totales.fuente.eq("INIA")].copy()
    # Anio está en todas las tablas: su denominador cuenta cada fila una vez.
    cobertura = detalle[detalle.fuente.eq("SNIG") & detalle.columna.eq("Anio")].groupby(
        ["tabla", "DepartamentoCodigo"], as_index=False)["n_registros"].sum()
    cobertura["departamento"] = cobertura.DepartamentoCodigo.map(nombres_dept)
    area = detalle[detalle.fuente.eq("SNIG") & detalle.columna.eq("AreaEnumeracion")].groupby(
        ["tabla", "DepartamentoCodigo"], as_index=False)["n_ceros"].sum().rename(columns={"n_ceros": "area_0_conservada"})
    cobertura = cobertura.merge(area, validate="one_to_one", on=["tabla", "DepartamentoCodigo"])
    cobertura["porcentaje_registros_tabla"] = (100 * cobertura.n_registros / cobertura.groupby("tabla").n_registros.transform("sum")).round(4)
    cobertura = cobertura[["tabla", "DepartamentoCodigo", "departamento", "n_registros", "porcentaje_registros_tabla", "area_0_conservada"]]

    # Verificaciones de reconciliación: mismos denominadores en toda columna,
    # todos los departamentos y todas las columnas originales representadas.
    for tabla in CENTRALES:
        r = resumen["snig"][tabla]
        c = cobertura[cobertura.tabla.eq(tabla)]
        assert set(c.DepartamentoCodigo) == departamentos
        assert int(c.n_registros.sum()) == r["retenidos"]
        assert int(c.area_0_conservada.sum()) == r["area_0_conservada"]
        assert snig.loc[snig.tabla.eq(tabla), "n_registros"].eq(r["retenidos"]).all()
        assert len(snig[snig.tabla.eq(tabla)]) == len(leer_csv(PROCESSED / "SNIG" / f"{tabla}.csv", nrows=0).columns)
    assert len(inia) == len(clima.columns) and inia.n_registros.eq(len(clima)).all()
    assert len(pies) == 53 and len(mapa_nombres_inia()) == 47
    huellas_despues = {relativo(p): huella(p) for p in sorted(entradas)}
    assert huellas_antes == huellas_despues, "Una entrada cambió durante el diagnóstico"

    salida.mkdir(parents=True, exist_ok=False)
    guardar_csv(snig, salida / "faltantes_snig.csv")
    guardar_csv(cobertura, salida / "cobertura_snig_departamento.csv")
    guardar_csv(inia, salida / "faltantes_inia.csv")
    guardar_csv(detalle, salida / "detalle_faltantes.csv")
    guardar_csv(diccionario, salida / "diccionario_columnas.csv")
    with (salida / "resumen_diagnostico.md").open("x", encoding="utf-8") as stream:
        stream.write(informe(snig, inia, detalle, cobertura, resumen, diccionario))
    verificacion = dict(resultado="OK", departamentos_incluidos=sorted(departamentos, key=int),
        marcadores_nulos=sorted(MARCADORES_NULOS), filtro="Solo DepartamentoCodigo en SNIG",
        sin_imputacion=True, sin_agregacion_a_unidad_estudio=True, area_cero_conservada=True,
        entradas_sin_cambios=True, sha256_entradas=huellas_antes, resumen_conteos=resumen,
        columnas_diagnostico_snig=len(snig), columnas_diagnostico_inia=len(inia),
        filas_diccionario=len(diccionario), tablas_diccionario=len(tablas),
        definiciones_por_pie_inia=len(pies), pies_inia_concordantes=len(raw_inia),
        version_pandas=pd.__version__, python=sys.version,
        csv_reabiertos_y_verificados=True)
    with (salida / "verificacion.json").open("x", encoding="utf-8") as stream:
        json.dump(verificacion, stream, ensure_ascii=False, indent=2)
    print(f"Listo: {len(snig)} columnas SNIG, {len(inia)} columnas INIA, {len(diccionario)} entradas del diccionario.", flush=True)
    print(f"Resultados: {salida}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
