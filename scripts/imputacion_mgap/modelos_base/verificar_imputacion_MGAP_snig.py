"""Verifica que las copias imputadas conserven las tablas SNIG originales."""
from itertools import zip_longest
from pathlib import Path
import json

import pandas as pd

from configuracion_MGAP_snig import TABLAS, OBJETIVOS
from funciones_MGAP_snig import SNIG

BASE = Path(__file__).resolve().parent
DESTINO = BASE.parents[2] / 'data'
CONFIANZAS = {
    'EspecializacionMGAPCodigo': 'EspecializacionMGAPConfianza',
    'TipoProduccionMGAPCodigo': 'TipoProduccionMGAPConfianza',
}
NULOS_TEXTO = {'', '-', 'NULL', 'null', 'nan', 'NaN'}


def es_faltante(serie):
    return serie.fillna('').astype(str).str.strip().isin(NULOS_TEXTO)


def verificar_tabla(tabla):
    origen = SNIG / f'{tabla}.csv'
    salida = DESTINO / f'{tabla}.csv'
    metadata = json.loads(
        (DESTINO / 'metadatos' / f'{tabla}.json').read_text(encoding='utf-8')
    )

    columnas_origen = pd.read_csv(origen, sep=';', nrows=0).columns.tolist()
    columnas_salida = pd.read_csv(salida, sep=';', nrows=0).columns.tolist()
    esperadas = columnas_origen + list(CONFIANZAS.values())
    if columnas_salida != esperadas:
        raise AssertionError(f'{tabla}: las columnas o su orden no coinciden')

    total = 0
    imputadas = 0
    otras = [col for col in columnas_origen if col not in OBJETIVOS]
    bloques_origen = pd.read_csv(
        origen, sep=';', dtype=str, keep_default_na=False, chunksize=100_000
    )
    bloques_salida = pd.read_csv(
        salida, sep=';', dtype=str, keep_default_na=False, chunksize=100_000
    )

    for numero, (antes, despues) in enumerate(
            zip_longest(bloques_origen, bloques_salida), start=1):
        if antes is None or despues is None or len(antes) != len(despues):
            raise AssertionError(f'{tabla}: distinta cantidad de filas en el bloque {numero}')
        if not antes[otras].equals(despues[otras]):
            raise AssertionError(f'{tabla}: cambiaron datos originales en el bloque {numero}')

        falta_esp = es_faltante(antes[OBJETIVOS[0]])
        falta_tipo = es_faltante(antes[OBJETIVOS[1]])
        if not falta_esp.equals(falta_tipo):
            raise AssertionError(f'{tabla}: el origen tiene objetivos parcialmente faltantes')
        mascara = falta_esp
        conocidas = ~mascara

        for objetivo in OBJETIVOS:
            confianza = CONFIANZAS[objetivo]
            if not antes.loc[conocidas, objetivo].equals(
                    despues.loc[conocidas, objetivo]):
                raise AssertionError(f'{tabla}: se modificaron valores conocidos de {objetivo}')
            if es_faltante(despues.loc[mascara, objetivo]).any():
                raise AssertionError(f'{tabla}: quedaron predicciones vacías en {objetivo}')
            if despues.loc[conocidas, confianza].ne('').any():
                raise AssertionError(f'{tabla}: hay confianza en filas que no fueron predichas')
            valores = pd.to_numeric(despues.loc[mascara, confianza], errors='coerce')
            if valores.isna().any() or not valores.between(0, 1).all():
                raise AssertionError(f'{tabla}: confianza inválida en {confianza}')

        total += len(antes)
        imputadas += int(mascara.sum())

    if total != metadata['filas'] or imputadas != metadata['filas_imputadas']:
        raise AssertionError(f'{tabla}: los conteos no coinciden con los metadatos')
    print(f'{tabla}: OK - {total} filas, {imputadas} imputadas')


for nombre_tabla in TABLAS:
    verificar_tabla(nombre_tabla)

print('Verificación completa: los siete datasets son válidos.')
