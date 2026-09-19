"""Entrena los mejores modelos MGAP y completa copias de las tablas SNIG."""
from pathlib import Path
import argparse
import gc
import json
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tabla', help='Procesar una sola tabla; por defecto procesa las siete.')
parser.add_argument('--hilos', type=int, default=4)
parser.add_argument('--tamano-bloque', type=int, default=100_000)
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

import numpy as np
import pandas as pd

from configuracion_MGAP_snig import TABLAS, OBJETIVOS
from funciones_MGAP_snig import (SNIG, SALIDAS, cargar_tabla, calcular_pesos,
                                 crear_modelo)

BASE = Path(__file__).resolve().parent
DESTINO = BASE.parents[1] / 'data'
METADATOS = DESTINO / 'metadatos'
CONFIANZAS = {
    'EspecializacionMGAPCodigo': 'EspecializacionMGAPConfianza',
    'TipoProduccionMGAPCodigo': 'TipoProduccionMGAPConfianza',
}
NULOS_TEXTO = {'', '-', 'NULL', 'null', 'nan', 'NaN', '********'}


def es_faltante(serie):
    """Reconoce los marcadores vacíos sin modificar el texto de otras columnas."""
    return serie.fillna('').astype(str).str.strip().isin(NULOS_TEXTO)


def normalizar_categoria(serie, categorias_entrenamiento):
    """Hace coincidir códigos como 01, 1 y 1.0 con la forma usada al entrenar."""
    resultado = serie.fillna('SIN_DATO').astype(str).str.strip()
    resultado = resultado.mask(resultado.isin(NULOS_TEXTO), 'SIN_DATO')
    conocidas = set(categorias_entrenamiento)
    pendientes = ~resultado.isin(conocidas)
    if pendientes.any():
        numero = pd.to_numeric(resultado.where(pendientes), errors='coerce')
        entero = numero.astype('Int64').astype(str)
        decimal = numero.astype(float).astype(str)
        usar_entero = pendientes & entero.isin(conocidas)
        resultado = resultado.mask(usar_entero, entero)
        usar_decimal = ~resultado.isin(conocidas) & decimal.isin(conocidas)
        resultado = resultado.mask(usar_decimal, decimal)
    return resultado


def preparar_bloque(tabla, bloque, categorias_entrenamiento):
    """Aplica al bloque la misma limpieza y las mismas proporciones del entrenamiento."""
    config = TABLAS[tabla]
    cat, num = config['categoricas'], config['numericas']
    X = bloque[cat + num].copy()
    for col in cat:
        X[col] = normalizar_categoria(X[col], categorias_entrenamiento[col])
    for col in num:
        original = X[col].fillna('').astype(str).str.strip()
        limpio = original.mask(original.isin(NULOS_TEXTO))
        convertido = pd.to_numeric(limpio.str.replace(',', '.', regex=False), errors='coerce')
        if (limpio.notna() & convertido.isna()).any():
            ejemplos = original.loc[limpio.notna() & convertido.isna()].drop_duplicates().head(5).tolist()
            raise ValueError(f'{tabla}: valores no numéricos en {col}: {ejemplos}')
        X[col] = convertido.fillna(0)
    for nueva, (numerador, denominador) in config['proporciones'].items():
        X[nueva] = X[numerador] / X[denominador].clip(lower=1)
    return X


def procesar_tabla(tabla):
    print(f'\n=== {tabla} ===', flush=True)
    config = TABLAS[tabla]
    ruta = SNIG / f'{tabla}.csv'
    salida = DESTINO / f'{tabla}.csv'
    temporal = salida.with_suffix('.csv.tmp')
    elegidos = json.loads((SALIDAS / tabla / 'modelos_elegidos.json').read_text(encoding='utf-8'))

    # Entrenar con todas las filas que ya tienen los dos códigos MGAP.
    df_train, X_train, y_train = cargar_tabla(tabla)
    categorias_entrenamiento = {
        col: set(X_train[col].astype(str).unique()) for col in config['categoricas']
    }
    X_train = pd.get_dummies(X_train, columns=config['categoricas'])
    columnas_modelo = X_train.columns
    modelos = {}
    for objetivo in OBJETIVOS:
        info = elegidos[objetivo]
        iteraciones = info['max_iter'] or 250
        modelo = crear_modelo(info['modelo'], y_train[objetivo], args.hilos, iteraciones)
        print(f'Entrenando {objetivo}: {info["modelo"]} con {len(y_train)} filas', flush=True)
        modelo.fit(X_train, y_train[objetivo],
                   sample_weight=calcular_pesos(y_train[objetivo], info['exponente_pesos']))
        modelos[objetivo] = modelo
    del X_train, df_train
    gc.collect()

    filas = 0
    imputadas = 0
    suma_confianza = {o: 0.0 for o in OBJETIVOS}
    minima_confianza = {o: 1.0 for o in OBJETIVOS}
    maxima_confianza = {o: 0.0 for o in OBJETIVOS}
    requeridas = config['categoricas'] + config['numericas'] + OBJETIVOS

    DESTINO.mkdir(parents=True, exist_ok=True)
    METADATOS.mkdir(parents=True, exist_ok=True)
    with temporal.open('w', encoding='utf-8-sig', newline='') as archivo_salida:
        for numero_bloque, bloque in enumerate(pd.read_csv(
                ruta, sep=';', dtype=str, keep_default_na=False,
                chunksize=args.tamano_bloque), start=1):
            faltantes = [c for c in requeridas if c not in bloque.columns]
            if faltantes:
                raise ValueError(f'{tabla}: faltan columnas requeridas: {faltantes}')
            mascara_esp = es_faltante(bloque[OBJETIVOS[0]])
            mascara_tipo = es_faltante(bloque[OBJETIVOS[1]])
            if not mascara_esp.equals(mascara_tipo):
                raise ValueError(f'{tabla}: se encontraron objetivos MGAP parcialmente faltantes')
            mascara = mascara_esp
            for confianza in CONFIANZAS.values():
                bloque[confianza] = ''

            if mascara.any():
                X_pred = preparar_bloque(tabla, bloque.loc[mascara], categorias_entrenamiento)
                X_pred = pd.get_dummies(X_pred, columns=config['categoricas'])
                X_pred = X_pred.reindex(columns=columnas_modelo, fill_value=0)
                for objetivo in OBJETIVOS:
                    modelo = modelos[objetivo]
                    prediccion = modelo.predict(X_pred).astype(int)
                    confianza = modelo.predict_proba(X_pred).max(axis=1)
                    bloque.loc[mascara, objetivo] = prediccion.astype(str)
                    bloque.loc[mascara, CONFIANZAS[objetivo]] = np.char.mod('%.6f', confianza)
                    suma_confianza[objetivo] += float(confianza.sum())
                    minima_confianza[objetivo] = min(minima_confianza[objetivo], float(confianza.min()))
                    maxima_confianza[objetivo] = max(maxima_confianza[objetivo], float(confianza.max()))
                imputadas += int(mascara.sum())
                del X_pred

            bloque.to_csv(archivo_salida, sep=';', index=False, header=numero_bloque == 1,
                          lineterminator='\n')
            filas += len(bloque)
            print(f'Bloque {numero_bloque}: {filas} filas escritas, {imputadas} imputadas', flush=True)
            gc.collect()

    temporal.replace(salida)
    resumen = {
        'tabla': tabla,
        'origen': str(ruta),
        'salida': str(salida),
        'filas': filas,
        'filas_entrenamiento': len(y_train),
        'filas_imputadas': imputadas,
        'columnas_confianza': CONFIANZAS,
        'modelos': elegidos,
        'confianza': {
            objetivo: {
                'media': suma_confianza[objetivo] / imputadas if imputadas else None,
                'minima': minima_confianza[objetivo] if imputadas else None,
                'maxima': maxima_confianza[objetivo] if imputadas else None,
            }
            for objetivo in OBJETIVOS
        },
        'nota_confianza': 'Máxima probabilidad estimada por el modelo; no es una probabilidad calibrada.',
        'marcadores_numericos_sin_dato': sorted(NULOS_TEXTO),
    }
    (METADATOS / f'{tabla}.json').write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Guardado: {salida} ({imputadas} filas imputadas)', flush=True)
    del y_train, modelos
    gc.collect()


tablas = [args.tabla] if args.tabla else list(TABLAS)
for nombre_tabla in tablas:
    if nombre_tabla not in TABLAS:
        raise ValueError(f'Tabla desconocida: {nombre_tabla}. Opciones: {list(TABLAS)}')
    procesar_tabla(nombre_tabla)
