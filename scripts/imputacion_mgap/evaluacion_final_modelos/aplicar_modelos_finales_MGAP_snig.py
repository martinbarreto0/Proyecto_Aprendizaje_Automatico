"""Completa las columnas MGAP con el modelo ganador de la comparación final."""
from pathlib import Path
import argparse
import gc
import json
import os
import sys


BASE = Path(__file__).resolve().parent
MGAP = BASE.parent
MODELOS_BASE = MGAP / 'modelos_base'
sys.path.insert(0, str(MODELOS_BASE))
sys.path.insert(0, str(MGAP / 'ensamble'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tabla', help='Completa una sola tabla; por defecto completa las siete.')
parser.add_argument('--hilos', type=int, default=4)
parser.add_argument('--tamano-bloque', type=int, default=100_000)
parser.add_argument('--max-knn', type=int, default=200_000,
                    help='Máximo de filas de entrenamiento para KNN en ensambles.')
args = parser.parse_args()
os.environ['LOKY_MAX_CPU_COUNT'] = str(args.hilos)
os.environ['OMP_NUM_THREADS'] = str(args.hilos)

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from configuracion_MGAP_snig import OBJETIVOS, TABLAS
from funciones_MGAP_snig import SNIG, calcular_pesos, cargar_tabla
from funciones_ensamble_MGAP_snig import (BASES, entrenar_base, entrenar_meta,
                                          probabilidades_oof, puntajes_ensamble)


MANIFIESTO = BASE / 'modelos_congelados.json'
RESUMEN = BASE / 'resultados_finales' / 'resumen_comparacion_final.csv'
DESTINO = MGAP.parents[1] / 'data'
METADATOS = DESTINO / 'metadatos_modelos_finales'
CONFIANZAS = {
    'EspecializacionMGAPCodigo': 'EspecializacionMGAPConfianza',
    'TipoProduccionMGAPCodigo': 'TipoProduccionMGAPConfianza',
}
NULOS_TEXTO = {'', '-', 'NULL', 'null', 'nan', 'NaN', '********'}


def es_faltante(serie):
    """Reconoce los marcadores de ausencia sin alterar otras columnas."""
    return serie.fillna('').astype(str).str.strip().isin(NULOS_TEXTO)


def normalizar_categoria(serie, categorias_entrenamiento):
    """Hace coincidir códigos como 01, 1 y 1.0 con el entrenamiento."""
    resultado = serie.fillna('SIN_DATO').astype(str).str.strip()
    resultado = resultado.mask(resultado.isin(NULOS_TEXTO), 'SIN_DATO')
    conocidas = set(categorias_entrenamiento)
    pendientes = ~resultado.isin(conocidas)
    if pendientes.any():
        numero = pd.to_numeric(resultado.where(pendientes), errors='coerce')
        entero = numero.astype('Int64').astype(str)
        decimal = numero.astype(float).astype(str)
        resultado = resultado.mask(pendientes & entero.isin(conocidas), entero)
        resultado = resultado.mask(~resultado.isin(conocidas) & decimal.isin(conocidas), decimal)
    return resultado


def preparar_bloque(tabla, bloque, categorias_entrenamiento):
    """Aplica al bloque la misma limpieza y proporciones del entrenamiento."""
    config = TABLAS[tabla]
    cat, num = config['categoricas'], config['numericas']
    X = bloque[cat + num].copy()
    for columna in cat:
        X[columna] = normalizar_categoria(X[columna], categorias_entrenamiento[columna])
    for columna in num:
        original = X[columna].fillna('').astype(str).str.strip()
        limpio = original.mask(original.isin(NULOS_TEXTO))
        convertido = pd.to_numeric(limpio.str.replace(',', '.', regex=False), errors='coerce')
        if (limpio.notna() & convertido.isna()).any():
            ejemplos = original.loc[limpio.notna() & convertido.isna()].drop_duplicates().head(5).tolist()
            raise ValueError(f'{tabla}: valores no numéricos en {columna}: {ejemplos}')
        X[columna] = convertido.fillna(0)
    for nueva, (numerador, denominador) in config['proporciones'].items():
        X[nueva] = X[numerador] / X[denominador].clip(lower=1)
    return X


def crear_modelo(familia, parametros, y_train):
    """Crea el clasificador individual usando los parámetros ganadores congelados."""
    if familia == 'RF':
        return RandomForestClassifier(random_state=42, n_jobs=args.hilos,
            n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'],
            min_samples_split=parametros['min_samples_split'],
            min_samples_leaf=parametros['min_samples_leaf'], max_features=parametros['max_features'])
    if familia == 'ExtraTrees':
        return ExtraTreesClassifier(random_state=42, n_jobs=args.hilos,
            n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'],
            min_samples_split=parametros['min_samples_split'],
            min_samples_leaf=parametros['min_samples_leaf'], max_features=parametros['max_features'])
    if familia == 'HistBoost':
        detener = len(y_train) > 10_000 and y_train.value_counts().min() >= 2
        return HistGradientBoostingClassifier(random_state=42,
            max_iter=parametros['max_iter'], learning_rate=parametros['learning_rate'],
            max_leaf_nodes=parametros['max_leaf_nodes'], min_samples_leaf=parametros['min_samples_leaf'],
            l2_regularization=parametros['l2_regularization'], early_stopping=detener)
    raise ValueError(f'Familia desconocida: {familia}')


def entrenar_individual(info, X_train, y_train):
    """Ajusta un modelo individual y conserva sus clases para predecir por bloques."""
    modelo = crear_modelo(info['familia'], info['parametros'], y_train)
    modelo.fit(X_train, y_train, sample_weight=calcular_pesos(y_train, info['parametros']['exponente_pesos']))
    return {'tipo': 'individual', 'modelo': modelo, 'descripcion': info}


def entrenar_ensamble(info, X_train, y_train):
    """Ajusta las bases y, para E4, la meta de stacking sin usar filas faltantes."""
    elegido = info['ensamble']
    hb = tuple(elegido['histboost'])
    bases = {nombre: entrenar_base(nombre, X_train, y_train, hb, args.hilos, args.max_knn)
             for nombre in BASES}
    meta = None
    if elegido['ensamble'] == 'E4_stacking':
        oof, _, _ = probabilidades_oof(X_train, y_train, hb, args.hilos, args.max_knn, pliegues=3)
        meta = entrenar_meta(oof, y_train)
    clases = bases[BASES[0]].classes_
    if not all(np.array_equal(clases, bases[nombre].classes_) for nombre in BASES[1:]):
        raise ValueError('Las bases del ensamble no tienen las mismas clases.')
    return {'tipo': 'ensamble', 'bases': bases, 'meta': meta, 'clases': clases, 'descripcion': elegido}


def predecir(modelo, X):
    """Devuelve predicción y confianza comparable entre modelos individuales y ensambles."""
    if modelo['tipo'] == 'individual':
        probabilidades = modelo['modelo'].predict_proba(X)
        return modelo['modelo'].classes_[probabilidades.argmax(axis=1)], probabilidades.max(axis=1)

    probs = {nombre: base.predict_proba(X) for nombre, base in modelo['bases'].items()}
    elegido = modelo['descripcion']
    puntajes = puntajes_ensamble(elegido['ensamble'], probs, modelo['meta'])
    prediccion = modelo['clases'][puntajes.argmax(axis=1)]
    if elegido['ensamble'] == 'E1_voto_duro':
        # Voto duro devuelve cantidad de votos más un desempate mínimo por probabilidad.
        confianza = np.floor(puntajes.max(axis=1)) / len(elegido['miembros'])
    else:
        confianza = puntajes.max(axis=1)
    return prediccion, confianza


def modelo_ganador(tabla, objetivo, manifest, resumen):
    """Construye la especificación del método ganador sin volver a seleccionar nada."""
    info = manifest['tablas'][tabla]['objetivos'][objetivo]
    ganador = resumen.loc[(tabla, objetivo), 'metodo_ganador']
    if ganador == 'Original':
        return {'familia': info['familia_original'], 'parametros': info['parametros_originales']}
    if ganador == 'Hiperparametros':
        return {'familia': info['familia_hiperparametros'], 'parametros': info['parametros_hiperparametros']}
    if ganador == 'Ensamble':
        return {'ensamble': info['ensamble']}
    raise ValueError(f'{tabla}/{objetivo}: ganador desconocido {ganador}')


def procesar_tabla(tabla, manifest, resumen):
    """Entrena los ganadores y completa las filas faltantes de una tabla original."""
    print(f'\n=== {tabla} ===', flush=True)
    config = TABLAS[tabla]
    ruta = SNIG / f'{tabla}.csv'
    salida = DESTINO / f'{tabla}.csv'
    temporal = salida.with_suffix('.csv.tmp')

    _, X_train, y_train = cargar_tabla(tabla)
    categorias = {col: set(X_train[col].astype(str).unique()) for col in config['categoricas']}
    X_train = pd.get_dummies(X_train, columns=config['categoricas'])
    columnas_modelo = X_train.columns
    modelos = {}
    especificaciones = {}
    for objetivo in OBJETIVOS:
        info = modelo_ganador(tabla, objetivo, manifest, resumen)
        especificaciones[objetivo] = info
        print(f'Entrenando {objetivo}: {"ensamble" if "ensamble" in info else info["familia"]}', flush=True)
        if 'ensamble' in info:
            modelos[objetivo] = entrenar_ensamble(info, X_train, y_train[objetivo])
        else:
            modelos[objetivo] = entrenar_individual(info, X_train, y_train[objetivo])
    del X_train
    gc.collect()

    DESTINO.mkdir(parents=True, exist_ok=True)
    METADATOS.mkdir(parents=True, exist_ok=True)
    filas = 0
    imputadas = {objetivo: 0 for objetivo in OBJETIVOS}
    confianza = {objetivo: [] for objetivo in OBJETIVOS}
    requeridas = config['categoricas'] + config['numericas'] + OBJETIVOS

    with temporal.open('w', encoding='utf-8-sig', newline='') as archivo:
        for numero_bloque, bloque in enumerate(pd.read_csv(
                ruta, sep=';', dtype=str, keep_default_na=False, chunksize=args.tamano_bloque), start=1):
            faltantes = [columna for columna in requeridas if columna not in bloque.columns]
            if faltantes:
                raise ValueError(f'{tabla}: faltan columnas requeridas: {faltantes}')
            mascaras = {objetivo: es_faltante(bloque[objetivo]) for objetivo in OBJETIVOS}
            for columna_confianza in CONFIANZAS.values():
                bloque[columna_confianza] = ''
            union = mascaras[OBJETIVOS[0]] | mascaras[OBJETIVOS[1]]
            if union.any():
                X_pred = preparar_bloque(tabla, bloque.loc[union], categorias)
                X_pred = pd.get_dummies(X_pred, columns=config['categoricas']).reindex(
                    columns=columnas_modelo, fill_value=0)
                for objetivo in OBJETIVOS:
                    mascara = mascaras[objetivo]
                    if not mascara.any():
                        continue
                    indices_objetivo = mascara[mascara].index
                    prediccion, conf = predecir(modelos[objetivo], X_pred.loc[indices_objetivo])
                    bloque.loc[mascara, objetivo] = prediccion.astype(str)
                    bloque.loc[mascara, CONFIANZAS[objetivo]] = np.char.mod('%.6f', conf)
                    imputadas[objetivo] += int(mascara.sum())
                    confianza[objetivo].extend(conf.tolist())
                del X_pred
            bloque.to_csv(archivo, sep=';', index=False, header=numero_bloque == 1, lineterminator='\n')
            filas += len(bloque)
            print(f'Bloque {numero_bloque}: {filas} filas escritas', flush=True)
            gc.collect()

    temporal.replace(salida)
    resumen_tabla = {
        'tabla': tabla, 'origen': str(ruta), 'salida': str(salida), 'filas': filas,
        'filas_entrenamiento': len(y_train), 'filas_imputadas': imputadas,
        'columnas_confianza': CONFIANZAS, 'modelos': especificaciones,
        'confianza': {
            objetivo: {'media': float(np.mean(valores)) if valores else None,
                       'minima': float(np.min(valores)) if valores else None,
                       'maxima': float(np.max(valores)) if valores else None}
            for objetivo, valores in confianza.items()
        },
        'nota_confianza': ('Máxima probabilidad del método; en E1 equivale a la fracción de votos '
                            'de la clase elegida. No son probabilidades calibradas.'),
    }
    (METADATOS / f'{tabla}.json').write_text(json.dumps(resumen_tabla, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Guardado: {salida}', flush=True)
    del y_train, modelos
    gc.collect()


if not MANIFIESTO.exists() or not RESUMEN.exists():
    raise FileNotFoundError('Faltan modelos_congelados.json o resumen_comparacion_final.csv.')
manifest = json.loads(MANIFIESTO.read_text(encoding='utf-8'))
resumen = pd.read_csv(RESUMEN).set_index(['tabla', 'objetivo'])
tablas = [args.tabla] if args.tabla else list(TABLAS)
for tabla in tablas:
    if tabla not in TABLAS or tabla not in manifest['tablas']:
        raise ValueError(f'Tabla desconocida o sin resultado final: {tabla}')
    procesar_tabla(tabla, manifest, resumen)
