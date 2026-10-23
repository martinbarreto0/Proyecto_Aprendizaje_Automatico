"""Revisa columnas, faltantes y clases de todas las tablas SNIG, por bloques."""
from pathlib import Path
import hashlib
import json
import pandas as pd

BASE = Path(__file__).resolve().parent
SNIG = BASE.parents[3] / 'data/processed/SNIG'
SALIDAS = BASE.parent / 'resultados_mgap_snig'
SALIDAS.mkdir(exist_ok=True)
OBJETIVOS = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']
perfiles = {}
columnas = []

for ruta in sorted(SNIG.glob('*.csv')):
    n = 0
    conocidos = 0
    parciales = 0
    faltantes = None
    clases = {col: {} for col in OBJETIVOS}
    por_anio = []
    ejemplos = {}
    for bloque in pd.read_csv(ruta, sep=';', na_values=['-', ' ', 'NULL', 'null'],
                              chunksize=200_000, low_memory=False):
        n += len(bloque)
        ambos = bloque[OBJETIVOS].notna().all(axis=1)
        conocidos += int(ambos.sum())
        parciales += int(bloque[OBJETIVOS].isna().sum(axis=1).eq(1).sum())
        faltantes = bloque.isna().sum() if faltantes is None else faltantes + bloque.isna().sum()
        for col in bloque.columns:
            if col not in ejemplos:
                valores = bloque[col].dropna()
                if len(valores):
                    ejemplos[col] = str(valores.iloc[0])
        for col in OBJETIVOS:
            for valor, cantidad in bloque.loc[ambos, col].value_counts().items():
                clave = str(int(valor))
                clases[col][clave] = clases[col].get(clave, 0) + int(cantidad)
        resumen = bloque.assign(AmbosConocidos=ambos.astype(int))
        por_anio.append(resumen.groupby('Anio').agg(
            filas=('ID', 'size'), conocidos=('AmbosConocidos', 'sum')))
    anios = pd.concat(por_anio).groupby(level=0).sum()
    perfil = dict(filas=n, ambos_conocidos=conocidos, faltantes_parciales=parciales,
                  columnas=list(faltantes.index), faltantes=faltantes.astype(int).to_dict(),
                  clases=clases, por_anio=anios.to_dict('index'),
                  sha256=hashlib.sha256(ruta.read_bytes()).hexdigest())
    perfiles[ruta.stem] = perfil
    for col in faltantes.index:
        columnas.append(dict(tabla=ruta.stem, columna=col, faltantes=int(faltantes[col]),
                             ejemplo=ejemplos.get(col, '')))
    print(ruta.stem, json.dumps(perfil, ensure_ascii=False), flush=True)

(SALIDAS / 'perfil_tablas.json').write_text(json.dumps(perfiles, indent=2, ensure_ascii=False), encoding='utf-8')
pd.DataFrame(columnas).to_csv(SALIDAS / 'columnas_por_tabla.csv', index=False)
