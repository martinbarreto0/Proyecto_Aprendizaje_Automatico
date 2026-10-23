"""Cuenta coincidencias de dimensiones al quitar los códigos MGAP de la clave.

Estas coincidencias no demuestran que una fila de otro año sea una agregación.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
SNIG = BASE.parents[3] / 'data/processed/SNIG'
SALIDAS = BASE.parent / 'resultados_mgap_snig'
comunes = ['Anio', 'DepartamentoCodigo', 'SeccionalPolicialCodigo', 'AreaSupervision',
           'AreaEnumeracion', 'ActividadCodigo', 'GiroCodigo', 'NaturalezaJuridicaCodigo', 'EstratoCodigo']
especificas = ['EspecieCodigo', 'CategoriaCodigo', 'TipoProduccionCodigo',
               'ProductoCodigo', 'TenenciaCodigo', 'UsoCodigo']
objetivos = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']
resultados = []

for ruta in sorted(SNIG.glob('*.csv')):
    encabezado = pd.read_csv(ruta, sep=';', nrows=0).columns
    clave = comunes + [c for c in especificas if c in encabezado]
    bloques = []
    for bloque in pd.read_csv(ruta, sep=';', usecols=clave + objetivos,
            dtype={c: str for c in clave}, na_values=['-', ' ', 'NULL', 'null'], chunksize=200_000):
        parte = bloque.loc[bloque.Anio.isin(['2021', '2022'])].copy()
        if len(parte):
            parte[clave] = parte[clave].fillna('SIN_DATO')
            bloques.append(parte)
    df = pd.concat(bloques, ignore_index=True)
    anterior = df.loc[df.Anio.eq('2021')]
    conocido = df.loc[df.Anio.eq('2022')].dropna(subset=objetivos)
    variedad = conocido.groupby(clave)[objetivos].nunique()
    ambiguos = variedad.gt(1).any(axis=1)
    repetidas = conocido.duplicated(clave, keep=False)
    fila = dict(tabla=ruta.stem, filas_2021=len(anterior), filas_2022=len(conocido),
        claves_2021=anterior[clave].drop_duplicates().shape[0],
        claves_2022=len(variedad), claves_varias_clasificaciones=int(ambiguos.sum()),
        filas_clave_repetida_2022=int(repetidas.sum()),
        proporcion_filas_clave_repetida_2022=float(repetidas.mean()),
        duplicados_con_mgap_2022=int(conocido.duplicated(clave + objetivos).sum()))
    resultados.append(fila)
    print(fila, flush=True)

SALIDAS.mkdir(exist_ok=True)
pd.DataFrame(resultados).to_csv(SALIDAS / 'diagnostico_grano.csv', index=False)
