"""Reúne las evaluaciones completas y verifica que los CSV SNIG sigan intactos."""
import ast
import hashlib
import json
from pathlib import Path
import pandas as pd
from configuracion_MGAP_snig import TABLAS, OBJETIVOS

BASE = Path(__file__).resolve().parent
SALIDAS = BASE.parent / 'resultados_mgap_snig'
SNIG = BASE.parents[3] / 'data/processed/SNIG'
perfil = json.loads((SALIDAS / 'perfil_tablas.json').read_text(encoding='utf-8'))
comparaciones = []
evaluaciones = []

for tabla, config in TABLAS.items():
    carpeta = SALIDAS / tabla
    elegidos = json.loads((carpeta / 'modelos_elegidos.json').read_text(encoding='utf-8'))
    # Verificar las columnas de los scripts independientes contra su configuración.
    nombre_script = 'imputar_MGAPv7.py' if tabla == 'DatosAnimales' else f'imputar_MGAP_{tabla}.py'
    fuente = ast.parse((BASE / nombre_script).read_text(encoding='utf-8'))
    constantes = {n.targets[0].id: ast.literal_eval(n.value) for n in fuente.body
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in ['cat', 'num', 'proporciones', 'nombres', 'exponentes', 'max_iteraciones']}
    assert constantes['cat'] == config['categoricas'], tabla
    assert constantes['num'] == config['numericas'], tabla
    assert constantes['proporciones'] == {k: list(v) for k, v in config['proporciones'].items()}, tabla
    assert not set(OBJETIVOS + ['ID', 'Anio']) & set(constantes['cat'] + constantes['num']), tabla
    assert set(constantes['cat'] + constantes['num']).issubset(perfil[tabla]['columnas']), tabla
    assert hashlib.sha256((SNIG / f'{tabla}.csv').read_bytes()).hexdigest() == perfil[tabla]['sha256'], tabla
    assert constantes['max_iteraciones'] == {o: info['max_iter'] for o, info in elegidos.items()}, tabla

    seleccion = pd.read_csv(carpeta / 'comparacion.csv', keep_default_na=False)
    esperados = 12 if tabla == 'DatosAnimalesDetallados' else 14
    assert len(seleccion) == esperados and not seleccion.duplicated(['objetivo', 'modelo']).any(), tabla
    for objetivo in OBJETIVOS:
        opciones = seleccion.loc[seleccion.objetivo.eq(objetivo) & seleccion.modelo.ne('Mayoritaria')]
        mejor = opciones.sort_values(['f1_macro', 'accuracy'], ascending=False).iloc[0]
        assert mejor.modelo == elegidos[objetivo]['modelo'] == constantes['nombres'][objetivo], tabla
        assert mejor.exponente_pesos == constantes['exponentes'][objetivo], tabla
    comparaciones.append(seleccion)

    referencias = pd.read_csv(carpeta / 'evaluacion_referencias.csv', keep_default_na=False)
    assert len(referencias) == 4, tabla
    evaluaciones.append(referencias)
    prueba = pd.read_csv(carpeta / 'evaluacion_elegidos_aleatorio.csv', keep_default_na=False)
    assert len(prueba) == 2 and set(prueba.objetivo) == set(OBJETIVOS), tabla
    assert set(prueba.etapa) == {'prueba_aleatoria'}, tabla
    evaluaciones.append(prueba)

comparacion = pd.concat(comparaciones, ignore_index=True)
evaluacion = pd.concat(evaluaciones, ignore_index=True)
assert len(evaluacion) == 6 * len(TABLAS)
assert not evaluacion.duplicated(['tabla', 'etapa', 'objetivo', 'rol', 'modelo']).any()
assert evaluacion[['accuracy', 'f1_macro', 'f1_weighted']].notna().all().all()
comparacion.to_csv(SALIDAS / 'comparacion_total.csv', index=False)
evaluacion.to_csv(SALIDAS / 'evaluacion_total.csv', index=False)

resumen = []
for (tabla, etapa, objetivo), grupo in evaluacion.groupby(['tabla', 'etapa', 'objetivo']):
    elegido = grupo.loc[grupo.rol.eq('Elegido')].iloc[0]
    rf = grupo.loc[grupo.rol.eq('Referencia') & grupo.modelo.eq('RF_balanceado')].iloc[0]
    may = grupo.loc[grupo.rol.eq('Referencia') & grupo.modelo.eq('Mayoritaria')].iloc[0]
    assert elegido.filas_prueba == rf.filas_prueba == may.filas_prueba
    assert elegido.clases_prueba == rf.clases_prueba == may.clases_prueba
    resumen.append(dict(tabla=tabla, etapa=etapa, objetivo=objetivo, modelo=elegido.modelo,
        accuracy=elegido.accuracy, f1_macro=elegido.f1_macro, f1_weighted=elegido.f1_weighted,
        accuracy_rf=rf.accuracy, f1_macro_rf=rf.f1_macro,
        accuracy_mayoritaria=may.accuracy, f1_macro_mayoritaria=may.f1_macro,
        mejora_f1_macro=elegido.f1_macro-rf.f1_macro, filas_prueba=elegido.filas_prueba,
        clases_prueba=elegido.clases_prueba, clases_sin_entrenamiento=elegido.clases_sin_entrenamiento))
resumen = pd.DataFrame(resumen)
texto = ('# Resultados MGAP por tabla\n\nDivisiones aleatorias estratificadas; '
         'los años no intervienen. F1 macro sobre clases con ejemplos reales en prueba.\n')
parte = resumen.loc[resumen.etapa.eq('prueba_aleatoria')]
parte.to_csv(SALIDAS / 'resumen_aleatorio.csv', index=False)
texto += '\n## prueba_aleatoria\n\n| Tabla | Objetivo | Modelo | Acierto | F1 macro | F1 macro RF balanceado |\n|---|---|---|---:|---:|---:|\n'
for r in parte.itertuples():
    objetivo = 'Especialización' if r.objetivo == OBJETIVOS[0] else 'Tipo producción'
    texto += f'| {r.tabla} | {objetivo} | {r.modelo} | {r.accuracy:.1%} | {r.f1_macro:.4f} | {r.f1_macro_rf:.4f} |\n'
(SALIDAS / 'resumen_resultados.md').write_text(texto, encoding='utf-8')
for tabla in TABLAS:
    (SALIDAS / tabla / 'datos_comparacion_sha256.txt').write_text(perfil[tabla]['sha256'], encoding='utf-8')
print(resumen.round(4).to_string(index=False))
