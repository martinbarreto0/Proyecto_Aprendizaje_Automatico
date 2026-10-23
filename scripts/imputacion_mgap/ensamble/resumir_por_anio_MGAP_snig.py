"""Resume la prueba temporal y la compara con la prueba aleatoria, por tabla y por columna."""
import json
import sys
import pandas as pd
from funciones_ensamble_MGAP_snig import ANTERIORES, BASE, BASES, SALIDAS as ALEATORIO
from configuracion_MGAP_snig import TABLAS, OBJETIVOS

SALIDAS = BASE / 'resultados_por_anio'
filas = []
for tabla in TABLAS:
    archivo = SALIDAS / tabla / 'evaluacion_por_anio.csv'
    if not archivo.exists():
        print(f'{tabla}: sin resultados todavía')
        continue
    d = pd.read_csv(archivo)
    anterior = pd.read_csv(ANTERIORES / tabla / 'evaluacion_elegidos_aleatorio.csv')
    elegidos = json.loads((ALEATORIO / tabla / 'ensambles_elegidos.json').read_text(encoding='utf-8'))
    aleatorio = pd.read_csv(ALEATORIO / 'resumen_ensambles.csv')
    for objetivo in OBJETIVOS:
        modelo_ant = anterior[anterior.objetivo.eq(objetivo)].modelo.iloc[0]
        base_ant = 'HistBoost' if modelo_ant.startswith('HistBoost_') else modelo_ant
        ensamble = elegidos[objetivo]['ensamble']
        sel = d[d.objetivo.eq(objetivo)].set_index(['modelo', 'subconjunto'])
        if ensamble not in sel.index.get_level_values(0):
            continue  # p. ej. stacking omitido en esta corrida
        a = aleatorio[aleatorio.tabla.eq(tabla) & aleatorio.objetivo.eq(objetivo)].iloc[0]
        fila = dict(tabla=tabla, objetivo=objetivo, modelo_anterior=modelo_ant, ensamble=ensamble,
                    delta_aleatorio=a.delta_f1_macro,
                    pct_gemelo=sel.loc[(base_ant, 'con_gemelo'), 'filas'] / sel.loc[(base_ant, 'todas'), 'filas'])
        for sub in ['todas', 'sin_gemelo']:
            fila[f'f1_anterior_{sub}'] = sel.loc[(base_ant, sub), 'f1_macro']
            fila[f'f1_ensamble_{sub}'] = sel.loc[(ensamble, sub), 'f1_macro']
            fila[f'delta_{sub}'] = fila[f'f1_ensamble_{sub}'] - fila[f'f1_anterior_{sub}']
        # El mejor modelo (base o ensamble) en 2025 sin gemelos, solo como referencia.
        sin = sel.xs('sin_gemelo', level='subconjunto').f1_macro
        fila['mejor_sin_gemelo'] = f'{sin.idxmax()} ({sin.max():.4f})'
        fila.update({f'knn_{sub}': sel.loc[('KNN', sub), 'f1_macro'] for sub in ['todas', 'sin_gemelo']})
        filas.append(fila)

if not filas:
    raise SystemExit('No hay resultados para resumir.')
r = pd.DataFrame(filas)
r.to_csv(SALIDAS / 'resumen_por_anio.csv', index=False)
n = lambda v: f'{v:.4f}'.replace('.', ',')
s = lambda v: ('+' if v >= 0 else '') + n(v)
L = ['# Prueba temporal: entrenar 2022–2024, medir en 2025', '',
     'F1 macro. "Sin gemelo" = filas de 2025 sin una fila idéntica en 2022–2024 (la medida más estricta).',
     'Ensamble = el elegido en la validación aleatoria. Δ = ensamble − modelo anterior.', '',
     '## Por tabla y columna', '',
     '| Tabla | Columna | % con gemelo | Anterior (todas / sin gemelo) | Ensamble | Ensamble (todas / sin gemelo) '
     '| Δ aleatorio | Δ 2025 todas | Δ 2025 sin gemelo |',
     '|---|---|---:|---:|---|---:|---:|---:|---:|']
for f in filas:
    L.append(f"| {f['tabla']} | {f['objetivo']} | {100 * f['pct_gemelo']:.0f}% | "
             f"{n(f['f1_anterior_todas'])} / {n(f['f1_anterior_sin_gemelo'])} | {f['ensamble']} | "
             f"{n(f['f1_ensamble_todas'])} / {n(f['f1_ensamble_sin_gemelo'])} | {s(f['delta_aleatorio'])} | "
             f"{s(f['delta_todas'])} | {s(f['delta_sin_gemelo'])} |")
for titulo, clave in [('Por columna (promedio sobre las tablas)', 'objetivo'),
                      ('Por tabla (promedio de las dos columnas)', 'tabla')]:
    g = r.groupby(clave)[['delta_aleatorio', 'delta_todas', 'delta_sin_gemelo']].mean()
    L += ['', f'## {titulo}', '', f'| {clave.capitalize()} | Δ aleatorio | Δ 2025 todas | Δ 2025 sin gemelo |',
          '|---|---:|---:|---:|']
    L += [f'| {k} | {s(v.delta_aleatorio)} | {s(v.delta_todas)} | {s(v.delta_sin_gemelo)} |' for k, v in g.iterrows()]
L += ['', '## KNN y mejor modelo en 2025 sin gemelo', '',
      '| Tabla | Columna | KNN (todas / sin gemelo) | Mejor sin gemelo |', '|---|---|---:|---|']
L += [f"| {f['tabla']} | {f['objetivo']} | {n(f['knn_todas'])} / {n(f['knn_sin_gemelo'])} | {f['mejor_sin_gemelo']} |"
      for f in filas]
(SALIDAS / 'resumen_por_anio.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print('\n'.join(L))
