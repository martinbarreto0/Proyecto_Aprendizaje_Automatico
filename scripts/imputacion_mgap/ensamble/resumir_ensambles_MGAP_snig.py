"""Compara el ensamble elegido contra el modelo elegido anteriormente, por tabla y por columna."""
import sys
import pandas as pd
from funciones_ensamble_MGAP_snig import ANTERIORES, SALIDAS, BASES
from configuracion_MGAP_snig import TABLAS, OBJETIVOS

filas, bases = [], []
for tabla in TABLAS:
    nuevo_csv = SALIDAS / tabla / 'evaluacion_ensambles.csv'
    if not nuevo_csv.exists():
        print(f'{tabla}: sin resultados de ensamble todavía')
        continue
    nuevo = pd.read_csv(nuevo_csv)
    anterior = pd.read_csv(ANTERIORES / tabla / 'evaluacion_elegidos_aleatorio.csv')
    for objetivo in OBJETIVOS:
        a = anterior[anterior.objetivo.eq(objetivo)].iloc[0]
        e = nuevo[nuevo.objetivo.eq(objetivo) & nuevo.rol.eq('Ensamble')].iloc[0]
        # Debe ser exactamente el mismo 20% de prueba.
        assert a.filas_prueba == e.filas_prueba, f'{tabla} {objetivo}: pruebas distintas'
        # El modelo anterior se vuelve a entrenar como base en este mismo entorno; la
        # comparación usa esa cifra para que la versión de las librerías no influya.
        base = 'HistBoost' if a.modelo.startswith('HistBoost_') else a.modelo
        r = nuevo[nuevo.objetivo.eq(objetivo) & nuevo.modelo.eq(base)].iloc[0]
        if base == 'HistBoost':
            assert r.histboost == a.modelo, f'{tabla} {objetivo}: otra variante de HistBoost'
        filas.append(dict(tabla=tabla, objetivo=objetivo, modelo_anterior=a.modelo,
            f1_macro_anterior_archivo=a.f1_macro,
            accuracy_anterior=r.accuracy, f1_macro_anterior=r.f1_macro,
            ensamble=e.modelo, accuracy_ensamble=e.accuracy, f1_macro_ensamble=e.f1_macro,
            delta_accuracy=e.accuracy - r.accuracy, delta_f1_macro=e.f1_macro - r.f1_macro,
            filas_prueba=e.filas_prueba))
        b = nuevo[nuevo.objetivo.eq(objetivo) & nuevo.rol.eq('Base')].set_index('modelo').f1_macro
        bases.append(dict(tabla=tabla, objetivo=objetivo, **b.reindex(BASES).to_dict(),
                          ensamble=e.f1_macro))

if not filas:
    raise SystemExit('No hay resultados para resumir.')
resumen = pd.DataFrame(filas)
resumen.to_csv(SALIDAS / 'resumen_ensambles.csv', index=False)
por_columna = resumen.groupby('objetivo')[['delta_accuracy', 'delta_f1_macro']].mean()
por_tabla = resumen.groupby('tabla')[['delta_accuracy', 'delta_f1_macro']].mean()
mejoras = resumen.groupby('objetivo').delta_f1_macro.apply(lambda d: f'{(d > 0).sum()} de {len(d)}')

n = lambda v: f'{v:.4f}'.replace('.', ',')
p = lambda v: f'{100 * v:.2f}%'.replace('.', ',')
s = lambda v: ('+' if v >= 0 else '') + n(v)
lineas = ['# Ensambles vs. modelos elegidos anteriormente', '',
          'Mismo 20% de prueba aleatorio (semilla 42). Cada celda: `acierto / F1 macro`.',
          'El modelo anterior se reentrenó en el mismo entorno que los ensambles; la columna',
          '"F1 archivo" es la cifra guardada en `resultados_mgap_snig` como referencia.', '',
          '## Por tabla y columna', '',
          '| Tabla | Columna | Anterior | F1 archivo | Acierto / F1 | Ensamble | Acierto / F1 | Δ F1 macro |',
          '|---|---|---|---:|---:|---|---:|---:|']
for f in filas:
    lineas.append(f"| {f['tabla']} | {f['objetivo']} | {f['modelo_anterior']} | "
                  f"{n(f['f1_macro_anterior_archivo'])} | "
                  f"{p(f['accuracy_anterior'])} / {n(f['f1_macro_anterior'])} | {f['ensamble']} | "
                  f"{p(f['accuracy_ensamble'])} / {n(f['f1_macro_ensamble'])} | {s(f['delta_f1_macro'])} |")
lineas += ['', '## Por columna (promedio sobre las tablas)', '',
           '| Columna | Δ acierto | Δ F1 macro | Celdas que mejoran |', '|---|---:|---:|---:|']
for objetivo, f in por_columna.iterrows():
    lineas.append(f'| {objetivo} | {s(f.delta_accuracy)} | {s(f.delta_f1_macro)} | {mejoras[objetivo]} |')
lineas += ['', '## Por tabla (promedio de las dos columnas)', '',
           '| Tabla | Δ acierto | Δ F1 macro |', '|---|---:|---:|']
for tabla, f in por_tabla.iterrows():
    lineas.append(f'| {tabla} | {s(f.delta_accuracy)} | {s(f.delta_f1_macro)} |')
lineas += ['', '## F1 macro de cada base y del ensamble en la prueba', '',
           '| Tabla | Columna | ' + ' | '.join(BASES) + ' | Ensamble |',
           '|---|---|' + '---:|' * (len(BASES) + 1)]
for b in bases:
    lineas.append(f"| {b['tabla']} | {b['objetivo']} | " +
                  ' | '.join(n(b[m]) for m in BASES) + f" | {n(b['ensamble'])} |")
(SALIDAS / 'resumen_ensambles.md').write_text('\n'.join(lineas) + '\n', encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')  # la consola de Windows no imprime "Δ" en cp1252
print('\n'.join(lineas))
