"""Consolida las métricas de la evaluación final de los tres métodos MGAP."""
from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / 'resultados_finales'
archivos = sorted(RESULTADOS.glob('*/metricas_finales.csv'))
if len(archivos) != 7:
    raise FileNotFoundError(f'Se esperaban 7 archivos de métricas y hay {len(archivos)}')

datos = pd.concat([pd.read_csv(archivo) for archivo in archivos], ignore_index=True)
if len(datos) != 42:
    raise ValueError(f'Se esperaban 42 filas (7 tablas × 2 objetivos × 3 métodos) y hay {len(datos)}')

datos['grupo'] = datos['metodo'].str.extract(r'^(Original|Hiperparametros|Ensamble)')
resumen = datos.pivot(index=['tabla', 'objetivo'], columns='grupo', values='f1_macro').reset_index()
resumen['delta_hiperparametros'] = resumen['Hiperparametros'] - resumen['Original']
resumen['delta_ensamble'] = resumen['Ensamble'] - resumen['Original']
resumen['metodo_ganador'] = resumen[['Original', 'Hiperparametros', 'Ensamble']].idxmax(axis=1)
resumen['f1_macro_ganador'] = resumen[['Original', 'Hiperparametros', 'Ensamble']].max(axis=1)
resumen.to_csv(RESULTADOS / 'resumen_comparacion_final.csv', index=False)

lineas = [
    '# Comparación final de modelos MGAP', '',
    'F1 macro sobre la misma división aleatoria congelada para los tres métodos.', '',
    '| Tabla | Objetivo | Original | Hiperparámetros | Δ hiperparámetros | Ensamble | Δ ensamble | Ganador |',
    '|---|---|---:|---:|---:|---:|---:|---|',
]
for _, fila in resumen.iterrows():
    lineas.append(
        f"| {fila.tabla} | {fila.objetivo} | {fila.Original:.4f} | "
        f"{fila.Hiperparametros:.4f} | {fila.delta_hiperparametros:+.4f} | "
        f"{fila.Ensamble:.4f} | {fila.delta_ensamble:+.4f} | {fila.metodo_ganador} |"
    )
(RESULTADOS / 'resumen_comparacion_final.md').write_text('\n'.join(lineas) + '\n', encoding='utf-8')
print(f'Resumen creado con {len(resumen)} objetivos en {RESULTADOS}')
