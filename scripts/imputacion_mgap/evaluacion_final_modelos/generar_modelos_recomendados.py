"""Genera el documento con el mejor modelo final por tabla y objetivo MGAP."""
from pathlib import Path
import json
import pandas as pd


BASE = Path(__file__).resolve().parent
MANIFIESTO = BASE / 'modelos_congelados.json'
RESULTADOS = BASE / 'resultados_finales'
RESUMEN = RESULTADOS / 'resumen_comparacion_final.csv'
SALIDA = RESULTADOS / 'modelos_recomendados.md'

# Los nombres se mantienen en un único lugar para explicar el ensamble elegido.
TIPOS_ENSAMBLE = {
    'E1_voto_duro': 'E1 — voto duro: mayoría de votos; los empates se resuelven con el promedio de probabilidades.',
    'E2_voto_blando': 'E2 — voto blando: promedio de probabilidades.',
    'E3_voto_blando_arboles': 'E3 — voto blando de árboles: promedio de RF, ExtraTrees e HistBoost.',
    'E4_stacking': 'E4 — stacking: regresión logística entrenada sobre las probabilidades fuera de pliegue.',
}


def formato_parametros(parametros):
    """Devuelve los parámetros de un modelo individual en una sola línea legible."""
    return '; '.join(f'{nombre}={valor}' for nombre, valor in parametros.items())


if not MANIFIESTO.exists() or not RESUMEN.exists():
    raise FileNotFoundError('Faltan el manifiesto o el resumen de la comparación final.')

manifest = json.loads(MANIFIESTO.read_text(encoding='utf-8'))
resumen = pd.read_csv(RESUMEN).set_index(['tabla', 'objetivo'])
lineas = [
    '# Modelos recomendados para la imputación MGAP', '',
    'Selección por mayor F1 macro en la comparación final aleatoria. Cada fila compara',
    'el modelo original, los hiperparámetros y el ensamble sobre las mismas filas.', '',
    '## Tipos de ensamble', '',
    '- **E1:** voto duro de KNN, Random Forest, ExtraTrees e HistBoost.',
    '- **E2:** voto blando de KNN, Random Forest, ExtraTrees e HistBoost.',
    '- **E3:** voto blando de Random Forest, ExtraTrees e HistBoost; no incluye KNN.',
    '- **E4:** stacking con regresión logística sobre las probabilidades de KNN, Random Forest, ExtraTrees e HistBoost.', '',
    '## Configuración fija de los componentes de ensamble', '',
    '- **KNN:** `StandardScaler`; 5 vecinos; `weights=distance`; muestra proporcional por clase de hasta 200.000 filas.',
    '- **RF_suavizado:** 100 árboles; `max_depth=20`; `min_samples_leaf=3`; `exponente_pesos=0.5`.',
    '- **ExtraTrees_suavizado:** 100 árboles; `max_depth=25`; `min_samples_leaf=3`; `exponente_pesos=0.5`.',
    '- **HistBoost:** la variante, exponente de pesos e iteraciones se indican en cada fila; `learning_rate=0.1`; `max_leaf_nodes=31`; `l2_regularization=1`.',
    '- **Capa de E4:** regresión logística con `max_iter=1000`, pesos con exponente 0,5 y tres pliegues estratificados fuera de pliegue.', '',
]

for tabla, datos_tabla in manifest['tablas'].items():
    lineas += [f'## {tabla}', '', '| Columna | Modelo recomendado | F1 macro | Configuración |', '|---|---|---:|---|']
    for objetivo, info in datos_tabla['objetivos'].items():
        resultado = resumen.loc[(tabla, objetivo)]
        ganador = resultado['metodo_ganador']
        f1 = resultado['f1_macro_ganador']
        if ganador == 'Hiperparametros':
            modelo = f"{info['familia_hiperparametros']} con hiperparámetros ajustados"
            detalle = formato_parametros(info['parametros_hiperparametros'])
        elif ganador == 'Original':
            modelo = f"{info['familia_original']} original ({info['modelo_original']})"
            detalle = formato_parametros(info['parametros_originales'])
        else:
            elegido = info['ensamble']
            nombre = elegido['ensamble']
            hb_nombre, hb_exponente, hb_iteraciones = elegido['histboost']
            modelo = TIPOS_ENSAMBLE[nombre].split(':')[0]
            detalle = (
                f"{TIPOS_ENSAMBLE[nombre]} Miembros: {', '.join(elegido['miembros'])}. "
                f"HistBoost: {hb_nombre}; exponente_pesos={hb_exponente}; max_iter={hb_iteraciones}."
            )
        lineas.append(f'| {objetivo} | {modelo} | {f1:.4f} | {detalle} |')
    lineas.append('')

SALIDA.write_text('\n'.join(lineas), encoding='utf-8')
print(f'Documento creado: {SALIDA}')
