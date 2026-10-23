"""Congela los modelos ya elegidos para la evaluación final comparativa.

No entrena modelos ni lee los CSV de SNIG. Copia las configuraciones elegidas
previamente para impedir que una modificación posterior cambie la comparación.
"""
from hashlib import sha256
from pathlib import Path
import json


BASE = Path(__file__).resolve().parent
MGAP = BASE.parent
BUSQUEDA = MGAP / 'modelos_busqueda_hiperparametros' / 'resultados'
ENSAMBLES = MGAP / 'ensamble' / 'resultados_ensamble'
SALIDA = BASE / 'modelos_congelados.json'

# Semilla nueva: no se usó en las evaluaciones previas de modelos MGAP.
SEMILLA_FINAL = 20261007


def huella(ruta):
    """Devuelve una huella para documentar exactamente qué archivo se congeló."""
    return sha256(ruta.read_bytes()).hexdigest()


tablas = {}
for carpeta in sorted(p for p in BUSQUEDA.iterdir() if p.is_dir()):
    tabla = carpeta.name
    ruta_busqueda = carpeta / 'mejores_parametros.json'
    ruta_ensamble = ENSAMBLES / tabla / 'ensambles_elegidos.json'
    if not ruta_busqueda.exists() or not ruta_ensamble.exists():
        raise FileNotFoundError(f'{tabla}: faltan resultados de búsqueda o ensamble')

    busqueda = json.loads(ruta_busqueda.read_text(encoding='utf-8'))['objetivos']
    ensamble = json.loads(ruta_ensamble.read_text(encoding='utf-8'))
    tablas[tabla] = {
        'objetivos': {
            objetivo: {
                # Modelo individual vigente antes de la búsqueda.
                'modelo_original': info['modelo_actual'],
                'familia_original': (
                    'RF' if info['modelo_actual'].startswith('RF_') else
                    'ExtraTrees' if info['modelo_actual'].startswith('ExtraTrees_') else
                    'HistBoost'
                ),
                'parametros_originales': info['parametros_actuales'],
                # Mejor configuración seleccionada por la búsqueda, aunque luego no
                # haya recomendado reemplazar el modelo en su test anterior.
                'familia_hiperparametros': info['familia_elegida'],
                'parametros_hiperparametros': info['parametros_elegidos'],
                # Ensamble elegido antes de conocer el test final nuevo.
                'ensamble': ensamble[objetivo],
            }
            for objetivo, info in busqueda.items()
        },
        'fuentes': {
            'busqueda': str(ruta_busqueda.relative_to(MGAP)),
            'busqueda_sha256': huella(ruta_busqueda),
            'ensamble': str(ruta_ensamble.relative_to(MGAP)),
            'ensamble_sha256': huella(ruta_ensamble),
        },
    }

manifest = {
    'descripcion': 'Comparación final: original, hiperparámetros y ensamble congelados.',
    'semilla_train_test_split': SEMILLA_FINAL,
    'test_size': 0.2,
    'tablas': tablas,
}
SALIDA.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Manifiesto creado: {SALIDA}')
print(f'Tablas: {len(tablas)}; semilla final: {SEMILLA_FINAL}')
