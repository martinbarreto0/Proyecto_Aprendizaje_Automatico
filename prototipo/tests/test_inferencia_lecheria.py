"""Pruebas del pipeline de inferencia del prototipo (guía PAA, apartado 4.5).

Cubren el esquema de entrada, la carga de artefactos, el flujo completo de
preprocesamiento e inferencia y el formato de las salidas. Ejecutar desde la
raíz del repositorio con:  python -m pytest PAA/prototipo/tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inferencia_lecheria as inf  # noqa: E402

CARPETA_PAA = Path(__file__).resolve().parents[2]
PREDICCIONES_ORIGINALES = (
    CARPETA_PAA / "reports" / "modelos_catboost" / "seleccion_variables" / "predicciones_top_12.csv"
)
VARIABLES_ORIGINALES = (
    CARPETA_PAA / "reports" / "modelos_catboost" / "seleccion_variables" / "variables_top_12.csv"
)


@pytest.fixture(scope="module")
def modelo():
    return inf.cargar_modelo()


@pytest.fixture(scope="module")
def metadatos():
    return inf.cargar_metadatos()


@pytest.fixture(scope="module")
def dataset():
    return inf.cargar_dataset()


@pytest.fixture
def perfil_valido():
    """Perfil lechero mediano escrito a mano, como lo ingresaría un usuario."""
    return pd.DataFrame([{
        "total_bovinos_leche": 320, "toros": 4, "vacas_ordene": 120, "vacas_secas": 40,
        "vaquillonas_mas_2": 30, "vaquillonas_1_2": 45, "terneros_menos_1": 20,
        "terneras_menos_1": 45, "Superficie": 280, "UnidadesGanaderas": 380,
        "SuperficieGanadera": 270,
    }])


# ------------------------------------------------------------- carga de artefactos
def test_modelo_carga_con_las_variables_del_entrenamiento(modelo):
    esperadas = pd.read_csv(VARIABLES_ORIGINALES, encoding="utf-8-sig")["variable"].tolist()
    assert list(modelo.feature_names_) == esperadas


def test_metadatos_contienen_umbrales_para_cada_banda(metadatos):
    assert set(metadatos["umbral_desvio_p90"]) == set(inf.BANDAS)
    assert metadatos["cortes_banda_litros"][0] < metadatos["cortes_banda_litros"][1]


def test_dataset_excluye_la_prueba_reservada(dataset, metadatos):
    assert not metadatos["prueba_incluida"]
    assert set(dataset["particion"]) == {"entrenamiento", "validacion"}
    assert len(dataset) == sum(metadatos["filas_por_particion"].values())


def test_modelo_inexistente_da_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="preparar_datos_prototipo"):
        inf.cargar_modelo(tmp_path / "no_existe.cbm")


# ------------------------------------------------------------- esquema de entrada
def test_falta_columna_requerida(modelo, perfil_valido):
    with pytest.raises(inf.ErrorEntrada, match="vacas_secas"):
        inf.validar_entrada(perfil_valido.drop(columns="vacas_secas"), modelo.feature_names_)


def test_valor_no_numerico(modelo, perfil_valido):
    perfil_valido["vacas_ordene"] = perfil_valido["vacas_ordene"].astype(object)
    perfil_valido.loc[0, "vacas_ordene"] = "ciento veinte"
    with pytest.raises(inf.ErrorEntrada, match="no numéricos"):
        inf.validar_entrada(perfil_valido, modelo.feature_names_)


def test_valor_negativo(modelo, perfil_valido):
    perfil_valido.loc[0, "toros"] = -1
    with pytest.raises(inf.ErrorEntrada, match="negativos"):
        inf.validar_entrada(perfil_valido, modelo.feature_names_)


def test_valor_infinito(modelo, perfil_valido):
    perfil_valido["Superficie"] = perfil_valido["Superficie"].astype("float64")
    perfil_valido.loc[0, "Superficie"] = np.inf
    with pytest.raises(inf.ErrorEntrada, match="infinitos"):
        inf.validar_entrada(perfil_valido, modelo.feature_names_)


def test_tabla_vacia_y_fila_sin_datos(modelo, perfil_valido):
    with pytest.raises(inf.ErrorEntrada, match="no contiene filas"):
        inf.validar_entrada(perfil_valido.iloc[0:0], modelo.feature_names_)
    vacia = pd.DataFrame([{c: np.nan for c in perfil_valido.columns}])
    with pytest.raises(inf.ErrorEntrada, match="sin ningún valor"):
        inf.validar_entrada(vacia, modelo.feature_names_)


def test_entrada_que_no_es_tabla(modelo):
    with pytest.raises(inf.ErrorEntrada, match="DataFrame"):
        inf.validar_entrada({"vacas_ordene": 10}, modelo.feature_names_)


def test_carga_ganadera_se_calcula_como_en_el_entrenamiento(modelo, perfil_valido):
    limpio = inf.validar_entrada(perfil_valido, modelo.feature_names_)
    assert limpio.loc[0, "carga_ganadera"] == pytest.approx(380 / 270)
    perfil_valido.loc[0, "SuperficieGanadera"] = 0
    limpio = inf.validar_entrada(perfil_valido, modelo.feature_names_)
    assert np.isnan(limpio.loc[0, "carga_ganadera"])


def test_faltantes_puntuales_se_admiten(modelo, metadatos, perfil_valido):
    perfil_valido.loc[0, "toros"] = np.nan
    salida = inf.predecir(modelo, perfil_valido, metadatos)
    assert np.isfinite(salida.loc[0, "litros_estimados"])


# ------------------------------------------------------------- flujo integral
def test_flujo_reproduce_las_predicciones_del_entrenamiento(modelo, dataset):
    """El pipeline debe dar, fila a fila, las mismas predicciones guardadas en validación."""
    originales = pd.read_csv(PREDICCIONES_ORIGINALES, encoding="utf-8-sig")
    validacion = dataset[dataset["particion"] == "validacion"].reset_index(drop=True)
    crudas = modelo.predict(inf.validar_entrada(validacion, modelo.feature_names_))
    np.testing.assert_allclose(crudas, originales["Litros_predichos"].to_numpy(), rtol=1e-9)


def test_flujo_completo_con_perfil_ingresado(modelo, metadatos, perfil_valido):
    salida = inf.predecir(modelo, perfil_valido, metadatos)
    assert len(salida) == 1
    assert salida.loc[0, "litros_estimados"] > 0


# ------------------------------------------------------------- formato de salida
def test_salidas_tienen_tipo_y_rango_esperados(modelo, metadatos, dataset):
    muestra = dataset.sample(500, random_state=1)
    salida = inf.predecir(modelo, muestra, metadatos)
    assert list(salida.index) == list(muestra.index)
    assert salida["litros_estimados"].dtype == "float64"
    assert np.isfinite(salida["litros_estimados"]).all()
    assert (salida["litros_estimados"] >= 0).all()
    assert set(salida["banda_confiabilidad"]) <= set(inf.BANDAS)
    assert salida["prediccion_recortada"].dtype == bool


def test_predicciones_negativas_se_recortan_a_cero(metadatos, perfil_valido):
    class ModeloNegativo:
        feature_names_ = list(inf.cargar_modelo().feature_names_)

        def predict(self, x):
            return np.full(len(x), -500.0)

    salida = inf.predecir(ModeloNegativo(), perfil_valido, metadatos)
    assert salida.loc[0, "litros_estimados"] == 0
    assert salida.loc[0, "prediccion_recortada"]


def test_evaluar_desvio_marca_atipicos(metadatos):
    umbral = metadatos["umbral_desvio_p90"]["alta"]
    normal = inf.evaluar_desvio(1_000_000, 1_000_000, "alta", metadatos)
    atipico = inf.evaluar_desvio(1_000_000 * (1 + umbral + 0.1), 1_000_000, "alta", metadatos)
    assert not normal["atipico"] and normal["desvio_relativo"] == 0
    assert atipico["atipico"]
    assert not inf.evaluar_desvio(100, 0, "baja", metadatos)["atipico"]


def test_metricas_de_prueba_coinciden_con_la_evaluacion_final():
    original = pd.read_csv(
        CARPETA_PAA / "reports" / "modelos_catboost" / "evaluacion_final_test" / "metricas_test.csv",
        encoding="utf-8-sig",
    ).iloc[0]
    metricas = inf.cargar_metricas_prueba()
    assert metricas["filas"] == int(original["filas"]) == 2431
    assert metricas["wape"] == pytest.approx(original["wape_porcentaje"])
    assert metricas["r2"] == pytest.approx(original["r2"])


def test_metricas_de_prueba_ausentes_devuelven_none(tmp_path):
    assert inf.cargar_metricas_prueba(tmp_path / "no_existe.csv") is None
