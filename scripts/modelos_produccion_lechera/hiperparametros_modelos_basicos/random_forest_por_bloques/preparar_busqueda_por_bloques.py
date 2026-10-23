"""Prepara el manifiesto, las huellas y los pliegues de Random Forest."""
import argparse

from utilidades_bloques import (
    ARCHIVO_BUSQUEDA_ORIGINAL,
    ARCHIVO_FUNCIONES_ORIGINAL,
    ASIGNACION_PLIEGUES,
    CONFIGURACIONES_POR_BLOQUE,
    MANIFIESTO,
    METADATOS,
    PLIEGUES,
    RESULTADOS_BLOQUES,
    SKLEARN_VERSION,
    TOTAL_BLOQUES,
    TOTAL_CONFIGURACIONES,
    cargar_plan,
    crear_asignacion_pliegues,
    crear_manifiesto,
    fecha_utc,
    guardar_csv_atomico,
    guardar_json_atomico,
    hash_archivo,
    hash_grid,
    huella_dataframe,
    preparar_datos,
    ruta_resultado_bloque,
    verificar_datos,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Regenera el plan solo si todavía no existen resultados de bloques.",
    )
    args = parser.parse_args()

    resultados_existentes = [
        ruta_resultado_bloque(numero)
        for numero in range(1, TOTAL_BLOQUES + 1)
        if ruta_resultado_bloque(numero).exists()
    ]
    if args.forzar and resultados_existentes:
        raise RuntimeError(
            "No se puede regenerar el plan porque ya existen bloques terminados."
        )

    entrenamiento, validacion, test_reservado = preparar_datos()
    manifiesto = crear_manifiesto()
    asignacion = crear_asignacion_pliegues(len(entrenamiento))
    metadatos_nuevos = {
        "creado_utc": fecha_utc(),
        "total_bloques": TOTAL_BLOQUES,
        "configuraciones_por_bloque": CONFIGURACIONES_POR_BLOQUE,
        "total_configuraciones": TOTAL_CONFIGURACIONES,
        "pliegues": PLIEGUES,
        "ajustes_totales": TOTAL_CONFIGURACIONES * PLIEGUES,
        "sha256_grid": hash_grid(manifiesto),
        "filas_entrenamiento": len(entrenamiento),
        "filas_validacion": len(validacion),
        "filas_test": len(test_reservado),
        "sha256_entrenamiento": huella_dataframe(entrenamiento),
        "sha256_validacion": huella_dataframe(validacion),
        "sha256_test": huella_dataframe(test_reservado),
        "sha256_funciones_busqueda": hash_archivo(ARCHIVO_FUNCIONES_ORIGINAL),
        "sha256_busqueda_monolitica": hash_archivo(ARCHIVO_BUSQUEDA_ORIGINAL),
        "version_sklearn": SKLEARN_VERSION,
        "test_evaluado": False,
    }

    if MANIFIESTO.exists() and ASIGNACION_PLIEGUES.exists() and METADATOS.exists():
        if not args.forzar:
            _, _, metadatos_existentes = cargar_plan()
            verificar_datos(
                metadatos_existentes,
                entrenamiento,
                validacion,
                test_reservado,
            )
            print("El plan existente es válido para el código y los datos actuales.")
            print(f"Ubicación: {RESULTADOS_BLOQUES}")
            return

    guardar_csv_atomico(manifiesto, MANIFIESTO)
    guardar_csv_atomico(asignacion, ASIGNACION_PLIEGUES)
    guardar_json_atomico(metadatos_nuevos, METADATOS)

    print(f"Configuraciones: {len(manifiesto)}")
    print(f"Bloques: {TOTAL_BLOQUES}")
    print(f"Configuraciones por bloque: {CONFIGURACIONES_POR_BLOQUE}")
    print(f"Ajustes de CV totales: {TOTAL_CONFIGURACIONES * PLIEGUES}")
    print(f"Test reservado sin evaluar: {len(test_reservado)} filas")
    print(f"Plan guardado en: {RESULTADOS_BLOQUES}")


if __name__ == "__main__":
    main()
