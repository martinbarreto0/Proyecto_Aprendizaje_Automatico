"""Busca hiperparámetros de CatBoost usando las 16 variables originales."""

import argparse

from funciones_busqueda_catboost import consolidar, ejecutar_bloque


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operacion = parser.add_mutually_exclusive_group(required=True)
    operacion.add_argument("--bloque", type=int, choices=range(1, 7))
    operacion.add_argument("--consolidar", action="store_true")
    parser.add_argument("--hilos", type=int, default=4)
    args = parser.parse_args()
    if args.consolidar:
        consolidar("completo", args.hilos)
    else:
        ejecutar_bloque("completo", args.bloque, args.hilos)


if __name__ == "__main__":
    main()
