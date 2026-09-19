# Datasets SNIG con códigos MGAP completados

Esta carpeta contiene copias de las siete tablas de `data/processed/SNIG`. Los archivos fuente permanecen intactos.

En las filas donde faltaban simultáneamente `EspecializacionMGAPCodigo` y `TipoProduccionMGAPCodigo`, ambos valores fueron estimados con el mejor modelo encontrado para esa tabla y objetivo. Las filas con valores MGAP conocidos se conservaron sin cambios.

Se agregaron dos columnas al final de cada CSV:

- `EspecializacionMGAPConfianza`.
- `TipoProduccionMGAPConfianza`.

La confianza es la máxima probabilidad que devolvió el modelo para la clase elegida, entre 0 y 1, con seis decimales. Se deja vacía en las filas observadas. No es una probabilidad calibrada, por lo que sirve como señal relativa de seguridad del modelo y no como garantía de acierto.

| Archivo | Filas | Filas completadas |
|---|---:|---:|
| DatosAnimales.csv | 677.035 | 357.861 |
| DatosAnimalesDetallados.csv | 2.557.190 | 1.446.265 |
| DatosGenerales.csv | 315.536 | 168.179 |
| DatosProduccionLeche.csv | 59.127 | 38.191 |
| DatosProduccionLecheEnEstablecimiento.csv | 4.789 | 3.090 |
| DatosTenenciasTierra.csv | 404.566 | 222.703 |
| DatosUsosTierra.csv | 602.486 | 328.872 |

La subcarpeta `metadatos` contiene un JSON por tabla con el modelo usado, los conteos y el mínimo, máximo y promedio de cada confianza.

Para volver a generar y verificar los archivos desde la raíz del proyecto:

```powershell
python PAA/scripts/imputacion_mgap/aplicar_modelos_MGAP_snig.py
python PAA/scripts/imputacion_mgap/verificar_imputacion_MGAP_snig.py
```

La verificación comprueba en bloques que todas las filas y datos originales se conserven, que los códigos faltantes hayan sido completados y que las confianzas sean válidas.
