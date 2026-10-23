"""Columnas de cada tabla: las listas son explícitas, no se mezclan archivos."""

CATEGORICAS_COMUNES = ['GiroCodigo', 'DepartamentoCodigo', 'NaturalezaJuridicaCodigo']
OBJETIVOS = ['EspecializacionMGAPCodigo', 'TipoProduccionMGAPCodigo']

TABLAS = {
    'DatosAnimales': {
        'categoricas': CATEGORICAS_COMUNES + ['EspecieCodigo'],
        'numericas': ['EstratoCodigo', 'AnimalesPorEspecieConsumoPD_AD',
                      'AnimalesPorEspecieNacimientosPD_AD', 'AnimalesPorEspecieMortandadPD_AD',
                      'CantidadTenedoresPorEspecie', 'AnimalesPorEspeciePD_AD',
                      'AnimalesPorEspeciePD_PF'],
        'proporciones': {
            'ConsumoProporcion': ('AnimalesPorEspecieConsumoPD_AD', 'AnimalesPorEspeciePD_AD'),
            'NacimientosProporcion': ('AnimalesPorEspecieNacimientosPD_AD', 'AnimalesPorEspeciePD_AD'),
            'MortandadProporcion': ('AnimalesPorEspecieMortandadPD_AD', 'AnimalesPorEspeciePD_AD'),
            'PFProporcion': ('AnimalesPorEspeciePD_PF', 'AnimalesPorEspeciePD_AD'),
            'AnimalesPorTenedor': ('AnimalesPorEspeciePD_AD', 'CantidadTenedoresPorEspecie'),
        },
    },
    'DatosGenerales': {
        'categoricas': CATEGORICAS_COMUNES,
        'numericas': ['EstratoCodigo', 'Superficie', 'UnidadesGanaderas',
                      'SuperficieGanadera', 'CantidadTenedores'],
        'proporciones': {
            'ProporcionSuperficieGanadera': ('SuperficieGanadera', 'Superficie'),
            'UGPorHectareaGanadera': ('UnidadesGanaderas', 'SuperficieGanadera'),
            'SuperficiePorTenedor': ('Superficie', 'CantidadTenedores'),
            'UGPorTenedor': ('UnidadesGanaderas', 'CantidadTenedores'),
        },
    },
    'DatosProduccionLeche': {
        'categoricas': CATEGORICAS_COMUNES + ['EspecieCodigo', 'TipoProduccionCodigo'],
        'numericas': ['EstratoCodigo', 'Litros', 'CantidadTenedoresPorProduccionLeche'],
        'proporciones': {
            'LitrosPorTenedor': ('Litros', 'CantidadTenedoresPorProduccionLeche'),
        },
    },
    'DatosProduccionLecheEnEstablecimiento': {
        'categoricas': CATEGORICAS_COMUNES + ['EspecieCodigo', 'ProductoCodigo'],
        'numericas': ['EstratoCodigo', 'Litros'],
        'proporciones': {},
    },
    'DatosTenenciasTierra': {
        'categoricas': CATEGORICAS_COMUNES + ['TenenciaCodigo'],
        'numericas': ['EstratoCodigo', 'Hectareas'],
        'proporciones': {},
    },
    'DatosUsosTierra': {
        'categoricas': CATEGORICAS_COMUNES + ['UsoCodigo'],
        'numericas': ['EstratoCodigo', 'Hectareas'],
        'proporciones': {},
    },
    'DatosAnimalesDetallados': {
        'categoricas': CATEGORICAS_COMUNES + ['EspecieCodigo', 'CategoriaCodigo'],
        'numericas': ['EstratoCodigo', 'AnimalesPorCategoriaPD_AD', 'AnimalesPorCategoriaPD_PF'],
        'proporciones': {
            'RelacionPF_AD': ('AnimalesPorCategoriaPD_PF', 'AnimalesPorCategoriaPD_AD'),
        },
    },
}
