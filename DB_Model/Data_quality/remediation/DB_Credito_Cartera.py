import pandas as pd
import numpy as np
from datetime import datetime


# =============================================================================
# 1. CARGA Y LIMPIEZA DE DATOS DE CRÉDITOS (CRq)
# =============================================================================

# Cargar la base de datos de créditos financieros
creditos_inicial = pd.read_excel(
    '../../data_raw/mafi/CRq - Creditos Financieros (Plano).xlsx'
)

# RM_1: Filtrar líneas que corresponden a crédito interno (entre 42 y 55)
creditos_inicial = creditos_inicial[
    (creditos_inicial['Linea'].between(42, 55))
]

# RM_2: Filtrar créditos cancelados o en cartera
creditos_inicial = creditos_inicial[
    creditos_inicial['Estado_credito'].isin(['Cancelado', 'En Cartera'])
]

# RM_3: Filtrar créditos que no tienen valor neto
creditos_inicial = creditos_inicial[~creditos_inicial['Vr_neto_matricula'].isnull()].copy()

# RM_4: Eliminar registros sin período de facturación
creditos_inicial = creditos_inicial.dropna(subset=['Periodo_facturacion'])
creditos_inicial['Periodo_facturacion'] = creditos_inicial['Periodo_facturacion'].astype(int)

# RM_5: Calcular antigüedad del crédito (en meses) -- Nueva columna
fecha_actual = pd.Timestamp.now()
creditos_inicial['Antiguedad_meses'] = (
    (fecha_actual - creditos_inicial['Fecha_aprobacion']).dt.days / 30.44
).astype(int)

# RM_6: Crear llaves únicas para análisis
creditos_inicial['Llave'] = (
    creditos_inicial['Cliente'].astype(str) + "_" +
    creditos_inicial['Periodo_facturacion'].astype(str)
)
creditos_inicial['Llave2'] = (
    creditos_inicial['Cliente'].astype(str) + "_" +
    creditos_inicial['Periodo_facturacion'].astype(str) + "_" +
    creditos_inicial['No_credito'].astype(str)
)

# RM_7: Seleccionar columnas relevantes para el análisis de crédito
cols_credito = [
    'Llave', 'Llave2', 'Nombre_linea', 'Fecha_aprobacion', 'Antiguedad_meses',
    'Nombre_fondo', 'Valor_financiacion', 'Cuotas', 'Tipo_interes'
]

df_credito = creditos_inicial[cols_credito].copy().reset_index(drop=True)


# =============================================================================
# 2. CARGA Y LIMPIEZA DE DATOS DE CARTERA (CCq)
# =============================================================================

# Cargar base de cartera de edades
cartera_inicial = pd.read_excel(
    '../../data_raw/mafi/CCq - Cartera edades cliente (Plano).xlsx'
)

# RM_1: Filtrar concepto asociado a crédito interno
cartera_inicial = cartera_inicial[
    cartera_inicial['Concepto'] == '118/19 CARGO DE LA FINANCIACION MATRICULAS'
]

# RM_2: Convertir columna Genera_mora a tipo booleano
cartera_inicial['Genera_mora'] = cartera_inicial['Genera_mora'].map({'S': True, 'N': False})

# RM3: Convertir tipos de datos para unir llaves
cartera_inicial['Periodo'] = cartera_inicial['Periodo'].astype(int)
cartera_inicial['Credito'] = cartera_inicial['Credito'].astype(int)

# RM_4: Crear llaves únicas para análisis
cartera_inicial['Llave'] = (
    cartera_inicial['Cliente'].astype(str) + "_" +
    cartera_inicial['Periodo'].astype(str)
)
cartera_inicial['Llave2'] = (
    cartera_inicial['Cliente'].astype(str) + "_" +
    cartera_inicial['Periodo'].astype(str) + "_" +
    cartera_inicial['Credito'].astype(str)
)

# RM_5: Seleccionar columnas relevantes de cartera
cols_cartera = [
    'Llave', 'Llave2', 'Tipo_cartera', 'Nombre_periodo',
    'Genera_mora', 'Tasa_interes_mora'
]

df_cartera = cartera_inicial[cols_cartera].copy().reset_index(drop=True)


# =============================================================================
# 3. UNIÓN DE CRÉDITO Y CARTERA
# =============================================================================

# Copiar DataFrame de crédito
df_credito_cartera = df_credito.copy()

# Contar cuotas vencidas por Llave2
cuotas_vencidas = (
    df_cartera[df_cartera['Tipo_cartera'] == 'VENCIDA']
    .groupby('Llave2')
    .size()
)
df_credito_cartera['Cuotas_vencidas'] = (
    df_credito_cartera['Llave2'].map(cuotas_vencidas)
    .fillna(0).astype(int)
)

# Contar cuotas por vencer por Llave2
cuotas_por_vencer = (
    df_cartera[df_cartera['Tipo_cartera'] == 'POR VENCER']
    .groupby('Llave2')
    .size()
)
df_credito_cartera['Cuotas_por_vencer'] = (
    df_credito_cartera['Llave2'].map(cuotas_por_vencer)
    .fillna(0).astype(int)
)


# =============================================================================
# 4. CREACIÓN DE VARIABLES OBJETIVO (Y)
# =============================================================================

# Variable continua: proporción de cuotas vencidas sobre cuotas activas
df_credito_cartera['Y_continua'] = np.where(
    (df_credito_cartera['Cuotas'] - df_credito_cartera['Cuotas_por_vencer']) > 0,
    df_credito_cartera['Cuotas_vencidas'] /
    (df_credito_cartera['Cuotas'] - df_credito_cartera['Cuotas_por_vencer']),
    0
)

df_credito_cartera['Y_continua'] = df_credito_cartera['Y_continua'].round(2)

# Variable categórica: segmentación del riesgo de mora
df_credito_cartera['Y_categorica'] = pd.cut(
    df_credito_cartera['Y_continua'],
    bins=[-0.001, 0.20, 0.25, 1.00],
    labels=['Bajo', 'Medio', 'Alto']
)


# =============================================================================
# 5. EXPORTACIÓN DE RESULTADOS
# =============================================================================

# Exportar dataset limpio con variable respuesta
df_credito_cartera.to_csv(
    '../output/Credito_Cartera.csv',
    index=False,
    encoding='utf-8-sig'
)
