import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# BASE SCORING DE RIESGO Y COMPLIANCE
# =============================================================================

# Cargar base inicial
scoring_inicial = pd.read_excel("../../data_raw/mafi/Matriz total 2021-2025 .xlsx")

# Filtrar categorías que pasaron a la siguiente fase del crédito
scoring_inicial = scoring_inicial[
    scoring_inicial["Cate"].isin(["Sello Financiero", "Crédito aprobado"])
]

# Convertir columna "Mora" a booleano
scoring_inicial["Mora"] = scoring_inicial["Mora"].str.strip().str.lower()
scoring_inicial["Mora"] = scoring_inicial["Mora"].map({"si": True, "no": False})

# Crear llave única por estudiante y periodo
scoring_inicial["Llave"] = (
    scoring_inicial["ID Estudiante"].astype(str)
    + "_"
    + scoring_inicial["Periodo"].astype(str)
)

# Filtrar registros con "Valor_cuota_inicial" nulo
df_nulos_cuota = scoring_inicial[
    scoring_inicial["Valor_cuota_inicial"].isnull()
][
    [
        "Llave",
        "Valor_Matricula",
        "Valor_Financiado",
        "Valor_cuota_inicial",
        "Valor_Primera cuota",
    ]
]

# Imputar valores nulos de cuota inicial y primera cuota
scoring_inicial["Valor_cuota_inicial"] = scoring_inicial[
    "Valor_cuota_inicial"
].fillna(scoring_inicial["Valor_Primera cuota"])

scoring_inicial["Valor_Primera cuota"] = scoring_inicial[
    "Valor_Primera cuota"
].fillna(scoring_inicial["Valor_cuota_inicial"])

scoring_inicial = scoring_inicial.rename(
    columns={"Valor_Primera cuota": "Valor_primera_cuota"}
)

# Cargar CSV de validación de cartera
val_cartera = pd.read_csv("../../data_raw/mafi/val_Cartera.csv")

df_fecha = val_cartera[["Llave", "Fecha_vence"]]
df_fecha["Fecha_vence"] = pd.to_datetime(df_fecha["Fecha_vence"], errors="coerce")

# Extraer día estimado de pago
df_fecha["dia_pago_estimado"] = df_fecha["Fecha_vence"].dt.day

df_diapago = (
    df_fecha.groupby("Llave")["dia_pago_estimado"].first().reset_index()
)

# Merge con tabla principal
scoring_inicial = scoring_inicial.merge(df_diapago, on="Llave", how="left")

scoring_inicial["fecha de pago"] = scoring_inicial["fecha de pago"].fillna(
    scoring_inicial["dia_pago_estimado"]
)

# Imputar día del último crédito aprobado
scoring_inicial = scoring_inicial.sort_values(
    by=["ID Estudiante", "fecha_creacion_transaccion"],
    ascending=[True, False],
)

df_dias = (
    scoring_inicial[scoring_inicial["fecha de pago"].notnull()]
    .drop_duplicates(subset="ID Estudiante", keep="first")
    [["ID Estudiante", "fecha de pago"]]
)

dic_dia = dict(zip(df_dias["ID Estudiante"], df_dias["fecha de pago"]))

scoring_inicial["fecha de pago"] = scoring_inicial.apply(
    lambda row: dic_dia.get(row["ID Estudiante"], row["fecha de pago"])
    if pd.isnull(row["fecha de pago"])
    else row["fecha de pago"],
    axis=1,
)

# Revisar registros específicos
scoring_inicial.loc[
    scoring_inicial["ID Estudiante"] == 100171144,
    ["ID Estudiante", "Llave", "fecha_creacion_transaccion", "fecha de pago"],
]

scoring_inicial.loc[
    scoring_inicial["fecha de pago"].isnull(),
    ["ID Estudiante", "Llave", "fecha_creacion_transaccion", "fecha de pago"],
]

# Calcular la moda para imputar nulos restantes
moda_dia_pago = scoring_inicial["fecha de pago"].mode()[0]
scoring_inicial["fecha de pago"] = scoring_inicial["fecha de pago"].fillna(
    moda_dia_pago
)
scoring_inicial["fecha de pago"] = scoring_inicial["fecha de pago"].astype(int)

# Copiar columnas de dirección, ciudad y departamento
scoring_inicial["Dirección_Scoring"] = scoring_inicial["Dirección"]
scoring_inicial["Ciudad_Scoring"] = scoring_inicial["Ciudad"]
scoring_inicial["Departamento_Scoring"] = scoring_inicial["Departamento"]

# Validar nulos en valor financiado
scoring_inicial["Validación valor financiado"].isnull().sum()

# Imputar valor financiado promedio por estudiante
scoring_inicial["Validación valor financiado"] = (
    scoring_inicial.groupby("ID Estudiante")["Validación valor financiado"]
    .transform(lambda x: x.fillna(x.mean()))
)

# Crear variable tipo de estudiante
scoring_inicial["Tipo_estudiante"] = scoring_inicial["Área"]

# Imputación neutra de valores faltantes
scoring_inicial["VALOR MAXIMO"] = scoring_inicial["VALOR MAXIMO"].fillna(0)
scoring_inicial["VALOR MEDIO"] = scoring_inicial["VALOR MEDIO"].fillna(0)
scoring_inicial["VALOR BAJO"] = scoring_inicial["VALOR BAJO"].fillna(0)

# Selección de columnas finales
columnas_scoring = [
    "Llave",
    "fecha_creacion_transaccion",
    "Mora",
    "Valor_Matricula",
    "Valor_cuota_inicial",
    "Valor_primera_cuota",
    "fecha de pago",
    "Dirección_Scoring",
    "Ciudad_Scoring",
    "Departamento_Scoring",
    "Validación valor financiado",
    "Detalle estado final",
    "ID Estudiante",
    "Periodo",
    "Tipo_estudiante",
    "Operación",
    "Cate",
    "Subcate",
    "Cohorte",
    "Mes",
    "Cliente",
    "Media Score",
    "AñoB",
    "VALOR MAXIMO",
    "VALOR MEDIO",
    "VALOR BAJO",
    "Plataforma",
]

# Copiar y exportar dataset limpio
df_scoring = scoring_inicial[columnas_scoring].copy()
df_scoring.to_csv("../output/Scoring.csv", index=False, encoding="utf-8-sig")
