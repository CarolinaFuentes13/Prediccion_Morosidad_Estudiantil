import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml

# =============================================================================
# 1. CARGA Y LIMPIEZA DE DATOS DE BASE MAFI ACADÉMICA
# =============================================================================

# Ruta base
data_path = Path(__file__).resolve().parents[2] / "data_raw/mafi"

# Buscar archivos que contengan 'MAFI' con extensiones válidas
mafi_files = [
    f for f in data_path.rglob("*MAFI*")
    if f.suffix.lower() in [".xlsx", ".xls", ".csv", ".txt"]
]

# Diccionario donde guardaremos los DataFrames
mafi_dfs = {}

# Cargar dinámicamente todos los archivos
for file in mafi_files:
    print(f"Leyendo archivo: {file.name}")
    ext = file.suffix.lower()

    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file)
        elif ext == ".csv":
            df = pd.read_csv(file)
        elif ext == ".txt":
            df = pd.read_csv(file, sep=None, engine="python")
        else:
            print(f"Formato no soportado: {file}")
            continue

        # Guardar el DataFrame en el diccionario usando el nombre base del archivo
        key_name = file.stem 
        mafi_dfs[key_name] = df
        print(f"Archivo '{file.name}' cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")

    except Exception as e:
        print(f"Error al leer {file.name}: {e}")

# Renombrar "PLAN_ESTUDIO" a "PLAN" en todos los DataFrames del diccionario
for name, df in mafi_dfs.items():
    if "PLAN_ESTUDIO" in df.columns:
        df = df.rename(columns={"PLAN_ESTUDIO": "PLAN"})
        mafi_dfs[name] = df

# Ruta al archivo de configuración
config_path = Path("../inputs/config.yml")

# Leer archivo YAML
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Buscar secciones con "nombre" que contengan "MAFI"
mafi_configs = [
    item for item in config.get("archivos", [])
    if "nombre" in item and "MAFI" in item["nombre"].upper()
]

if not mafi_configs:
    raise ValueError("No se encontraron secciones con nombre que contenga 'MAFI' en config.yml")

# Obtener las columnas asociadas a esas secciones
columnas_mafi = set()
for item in mafi_configs:
    columnas_mafi.update(item.get("columnas", []))

# Filtrar todos los DataFrames del diccionario mafi_dfs
mafi_dfs_filtrados = {}

for nombre, df in mafi_dfs.items():
    cols_comunes = [col for col in df.columns if col in columnas_mafi]
    df_filtrado = df[cols_comunes].copy()
    mafi_dfs_filtrados[nombre] = df_filtrado
    print(f"{nombre}: {len(cols_comunes)} columnas retenidas")


# Concatenar todos los DataFrames del diccionario mafi_dfs
mafi_inicial = pd.concat(mafi_dfs.values(), ignore_index=True)

# Filtrar registros válidos
mafi_inicial = mafi_inicial[~mafi_inicial["IDBANNER"].isnull()].copy()
mafi_inicial = mafi_inicial[~mafi_inicial["PERIODO"].isnull()].copy()
mafi_inicial["PERIODO"] = mafi_inicial["PERIODO"].astype(int)
mafi_inicial = mafi_inicial[~mafi_inicial["FECHA_NACIMIENTO"].isnull()].copy()

# Convertir fechas
mafi_inicial["FECHA_NACIMIENTO"] = pd.to_datetime(
    mafi_inicial["FECHA_NACIMIENTO"], errors="coerce"
)

# Rellenar valores faltantes
mafi_inicial["ESTADO_CIVIL"] = mafi_inicial["ESTADO_CIVIL"].fillna("NO DECLARADO")

# Imputar dirección según periodos previos
direccion_ref = (
    mafi_inicial.groupby("IDBANNER")["DIRECCION"]
    .apply(lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else None)
)

mafi_inicial["DIRECCION"] = mafi_inicial.apply(
    lambda row: direccion_ref[row["IDBANNER"]]
    if pd.isna(row["DIRECCION"])
    else row["DIRECCION"],
    axis=1,
)

# Imputar código de ciudad
mapa_codciu = (
    mafi_inicial[["CIUDAD", "CODCIUDAD"]]
    .dropna()
    .drop_duplicates(subset=["CIUDAD"])
    .set_index("CIUDAD")["CODCIUDAD"]
)

mafi_inicial["CODCIUDAD"] = mafi_inicial.apply(
    lambda row: mapa_codciu[row["CIUDAD"]]
    if pd.isna(row["CODCIUDAD"]) and row["CIUDAD"] in mapa_codciu.index
    else row["CODCIUDAD"],
    axis=1,
)

# Validaciones de nulos
df_nulos_ciudad = mafi_inicial[mafi_inicial["CODCIUDAD"].isnull()][
    ["IDBANNER", "CODCIUDAD", "CIUDAD"]
]

df_sede = mafi_inicial[mafi_inicial["SEDE"].isnull()][
    ["IDBANNER", "SEDE", "PROGRAMA"]
]

# Asignar sede por defecto
mafi_inicial["SEDE"] = mafi_inicial["SEDE"].fillna("VIR")

# Imputar grupo étnico según periodos previos
mafi_inicial = mafi_inicial.sort_values(by=["IDBANNER", "PERIODO"])
mafi_inicial["GRUPO_ETNICO"] = (
    mafi_inicial.groupby("IDBANNER")["GRUPO_ETNICO"].ffill()
)
mafi_inicial["GRUPO_ETNICO"] = mafi_inicial["GRUPO_ETNICO"].fillna("NO PERTENECE")

# Imputar tipo de discapacidad según periodos previos
mafi_inicial = mafi_inicial.sort_values(by=["IDBANNER", "PERIODO"])
mafi_inicial["TIPO_DISCAPACIDAD"] = (
    mafi_inicial.groupby("IDBANNER")["TIPO_DISCAPACIDAD"].ffill()
)
mafi_inicial["TIPO_DISCAPACIDAD"] = mafi_inicial["TIPO_DISCAPACIDAD"].fillna(
    "NO REPORTA"
)


# -----------------------------------------------------------------------------
# Función de limpieza de tipo de discapacidad
# -----------------------------------------------------------------------------
def limpiar_discapacidad(valor):
    """Estandariza valores en la columna 'TIPO_DISCAPACIDAD'."""
    if pd.isna(valor):
        return "NO APLICA"

    v = str(valor).strip().upper()

    no_aplica = [
        "NO", "NO REPORTA", "NINGUNO", "NINGUNO.", "NIGUNO", "NIGUNOS",
        "NO TENGO", "NO PRESENTA", "NO PRECENTA", "NO PRESENTO", "N",
        "NORMAL", "ADULTO SANO", "SANA", "SIN NOVEDAD",
        "NO TENGO NINGÚN ANTECEDENTE", "NO PRESENTO ANTECEDENTES MÉDICOS",
        ".", ",", "0", "A", "A+", "NO APLICA"
    ]
    if any(v.startswith(x) for x in no_aplica):
        return "NO APLICA"

    if any(term in v for term in ["VISUAL", "ESTRAVISMO", "ESTRABIS"]):
        return "DISCAPACIDAD VISUAL"

    cronicas = ["ASMA", "HIPER", "CRONICA", "MIGRA", "RINITIS", "HEPATITIS", "BRONCO"]
    if any(term in v for term in cronicas):
        return "ENFERMEDAD CRÓNICA"

    if "NO INFORM" in v or "NO DECLAR" in v:
        return "NO INFORMADO"

    return "OTRA"


mafi_inicial["TIPO_DISCAPACIDAD"] = mafi_inicial["TIPO_DISCAPACIDAD"].apply(
    limpiar_discapacidad
)

# Nacionalidad
df_nal = mafi_inicial[mafi_inicial["NACIONALIDAD"].isnull()][
    ["IDBANNER", "NACIONALIDAD", "CIUDAD", "CODCIUDAD"]
]

mafi_inicial["NACIONALIDAD"] = mafi_inicial["NACIONALIDAD"].fillna("COLOMBIA")
mafi_inicial["NACIONALIDAD"] = (
    mafi_inicial["NACIONALIDAD"]
    .str.strip()
    .str.upper()
    .replace({"COLOMBIANO": "COLOMBIA"})
)

# -----------------------------------------------------------------------------
# Selección de columnas finales
# -----------------------------------------------------------------------------
columnas_finales = [
    "IDBANNER", "FECHA_NACIMIENTO", "ESTADO_CIVIL",
    "DIRECCION", "CIUDAD", "CODCIUDAD",
    "GENERO", "FACULTAD", "PROGRAMA", "NIVEL",
    "PERIODO", "ESTADO", "TIPOESTUDIANTE", "SEDE", "SELLO",
    "CARGA", "GRUPO_ETNICO", "TIPO_DISCAPACIDAD", "NACIONALIDAD"
]

df_mafi = mafi_inicial[columnas_finales].copy()

# Crear llave de análisis
df_mafi["Llave"] = df_mafi["IDBANNER"].astype(str) + "_" + df_mafi["PERIODO"].astype(str)

# Guardar resultado final
df_mafi.to_csv("../output/MAFI.csv", index=False, encoding="utf-8-sig")
