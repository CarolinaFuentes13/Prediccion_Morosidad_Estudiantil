"""Capa de datos: carga, limpieza, variables derivadas, priorización y dataset activo.

El tablero trabaja siempre sobre un *dataset activo* (la base oficial de
predicciones o un CSV cargado y puntuado por el usuario) guardado en
``st.session_state``. Todas las páginas lo obtienen con :func:`get_active_df`
y lo filtran con :func:`core.filters.apply_filters`.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from core.config import (
    BASE_DATA_PATH, MAX_FINANCIACION_PCT, RISK_ORDER, SCORE_ORDER, SCORE_SHORT,
)

# ================= Utilidades de texto =================

def nrm(s) -> str:
    """Minúsculas, sin tildes ni símbolos (para reglas y búsquedas)."""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9\s]", " ", s).strip()


_NOMBRES_F = ["María", "Laura", "Ana", "Camila", "Valentina", "Carolina", "Paula", "Daniela", "Sara", "Gabriela",
              "Juliana", "Natalia", "Mariana", "Isabella", "Luisa", "Andrea", "Catalina", "Diana", "Manuela",
              "Sofía", "Lucía", "Alejandra", "Adriana", "Paola", "Viviana", "Tatiana", "Lorena", "Ángela",
              "Marcela", "Ximena"]
_NOMBRES_M = ["Juan", "Carlos", "Andrés", "Diego", "Luis", "Mateo", "Jorge", "Felipe", "Daniel", "Santiago",
              "Sebastián", "Nicolás", "Camilo", "Alejandro", "David", "Julián", "Esteban", "Miguel", "Samuel",
              "Tomás", "Óscar", "Fernando", "Ricardo", "Mauricio", "Iván", "Hernán", "Álvaro", "Cristian",
              "Fabián", "Germán"]
_APELLIDOS = ["García", "Rodríguez", "López", "Martínez", "Hernández", "Gómez", "Díaz", "Ramírez", "Torres",
              "Vargas", "Rojas", "Moreno", "Jiménez", "Castro", "Ortiz", "Rubio", "Suárez", "Mejía", "Cárdenas",
              "Pardo", "Restrepo", "Castaño", "Salazar", "Ospina", "Quintero", "Mendoza", "Rincón", "Pineda",
              "Aguilar", "Beltrán"]


def nombre_fake(seed, genero=None) -> str:
    """Nombre anonimizado y determinístico (mismo estudiante → mismo nombre)."""
    h = int(hashlib.sha256(str(seed).encode()).hexdigest(), 16)
    base = _NOMBRES_F if str(genero).strip().lower() in {"f", "femenino"} else _NOMBRES_M
    n = len(_APELLIDOS)
    return f"{base[h % len(base)]} {_APELLIDOS[(h // 31) % n]} {_APELLIDOS[(h // 997) % n]}"


# ================= Segmentos de programa =================
_CLUSTER_RULES = [
    ("Derecho", r"\bderech"),
    ("Salud", r"salud|fisio|fonoau|enfermer|audiolog|disfagia|pulmonar|seg y sal|\bsst\b|telepsic|atencion primaria"),
    ("Psicología y Humanidades", r"psicol|neurops|neup|trabajo social|comunicacion|humanidades|autist"),
    ("Educación", r"educ|\blic\b|licenciatura|pedag|infantil|\binf\b|adoles|ingles|lenguas|artes visuales|inclu|ambien digt|did flex"),
    ("Negocios y Administración", r"admin|negoc|\bneg\b|finan|conta|mercad|\bmkt\b|mark|geren|\bgere\b|gcia|econom|niif|talento|power bi|"
                                  r"proyect|\bmba\b|\bibn\b|lider|calidad|audit|comport|sostenible|rrhh|banc|fintec"),
    ("Ingeniería y TI", r"ingenier|\bing\b|sistemas|software|datos|data|web|boot|program|ciberseg|big data|react|"
                        r"lean|logis|\blog\b|analiti|python|ux"),
]


def programa_cluster(programa) -> str:
    t = nrm(programa)
    for name, pattern in _CLUSTER_RULES:
        if re.search(pattern, t):
            return name
    return "Otros"


def _clean_cliente(x) -> str:
    t = nrm(x)
    if t == "estudiante":
        return "Estudiante"
    if t in {"no estudiante", "noestudiante"}:
        return "No estudiante"
    return "Otro" if t else "Sin dato"


def _clean_depto(x) -> str:
    t = str(x).strip()
    if nrm(t).startswith("bogota"):
        return "Bogotá D.C."
    return t if t and t.lower() != "nan" else "Sin dato"


def _clean_operacion(x) -> str:
    t = str(x).strip()
    if not t or t.lower() == "nan":
        return "Sin dato"
    return t[:1].upper() + t[1:].lower() if t.isupper() else t


def _parse_mora(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "si", "sí", "yes", "y", "mora", "en mora"}).astype(int)


# ================= Priorización =================
DEFAULT_WEIGHTS = {"riesgo": 0.60, "exposicion": 0.25, "mora": 0.15}


def risk_intensity(y_pred: pd.Series, proba: pd.Series) -> pd.Series:
    """Intensidad de riesgo en [0, 1] a partir de la clase y la confianza del modelo.

    Alto: 0,55–1,00 · Medio: 0,35–0,60 · Bajo: 0,00–0,35 (un 'Bajo' poco seguro sube).
    """
    p = proba.fillna(0.5).clip(0, 1)
    yp = y_pred.astype(str)
    r = np.where(yp == "Alto", 0.55 + 0.45 * p,
        np.where(yp == "Medio", 0.35 + 0.25 * p, 0.35 * (1 - p)))
    return pd.Series(r, index=y_pred.index)


def compute_priority(df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Índice de prioridad 0–100 = combinación ponderada de riesgo, exposición y mora histórica."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = max(sum(w.values()), 1e-9)
    r = risk_intensity(df["y_pred"], df["proba_pred"])
    e = df["valor_financiacion"].rank(pct=True).fillna(0.5)
    m = df["mora_flag"].astype(float)
    return (100 * (w["riesgo"] * r + w["exposicion"] * e + w["mora"] * m) / total).round(1)


def assign_route(df: pd.DataFrame) -> pd.Series:
    """Rutas de gestión preventiva (R1–R4) según riesgo predicho, confianza y mora."""
    yp = df["y_pred"].astype(str)
    p = df["proba_pred"].fillna(0.5)
    mora = df["mora_flag"] == 1
    route = np.select(
        [
            (yp == "Alto") & ((p >= 0.5) | mora),
            (yp == "Alto") | (yp == "Medio"),
            (yp == "Bajo") & ((p < 0.65) | mora),
        ],
        ["R1", "R2", "R3"],
        default="R4",
    )
    return pd.Series(route, index=df.index)


# ================= Enriquecimiento =================

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos y crea las variables que usan las páginas.

    Funciona tanto con la base oficial como con CSV cargados (columnas faltantes
    se crean vacías para que el tablero nunca falle por esquema).
    """
    df = df.copy()
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower)

    # --- fechas ---
    df["fecha_aprobacion"] = pd.to_datetime(df.get("fecha_aprobacion"), errors="coerce")
    df["fecha_nacimiento"] = pd.to_datetime(df.get("fecha_nacimiento"), errors="coerce")
    df["anio"] = df["fecha_aprobacion"].dt.year.astype("Int64")
    df["periodo"] = df["fecha_aprobacion"].dt.to_period("M").astype(str).replace("NaT", np.nan)
    df["trimestre"] = df["fecha_aprobacion"].dt.to_period("Q").astype(str).replace("NaT", np.nan)
    sem = np.where(df["fecha_aprobacion"].dt.month <= 6, "S1", "S2")
    df["semestre"] = np.where(df["anio"].notna(), df["anio"].astype(str) + "-" + sem, np.nan)
    edad = (df["fecha_aprobacion"] - df["fecha_nacimiento"]).dt.days / 365.25
    df["edad"] = edad.where((edad > 14) & (edad < 95)).round(0)
    df["rango_edad"] = pd.cut(df["edad"], bins=[0, 24, 29, 34, 44, 120],
                              labels=["≤ 24", "25–29", "30–34", "35–44", "45 +"]).astype(str).replace("nan", "Sin dato")

    # --- numéricas ---
    for c in ["valor_financiacion", "vr_neto_matricula", "cuotas", "antiguedad_meses", "valor_cuota_inicial",
              "valor_primera_cuota", "fecha_de_pago", "validacion_valor_financiado", "valor_maximo",
              "valor_medio", "valor_bajo", "latitud", "longitud", "proba_pred", "id_estudiante"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["ratio_financiacion"] = (df["valor_financiacion"] / df["vr_neto_matricula"]).replace([np.inf, -np.inf], np.nan)
    df["flag_da02"] = ((df["valor_financiacion"] <= 0) | (df["ratio_financiacion"] > MAX_FINANCIACION_PCT)).astype(int)

    # --- riesgo ---
    for c in ["y_pred", "y_true"]:
        if c in df.columns:
            s = df[c].astype(str).str.strip().str.capitalize()
            s = s.where(s.isin(RISK_ORDER))
            df[c] = pd.Categorical(s, categories=RISK_ORDER, ordered=True)
        else:
            df[c] = pd.Categorical([np.nan] * len(df), categories=RISK_ORDER, ordered=True)
    df["has_truth"] = df["y_true"].notna()
    df["acierto"] = np.where(df["has_truth"] & df["y_pred"].notna(),
                             (df["y_true"].astype(str) == df["y_pred"].astype(str)).astype(float), np.nan)
    for c in ["proba_alto", "proba_medio", "proba_bajo"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # --- categóricas limpias ---
    df["mora_flag"] = _parse_mora(df["mora"]) if "mora" in df.columns else 0
    df["mora_txt"] = np.where(df["mora_flag"] == 1, "Con mora", "Sin mora")
    df["cliente_limpio"] = df["cliente"].map(_clean_cliente) if "cliente" in df.columns else "Sin dato"
    df["departamento_limpio"] = df["departamento"].map(_clean_depto) if "departamento" in df.columns else "Sin dato"
    df["operacion_limpia"] = df["operacion"].map(_clean_operacion) if "operacion" in df.columns else "Sin dato"
    for c in ["programa", "facultad", "nivel", "sede", "tipo_interes", "tipo_estudiante", "genero", "estado_civil",
              "cohorte", "media_score", "plataforma", "nombre_linea", "nombre_fondo", "ciudad_norm", "tipoestudiante"]:
        if c not in df.columns:
            df[c] = "Sin dato"
        df[c] = df[c].fillna("Sin dato").astype(str).str.strip()
    df["programa_cluster"] = df["programa"].map(programa_cluster)
    df["score_corto"] = df["media_score"].map(SCORE_SHORT).fillna("Sin dato")
    df["score_rank"] = df["media_score"].map({k: i for i, k in enumerate(SCORE_ORDER)})
    df["genero_txt"] = df["genero"].map({"F": "Femenino", "M": "Masculino"}).fillna("Sin dato")

    # --- identidad anonimizada ---
    if "llave2" not in df.columns:
        df["llave2"] = [f"SIN-LLAVE-{i}" for i in range(len(df))]
    df["llave2"] = df["llave2"].astype(str)
    if df["id_estudiante"].isna().all():
        df["id_estudiante"] = df["llave2"].str.split("_").str[0]
    df["id_estudiante"] = df["id_estudiante"].astype("string").str.replace(r"\.0$", "", regex=True)
    if "nombre" not in df.columns:
        pairs = df[["id_estudiante", "genero"]].astype(str).drop_duplicates()
        names = {(i, g): nombre_fake(f"{i}-{g}", g) for i, g in pairs.itertuples(index=False)}
        df["nombre"] = [names[(i, g)] for i, g in df[["id_estudiante", "genero"]].astype(str).itertuples(index=False)]
    df["n_creditos_estudiante"] = df.groupby("id_estudiante")["llave2"].transform("size")

    # --- priorización ---
    df["intensidad_riesgo"] = risk_intensity(df["y_pred"], df["proba_pred"])
    df["prioridad"] = compute_priority(df)
    df["ruta"] = assign_route(df)
    # Exposición esperada: monto × intensidad de riesgo (proxy de pérdida potencial para priorizar).
    df["exposicion_riesgo"] = df["valor_financiacion"] * df["intensidad_riesgo"]

    return df.sort_values("fecha_aprobacion", ascending=False, na_position="last").reset_index(drop=True)


@st.cache_data(show_spinner="Cargando la base de predicciones…")
def load_base() -> pd.DataFrame:
    raw = pd.read_csv(BASE_DATA_PATH)
    df = enrich(raw)
    df["origen"] = "Base oficial"
    return df


# ================= Dataset activo =================
_KEY = "sat_dataset"


def get_active_meta() -> dict:
    meta = st.session_state.get(_KEY)
    if not meta:
        return {"name": "Base oficial de predicciones", "source": "base", "rows": None, "engine": "Modelo RF (test)"}
    return {k: v for k, v in meta.items() if k != "df"}


def get_active_df() -> pd.DataFrame:
    meta = st.session_state.get(_KEY)
    if meta and meta.get("df") is not None:
        return meta["df"]
    return load_base()


def set_active_df(df: pd.DataFrame, name: str, source: str = "upload", engine: str = "") -> None:
    st.session_state[_KEY] = {"df": df, "name": name, "source": source, "rows": len(df), "engine": engine}


def reset_active_df() -> None:
    st.session_state.pop(_KEY, None)


# ================= Agregaciones reutilizables =================

def risk_share(df: pd.DataFrame, by: str | list[str], col: str = "y_pred") -> pd.DataFrame:
    """Conteo y % por clase de riesgo dentro de cada grupo (formato largo)."""
    by = [by] if isinstance(by, str) else list(by)
    g = df.groupby(by + [col], observed=False).size().rename("n").reset_index()
    g["total"] = g.groupby(by)["n"].transform("sum")
    g["pct"] = np.where(g["total"] > 0, g["n"] / g["total"], np.nan)
    return g


def segment_summary(df: pd.DataFrame, by: str | list[str], min_n: int = 1) -> pd.DataFrame:
    """Resumen por segmento: créditos, % Alto predicho, % Alto observado, mora y exposición."""
    by = [by] if isinstance(by, str) else list(by)
    tmp = df.assign(
        _alto=(df["y_pred"].astype(str) == "Alto").astype(int),
        _alto_real=np.where(df["has_truth"], (df["y_true"].astype(str) == "Alto").astype(float), np.nan),
    )
    out = tmp.groupby(by, observed=True).agg(
        creditos=("llave2", "size"),
        estudiantes=("id_estudiante", "nunique"),
        pct_alto=("_alto", "mean"),
        pct_alto_real=("_alto_real", "mean"),
        pct_mora=("mora_flag", "mean"),
        exposicion=("valor_financiacion", "sum"),
        exposicion_alto=("valor_financiacion", lambda s: s[tmp.loc[s.index, "_alto"] == 1].sum()),
        ticket=("valor_financiacion", "mean"),
        prioridad=("prioridad", "mean"),
    ).reset_index()
    return out[out["creditos"] >= min_n].sort_values("creditos", ascending=False)


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Varias hojas → .xlsx en memoria (requiere openpyxl)."""
    import io

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d2 = d.copy()
            for c in d2.columns:
                if isinstance(d2[c].dtype, pd.CategoricalDtype):
                    d2[c] = d2[c].astype(str)
                if pd.api.types.is_datetime64_any_dtype(d2[c]):
                    d2[c] = d2[c].dt.strftime("%Y-%m-%d")
            d2.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
