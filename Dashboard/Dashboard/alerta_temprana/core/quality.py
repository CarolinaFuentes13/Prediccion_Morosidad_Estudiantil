"""Controles automáticos de calidad de datos (ME-02) sobre la base activa o un CSV cargado.

Replica, sobre el dataset analítico final, las dimensiones priorizadas con los
stakeholders (completitud, exactitud, validez) y agrega consistencia y unicidad.
Cada regla devuelve evaluados / cumplen / incumplen y una máscara de filas que
fallan para poder listarlas y exportarlas (p. ej. DA-02).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from core.config import MAX_FINANCIACION_PCT

VALID_DETALLE = {"desembolsado", "aprobada en preanalisis y analisis sin desembolsar", "aprobado completo",
                 "aprobado incompleto"}
VALID_INTERES = {"Cap. Fijo-Int. vencido", "Anualidad Vencida"}
CRITICAL_FIELDS = ["llave2", "fecha_aprobacion", "valor_financiacion", "vr_neto_matricula", "cuotas", "tipo_interes",
                   "programa", "facultad", "genero", "fecha_nacimiento", "media_score", "tipo_estudiante"]


@dataclass
class Rule:
    id: str
    dimension: str
    campo: str
    descripcion: str
    fail: Callable[[pd.DataFrame], pd.Series]   # True = incumple
    applies: Callable[[pd.DataFrame], pd.Series] | None = None  # filas evaluadas
    severidad: str = "Media"


def _col(df, c):
    return df[c] if c in df.columns else pd.Series(np.nan, index=df.index)


def _num(df, c):
    return pd.to_numeric(_col(df, c), errors="coerce")


def _date(df, c):
    s = _col(df, c)
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")


def _txt(df, c):
    return _col(df, c).astype(str).str.strip()


def _rules() -> list[Rule]:
    rules: list[Rule] = []
    for i, c in enumerate(CRITICAL_FIELDS, 1):
        rules.append(Rule(f"CO-{i:02d}", "Completitud", c, f"«{c}» diligenciado",
                          fail=lambda d, c=c: _col(d, c).isna() | _txt(d, c).isin(["", "nan", "None", "NaT", "Sin dato"]),
                          severidad="Alta" if c in {"llave2", "valor_financiacion", "fecha_aprobacion"} else "Media"))
    rules += [
        Rule("VA-01", "Validez", "valor_financiacion", "Valor financiado > 0",
             fail=lambda d: ~(_num(d, "valor_financiacion") > 0), severidad="Alta"),
        Rule("VA-02", "Validez", "valor_financiacion",
             f"Financiación ≤ {MAX_FINANCIACION_PCT:.0%} de Vr_neto_matricula (DA-02)",
             fail=lambda d: _num(d, "valor_financiacion") > MAX_FINANCIACION_PCT * _num(d, "vr_neto_matricula"),
             applies=lambda d: _num(d, "valor_financiacion").notna() & _num(d, "vr_neto_matricula").notna(),
             severidad="Alta"),
        Rule("VA-03", "Validez", "cuotas", "Número de cuotas entre 1 y 24",
             fail=lambda d: ~_num(d, "cuotas").between(1, 24)),
        Rule("VA-04", "Validez", "fecha_de_pago", "Día de pago entre 1 y 30",
             fail=lambda d: ~_num(d, "fecha_de_pago").between(1, 30)),
        Rule("VA-05", "Validez", "valor_cuota_inicial", "Cuota inicial numérica > 0",
             fail=lambda d: ~(_num(d, "valor_cuota_inicial") > 0)),
        Rule("VA-06", "Validez", "valor_primera_cuota", "Primera cuota numérica > 0",
             fail=lambda d: ~(_num(d, "valor_primera_cuota") > 0)),
        Rule("VA-07", "Validez", "tipo_interes", "Tipo de interés en el catálogo",
             fail=lambda d: ~_txt(d, "tipo_interes").isin(VALID_INTERES)),
        Rule("VA-08", "Validez", "detalle_estado_final", "Estado final en el catálogo de aprobación/desembolso",
             fail=lambda d: ~_txt(d, "detalle_estado_final").str.lower().isin(VALID_DETALLE)),
        Rule("VA-09", "Validez", "genero", "Género ∈ {F, M}", fail=lambda d: ~_txt(d, "genero").isin(["F", "M"])),
        Rule("VA-10", "Validez", "fecha_nacimiento", "Edad al aprobar entre 15 y 90 años",
             fail=lambda d: ~(((_date(d, "fecha_aprobacion") - _date(d, "fecha_nacimiento")).dt.days / 365.25)
                              .between(15, 90))),
        Rule("VA-11", "Validez", "fecha_aprobacion", "Fecha de aprobación válida y no futura",
             fail=lambda d: _date(d, "fecha_aprobacion").isna() | (_date(d, "fecha_aprobacion") > pd.Timestamp.today()),
             severidad="Alta"),
        Rule("EX-01", "Exactitud", "anob", "Año del scoring (AñoB) = año de aprobación",
             fail=lambda d: _num(d, "anob") != _date(d, "fecha_aprobacion").dt.year,
             applies=lambda d: _num(d, "anob").notna() & _date(d, "fecha_aprobacion").notna()),
        Rule("EX-02", "Exactitud", "valor_primera_cuota", "Primera cuota ≤ valor financiado",
             fail=lambda d: _num(d, "valor_primera_cuota") > _num(d, "valor_financiacion"),
             applies=lambda d: _num(d, "valor_primera_cuota").notna() & _num(d, "valor_financiacion").notna()),
        Rule("CS-01", "Consistencia", "cliente", "Tipo de cliente con escritura estándar",
             fail=lambda d: ~_txt(d, "cliente").isin(["Estudiante", "No estudiante"]), severidad="Baja"),
        Rule("CS-02", "Consistencia", "operacion", "Operación sin variantes de mayúsculas (p. ej. SUR / Sur)",
             fail=lambda d: _txt(d, "operacion").str.isupper() & (_txt(d, "operacion").str.len() > 3), severidad="Baja"),
        Rule("CS-03", "Consistencia", "departamento", "Departamento homologado (DIVIPOLA)",
             fail=lambda d: _txt(d, "departamento").str.contains(",", regex=False), severidad="Baja"),
        Rule("UN-01", "Unicidad", "llave2", "Llave del crédito única",
             fail=lambda d: _col(d, "llave2").duplicated(keep=False) & _col(d, "llave2").notna(), severidad="Alta"),
    ]
    return rules


RULES = _rules()


def run_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta todas las reglas → tabla de resultados."""
    rows = []
    for r in RULES:
        try:
            applies = r.applies(df) if r.applies else pd.Series(True, index=df.index)
            fail = r.fail(df).fillna(True) & applies
            n = int(applies.sum())
            k = int(fail.sum())
        except Exception:
            n, k = 0, 0
        rows.append({"id": r.id, "dimension": r.dimension, "campo": r.campo, "regla": r.descripcion,
                     "severidad": r.severidad, "evaluados": n, "incumplen": k, "cumplen": n - k,
                     "cumplimiento": (n - k) / n if n else np.nan})
    return pd.DataFrame(rows)


def failing_rows(df: pd.DataFrame, rule_id: str) -> pd.DataFrame:
    r = next(x for x in RULES if x.id == rule_id)
    applies = r.applies(df) if r.applies else pd.Series(True, index=df.index)
    return df[(r.fail(df).fillna(True) & applies)]


def quality_score(res: pd.DataFrame) -> dict:
    """Cumplimiento global y por dimensión (promedio ponderado por evaluados)."""
    out = {}
    for dim, g in res.groupby("dimension"):
        ev = g["evaluados"].sum()
        out[dim] = float(g["cumplen"].sum() / ev) if ev else np.nan
    ev = res["evaluados"].sum()
    out["Global"] = float(res["cumplen"].sum() / ev) if ev else np.nan
    return out
