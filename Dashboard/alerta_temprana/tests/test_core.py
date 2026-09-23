"""Pruebas unitarias de la capa base del SAT (datos, métricas, calidad, filtros, motor).

Ejecutar desde la raíz del repo:  python -m pytest Dashboard/alerta_temprana/tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from core import metrics as M  # noqa: E402
from core.config import BASE_DATA_PATH, MODEL_ARTIFACTS_PATH, MODEL_FEATURES, RISK_ORDER  # noqa: E402
from core.data import assign_route, compute_priority, enrich, programa_cluster, segment_summary  # noqa: E402
from core.engine import missing_required, prepare_inputs  # noqa: E402
from core.filters import apply_filters, default_state  # noqa: E402
from core.quality import failing_rows, quality_score, run_quality  # noqa: E402
from core.theme import fmt_cop, fmt_int, fmt_pct, fmt_period  # noqa: E402


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return enrich(pd.read_csv(BASE_DATA_PATH))


# ---------------- datos ----------------
def test_enrich_shape_and_types(base):
    assert len(base) == 14660
    assert list(base["y_pred"].cat.categories) == RISK_ORDER
    assert base["prioridad"].between(0, 100).all()
    assert set(base["ruta"].unique()) <= {"R1", "R2", "R3", "R4"}
    assert base["nombre"].notna().all()
    # mismo estudiante → mismo nombre anonimizado
    assert base.groupby("id_estudiante")["nombre"].nunique().max() == 1
    assert set(base["cliente_limpio"].unique()) <= {"Estudiante", "No estudiante", "Otro", "Sin dato"}


def test_enrich_tolerates_minimal_upload():
    df = enrich(pd.DataFrame({"llave2": ["a_1", "b_2"], "valor_financiacion": [1e6, 2e6], "y_pred": ["Alto", "bajo"]}))
    assert len(df) == 2
    assert sorted(df["y_pred"].astype(str)) == ["Alto", "Bajo"]
    assert not df["has_truth"].any()


def test_program_clusters_cover_all(base):
    assert (base["programa_cluster"] != "Otros").all()
    assert programa_cluster("DERECHO VIRTUAL") == "Derecho"
    assert programa_cluster("INGENIERIA DE SOFTWARE VIRT") == "Ingeniería y TI"
    assert programa_cluster("PSICOLOGIA VIRTUAL") == "Psicología y Humanidades"


def test_routes_concentrate_observed_risk(base):
    rate = base.assign(a=(base["y_true"].astype(str) == "Alto")).groupby("ruta")["a"].mean()
    assert rate["R1"] > 2 * rate["R4"], rate.to_dict()


def test_priority_weights_change_order(base):
    p1 = compute_priority(base, {"riesgo": 1, "exposicion": 0, "mora": 0})
    p2 = compute_priority(base, {"riesgo": 0, "exposicion": 1, "mora": 0})
    assert not p1.equals(p2)
    assert (assign_route(base) == base["ruta"]).all()


def test_segment_summary(base):
    s = segment_summary(base, "programa_cluster")
    assert s["creditos"].sum() == len(base)
    assert s["pct_alto"].between(0, 1).all()


# ---------------- métricas (contra valores conocidos) ----------------
def test_metrics_small_example():
    yt = ["Alto", "Alto", "Bajo", "Bajo", "Medio", "Bajo"]
    yp = ["Alto", "Bajo", "Bajo", "Bajo", "Medio", "Alto"]
    m = M.confusion(yt, yp)
    assert m.tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 2]]
    rep = M.per_class_report(yt, yp).set_index("clase")
    assert rep.loc["Alto", "precision"] == pytest.approx(0.5)
    assert rep.loc["Bajo", "recall"] == pytest.approx(2 / 3)
    s = M.summary_metrics(yt, yp)
    assert s["accuracy"] == pytest.approx(4 / 6)


def test_auc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert M.roc_auc_binary(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert M.roc_auc_binary(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    sw = M.threshold_sweep(y, np.array([0.1, 0.4, 0.35, 0.8]), thresholds=[0.3])
    assert sw.loc[0, "recall"] == pytest.approx(1.0) and sw.loc[0, "precision"] == pytest.approx(2 / 3)


def test_artifacts_consistent():
    art = json.loads(MODEL_ARTIFACTS_PATH.read_text(encoding="utf-8"))
    rf = next(m for m in art["modelos"] if m["key"] == "rf")
    assert rf["metricas"]["auc_macro"] == pytest.approx(0.8236, abs=1e-3)
    assert rf["seleccionado"]
    target = art["alto"]["rf"]["umbral_objetivo"]
    assert target["recall"] >= 0.75
    assert len(art["temporal"]) >= 8
    assert "importancia" in art and art["importancia"]["items"]


# ---------------- calidad ----------------
def test_quality_rules_run(base):
    res = run_quality(base)
    assert len(res) > 20 and res["evaluados"].gt(0).all()
    sc = quality_score(res)
    assert 0.8 < sc["Global"] <= 1
    da02 = failing_rows(base, "VA-02")
    assert (da02["valor_financiacion"] > 0.85 * da02["vr_neto_matricula"]).all()
    dup = res.set_index("id").loc["UN-01", "incumplen"]
    assert dup > 0  # la base tiene llaves repetidas


# ---------------- filtros ----------------
def test_filters(base):
    s = default_state(base)
    assert len(apply_filters(base, s)) == len(base)
    s["riesgo"] = ["Alto"]
    out = apply_filters(base, s)
    assert (out["y_pred"].astype(str) == "Alto").all() and len(out) == (base["y_pred"] == "Alto").sum()
    s["periodo"] = ("2025-01", "2025-06")
    out = apply_filters(base, s)
    assert out["periodo"].between("2025-01", "2025-06").all()
    s2 = default_state(base)
    s2["q"] = base.iloc[0]["nombre"].split()[0].lower()
    assert len(apply_filters(base, s2)) > 0
    s3 = default_state(base)
    s3["facultad"] = ["NO EXISTE"]
    assert apply_filters(base, s3).empty


# ---------------- motor ----------------
def test_prepare_inputs_schema():
    raw = pd.read_csv(BASE_DATA_PATH).head(5)
    X = prepare_inputs(raw)
    assert list(X.columns) == MODEL_FEATURES
    assert X["fecha_aprobacion"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    assert missing_required(raw) == []
    assert "cuotas" in missing_required(raw.drop(columns=["cuotas"]))


# ---------------- formato ----------------
def test_formatters():
    assert fmt_int(14660) == "14.660"
    assert fmt_pct(0.1234) == "12,3 %"
    assert fmt_cop(2_345_000_000) == "$ 2,3 mil M"
    assert fmt_period("2024-03") == "mar 2024"
