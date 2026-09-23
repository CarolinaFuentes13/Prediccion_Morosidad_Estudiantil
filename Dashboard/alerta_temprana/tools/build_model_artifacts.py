"""Precalcula los artefactos de evaluación del modelo para el tablero.

Lee las predicciones de prueba (80/20 estratificado) de los cinco notebooks de
``Predictive_models`` y genera ``Dashboard/alerta_temprana/data/model_artifacts.json``
con: métricas globales y por clase (MO-02 / EV-02), matrices de confusión, curvas
ROC y PR one-vs-rest, barrido de umbrales para la clase Alto (DE-03), curva de
ganancia, calibración y desempeño por semestre de aprobación (ME-01).

Si existe ``data/feature_importance.json`` (ver ``extract_feature_importance.py``)
se incorpora como importancia de variables (MO-03).

Uso:
    python Dashboard/alerta_temprana/tools/build_model_artifacts.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from core.config import (  # noqa: E402
    APP_DATA_DIR, AUC_RETRAIN_THRESHOLD, MODEL_ARTIFACTS_PATH, MODEL_LABELS, RECALL_ALTO_TARGET, RISK_ORDER,
    SELECTED_MODEL, TEST_PRED_FILES,
)
from core.metrics import (  # noqa: E402
    calibration_bins, confusion, downsample_curve, gains_curve, per_class_report, pr_curve_np, roc_auc_binary,
    roc_curve_np, summary_metrics, threshold_sweep,
)

XGB_LABELS = {"0": "Alto", "1": "Bajo", "2": "Medio"}  # LabelEncoder (orden alfabético)


def load_test(key: str) -> pd.DataFrame:
    df = pd.read_csv(TEST_PRED_FILES[key])
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    if "proba_0" in df.columns:
        df = df.rename(columns={f"proba_{k}": f"proba_{v}" for k, v in XGB_LABELS.items()})
        df["y_pred"] = df["y_pred"].astype(str).map(XGB_LABELS)
    df["y_true"] = df["y_true"].astype(str).str.capitalize()
    df["y_pred"] = df["y_pred"].astype(str).str.capitalize()
    df["fecha"] = pd.to_datetime(df.get("Fecha_aprobacion"), errors="coerce", dayfirst=True)
    return df


def auc_macro_ovr(df: pd.DataFrame) -> tuple[float, dict]:
    per = {}
    for c in RISK_ORDER:
        y = (df["y_true"] == c).astype(int).to_numpy()
        if 0 < y.sum() < len(y):
            per[c] = roc_auc_binary(y, df[f"proba_{c}"].to_numpy())
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def model_block(key: str, df: pd.DataFrame) -> dict:
    s = summary_metrics(df["y_true"], df["y_pred"])
    rep = per_class_report(df["y_true"], df["y_pred"])
    auc_macro, auc_per = auc_macro_ovr(df)
    roc, pr = {}, {}
    for c in RISK_ORDER:
        y = (df["y_true"] == c).astype(int).to_numpy()
        sc = df[f"proba_{c}"].to_numpy()
        fpr, tpr, _ = roc_curve_np(y, sc)
        fx, ty = downsample_curve(fpr, tpr, 150)
        roc[c] = {"fpr": fx, "tpr": ty, "auc": auc_per.get(c)}
        prec, rec, _, ap = pr_curve_np(y, sc)
        rx, py = downsample_curve(rec[::-1], prec[::-1], 150)
        pr[c] = {"recall": rx, "precision": py, "ap": ap, "base_rate": float(y.mean())}
    rep_d = rep.set_index("clase")
    return {
        "key": key,
        "nombre": MODEL_LABELS[key],
        "seleccionado": key == SELECTED_MODEL,
        "n_test": int(len(df)),
        "metricas": {
            **s,
            "auc_macro": auc_macro,
            "recall_alto": float(rep_d.loc["Alto", "recall"]),
            "precision_alto": float(rep_d.loc["Alto", "precision"]),
            "f1_alto": float(rep_d.loc["Alto", "f1"]),
            "auc_alto": auc_per.get("Alto"),
        },
        "por_clase": [
            {**{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in row.items()}, "auc": auc_per.get(row["clase"])}
            for row in rep.to_dict(orient="records")
        ],
        "confusion": confusion(df["y_true"], df["y_pred"]).tolist(),
        "roc": roc,
        "pr": pr,
    }


def alto_block(df: pd.DataFrame) -> dict:
    y = (df["y_true"] == "Alto").astype(int).to_numpy()
    sc = df["proba_Alto"].to_numpy()
    sweep = threshold_sweep(y, sc)
    ok = sweep[sweep["recall"] >= RECALL_ALTO_TARGET]
    best_target = ok.sort_values(["precision", "umbral"], ascending=[False, False]).iloc[0].to_dict() if len(ok) else None
    best_f1 = sweep.sort_values("f1", ascending=False).iloc[0].to_dict()
    # Regla actual del modelo: la clase con mayor probabilidad (argmax).
    current = {
        "recall": float(((df["y_pred"] == "Alto") & (df["y_true"] == "Alto")).sum() / max(y.sum(), 1)),
        "precision": float(((df["y_pred"] == "Alto") & (df["y_true"] == "Alto")).sum() / max((df["y_pred"] == "Alto").sum(), 1)),
        "pct_alertas": float((df["y_pred"] == "Alto").mean()),
    }
    return {
        "barrido": sweep.replace({np.nan: None}).to_dict(orient="list"),
        "umbral_objetivo": best_target,
        "umbral_f1": best_f1,
        "regla_argmax": current,
        "ganancia": gains_curve(y, sc).replace({np.nan: None}).to_dict(orient="list"),
        "calibracion": calibration_bins(y, sc).to_dict(orient="list"),
        "tasa_base": float(y.mean()),
        "n": int(len(y)),
        "histograma": {
            "alto": np.histogram(sc[y == 1], bins=40, range=(0, 1))[0].tolist(),
            "no_alto": np.histogram(sc[y == 0], bins=40, range=(0, 1))[0].tolist(),
            "bordes": np.linspace(0, 1, 41).round(3).tolist(),
        },
    }


def temporal_block(df: pd.DataFrame) -> list[dict]:
    d = df.dropna(subset=["fecha"]).copy()
    d["semestre"] = d["fecha"].dt.year.astype(str) + np.where(d["fecha"].dt.month <= 6, "-S1", "-S2")
    rows = []
    for sem, g in d.groupby("semestre"):
        if len(g) < 150:
            continue
        auc_m, per = auc_macro_ovr(g)
        rep = per_class_report(g["y_true"], g["y_pred"]).set_index("clase")
        rows.append({
            "semestre": sem, "n": int(len(g)), "auc_macro": auc_m, "auc_alto": per.get("Alto"),
            "recall_alto": float(rep.loc["Alto", "recall"]), "precision_alto": float(rep.loc["Alto", "precision"]),
            "accuracy": float((g["y_true"] == g["y_pred"]).mean()),
            "pct_alto_real": float((g["y_true"] == "Alto").mean()),
            "pct_alto_pred": float((g["y_pred"] == "Alto").mean()),
            "alerta_reentrenamiento": bool(auc_m < AUC_RETRAIN_THRESHOLD) if not np.isnan(auc_m) else None,
        })
    return rows


def validate_against_sklearn(df: pd.DataFrame) -> dict:
    """Comprueba que las métricas numpy coinciden con scikit-learn (si está instalado)."""
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    except Exception:
        return {"sklearn": "no disponible"}
    ours = summary_metrics(df["y_true"], df["y_pred"])
    auc_ours, _ = auc_macro_ovr(df)
    ref = {
        "f1_macro": f1_score(df["y_true"], df["y_pred"], average="macro", labels=RISK_ORDER),
        "precision_macro": precision_score(df["y_true"], df["y_pred"], average="macro", labels=RISK_ORDER, zero_division=0),
        "recall_macro": recall_score(df["y_true"], df["y_pred"], average="macro", labels=RISK_ORDER),
        "auc_macro": roc_auc_score(df["y_true"], df[["proba_Alto", "proba_Bajo", "proba_Medio"]].to_numpy(),
                                   multi_class="ovr", average="macro", labels=["Alto", "Bajo", "Medio"]),
    }
    diffs = {k: abs(ours[k] - v) if k != "auc_macro" else abs(auc_ours - v) for k, v in ref.items()}
    assert max(diffs.values()) < 1e-6, f"Métricas numpy difieren de sklearn: {diffs}"
    return {k: round(float(v), 6) for k, v in ref.items()}


def main() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    models, validation, alto_all = [], {}, {}
    rf_df = None
    for key in TEST_PRED_FILES:
        df = load_test(key)
        models.append(model_block(key, df))
        alto_all[key] = alto_block(df)
        validation[key] = validate_against_sklearn(df)
        if key == SELECTED_MODEL:
            rf_df = df
        m = models[-1]["metricas"]
        print(f"{MODEL_LABELS[key]:<18} AUC={m['auc_macro']:.4f} Acc={m['accuracy']:.4f} "
              f"P={m['precision_macro']:.4f} R={m['recall_macro']:.4f} F1={m['f1_macro']:.4f} RecAlto={m['recall_alto']:.4f}")

    artifacts = {
        "generado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "fuente": "Predicciones de prueba (partición estratificada 80/20) de Predictive_models/Modelo 1–5",
        "clases": RISK_ORDER,
        "modelo_seleccionado": SELECTED_MODEL,
        "modelos": models,
        "alto": alto_all,
        "temporal": temporal_block(rf_df),
        "validacion_sklearn": validation,
    }
    fi_path = APP_DATA_DIR / "feature_importance.json"
    if fi_path.exists():
        artifacts["importancia"] = json.loads(fi_path.read_text(encoding="utf-8"))
    MODEL_ARTIFACTS_PATH.write_text(json.dumps(artifacts, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"OK → {MODEL_ARTIFACTS_PATH} ({MODEL_ARTIFACTS_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
