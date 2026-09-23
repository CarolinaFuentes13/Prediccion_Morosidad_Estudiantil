"""Métricas de clasificación en numpy puro (sin dependencia de scikit-learn en el tablero)
y acceso a los artefactos precalculados del modelo (``data/model_artifacts.json``).

Las funciones se validan contra scikit-learn en ``tools/build_model_artifacts.py``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from core.config import MODEL_ARTIFACTS_PATH, RISK_ORDER


# ================= Métricas básicas =================

def confusion(y_true, y_pred, labels=RISK_ORDER) -> np.ndarray:
    """Matriz de confusión (filas = real, columnas = predicho) en el orden de ``labels``."""
    yt = pd.Series(y_true).astype(str).to_numpy()
    yp = pd.Series(y_pred).astype(str).to_numpy()
    idx = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for a, b in zip(yt, yp):
        if a in idx and b in idx:
            m[idx[a], idx[b]] += 1
    return m


def per_class_report(y_true, y_pred, labels=RISK_ORDER) -> pd.DataFrame:
    m = confusion(y_true, y_pred, labels)
    tp = np.diag(m).astype(float)
    support = m.sum(axis=1).astype(float)
    predicted = m.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return pd.DataFrame({"clase": labels, "precision": precision, "recall": recall, "f1": f1,
                         "soporte": support.astype(int), "predichos": predicted.astype(int)})


def summary_metrics(y_true, y_pred, labels=RISK_ORDER) -> dict:
    rep = per_class_report(y_true, y_pred, labels)
    m = confusion(y_true, y_pred, labels)
    n = m.sum()
    return {
        "accuracy": float(np.trace(m) / n) if n else float("nan"),
        "precision_macro": float(rep["precision"].mean()),
        "recall_macro": float(rep["recall"].mean()),
        "f1_macro": float(rep["f1"].mean()),
        "f1_weighted": float(np.average(rep["f1"], weights=rep["soporte"])) if rep["soporte"].sum() else float("nan"),
        "n": int(n),
    }


def roc_curve_np(y_bin: np.ndarray, score: np.ndarray):
    """Curva ROC (fpr, tpr, umbrales) para un problema binario."""
    y_bin = np.asarray(y_bin).astype(int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], y_bin[order]
    distinct = np.where(np.diff(s))[0]
    thr_idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[thr_idx]
    fps = (1 + thr_idx) - tps
    P, N = y.sum(), y.size - y.sum()
    tpr = np.r_[0, tps / P] if P else np.r_[0, np.zeros_like(tps, dtype=float)]
    fpr = np.r_[0, fps / N] if N else np.r_[0, np.zeros_like(fps, dtype=float)]
    thr = np.r_[np.inf, s[thr_idx]]
    return fpr, tpr, thr


def auc_trapz(x, y) -> float:
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def roc_auc_binary(y_bin, score) -> float:
    fpr, tpr, _ = roc_curve_np(y_bin, score)
    return auc_trapz(fpr, tpr)


def pr_curve_np(y_bin, score):
    """Precisión-recall (precision, recall, umbrales) y average precision."""
    y_bin = np.asarray(y_bin).astype(int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    s, y = score[order], y_bin[order]
    distinct = np.where(np.diff(s))[0]
    thr_idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[thr_idx]
    fps = (1 + thr_idx) - tps
    P = y.sum()
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / P if P else np.zeros_like(tps, dtype=float)
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision)) if P else float("nan")
    return precision, recall, s[thr_idx], ap


def threshold_sweep(y_bin, score, thresholds=None, total_ref: int | None = None) -> pd.DataFrame:
    """Métricas de la regla 'alerta si score ≥ t' para una malla de umbrales."""
    y_bin = np.asarray(y_bin).astype(int)
    score = np.asarray(score, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.981, 0.01), 2)
    P = y_bin.sum()
    rows = []
    for t in thresholds:
        flag = score >= t
        tp = int((flag & (y_bin == 1)).sum())
        fp = int((flag & (y_bin == 0)).sum())
        fn = int(P - tp)
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / P if P else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) else 0.0
        rows.append({"umbral": float(t), "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
                     "alertas": int(flag.sum()), "pct_alertas": float(flag.mean())})
    return pd.DataFrame(rows)


def calibration_bins(y_bin, score, n_bins: int = 10) -> pd.DataFrame:
    y_bin = np.asarray(y_bin).astype(int)
    score = np.asarray(score, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(score, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({"bin": b, "prob_media": float(score[mask].mean()), "tasa_real": float(y_bin[mask].mean()),
                     "n": int(mask.sum())})
    return pd.DataFrame(rows)


def gains_curve(y_bin, score, points: int = 50) -> pd.DataFrame:
    """Curva de ganancia acumulada: % población gestionada vs % de positivos capturados."""
    y_bin = np.asarray(y_bin).astype(int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = y_bin[order]
    cum = np.cumsum(y) / max(y.sum(), 1)
    n = y.size
    qs = np.linspace(0, 1, points + 1)
    idx = np.clip((qs * n).astype(int) - 1, 0, n - 1)
    cap = np.where(qs == 0, 0, cum[idx])
    return pd.DataFrame({"pct_poblacion": qs, "pct_capturado": cap,
                         "lift": np.where(qs > 0, cap / np.maximum(qs, 1e-9), np.nan)})


def downsample_curve(x, y, n: int = 120):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size <= n:
        return x.tolist(), y.tolist()
    idx = np.unique(np.r_[0, np.linspace(0, x.size - 1, n).astype(int), x.size - 1])
    return x[idx].tolist(), y[idx].tolist()


# ================= Artefactos precalculados =================

@st.cache_data(show_spinner=False)
def load_artifacts() -> dict | None:
    """Lee ``data/model_artifacts.json`` (generado por tools/build_model_artifacts.py)."""
    if not MODEL_ARTIFACTS_PATH.exists():
        return None
    with open(MODEL_ARTIFACTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def artifacts_or_warn() -> dict | None:
    art = load_artifacts()
    if art is None:
        st.warning(
            "No se encontraron los artefactos del modelo (`Dashboard/alerta_temprana/data/model_artifacts.json`). "
            "Genérelos con `python Dashboard/alerta_temprana/tools/build_model_artifacts.py`.",
            icon="⚠️",
        )
    return art
