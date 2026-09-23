"""Motor de predicción: API FastAPI → modelo local (paquete .whl) → sin motor.

* **API**: ``[api] url = "https://…"`` en ``st.secrets`` (o variable de entorno
  ``SAT_API_URL``). Usa ``GET /api/v1/health`` y ``POST /api/v1/predict``
  (esquema ``MultipleDataInputs`` del paquete). Devuelve solo la clase.
* **Local**: si el paquete ``model_morosidad`` está instalado
  (``requirements-model.txt``), se usa el pipeline directamente y se obtienen
  además las probabilidades por clase.
* Si no hay motor, el tablero acepta CSV que ya traigan ``y_pred``.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from core.config import MODEL_FEATURES, NUMERIC_FEATURES

BATCH = 400


# ================= Configuración =================

def api_url() -> str | None:
    url = None
    try:
        url = st.secrets.get("api", {}).get("url")
    except Exception:
        url = None
    url = url or os.environ.get("SAT_API_URL")
    return url.rstrip("/") if url else None


@st.cache_data(ttl=60, show_spinner=False)
def api_health(url: str) -> dict:
    import requests

    try:
        t = time.time()
        r = requests.get(f"{url}/api/v1/health", timeout=4)
        r.raise_for_status()
        data = r.json()
        return {"ok": True, "latency_ms": round((time.time() - t) * 1000), **data}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


@st.cache_resource(show_spinner="Cargando el modelo local…")
def _local_model():
    """Carga perezosa del pipeline empaquetado (≈ 550 MB en RAM)."""
    try:
        from model import __version__ as version
        from model.predict import _morosidad_pipe as pipe

        return {"ok": True, "pipe": pipe, "version": version, "classes": [str(c) for c in pipe.named_steps["clf"].classes_]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def local_available() -> bool:
    """Comprueba si el paquete está instalado sin cargar el pickle."""
    import importlib.util

    return importlib.util.find_spec("model") is not None and importlib.util.find_spec("sklearn") is not None


def engine_status() -> dict:
    url = api_url()
    api = {"configured": bool(url), "url": url}
    if url:
        api.update(api_health(url))
    local = {"installed": local_available()}
    preferred = "api" if api.get("ok") else ("local" if local["installed"] else None)
    return {"api": api, "local": local, "preferred": preferred}


# ================= Preparación de insumos =================

def prepare_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Deja el DataFrame con las 42 variables del modelo y tipos compatibles con el esquema."""
    d = df.copy()
    d.columns = [str(c).replace("﻿", "").strip().lower() for c in d.columns]
    for c in MODEL_FEATURES:
        if c not in d.columns:
            d[c] = np.nan
    d = d[MODEL_FEATURES].copy()
    for c in ["fecha_aprobacion", "fecha_nacimiento"]:
        d[c] = pd.to_datetime(d[c], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in NUMERIC_FEATURES:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if d["mora"].dtype != bool:
        d["mora"] = d["mora"].astype(str).str.strip().str.lower().map(
            {"true": True, "1": True, "si": True, "sí": True, "false": False, "0": False, "no": False})
    for c in set(MODEL_FEATURES) - set(NUMERIC_FEATURES) - {"mora", "fecha_aprobacion", "fecha_nacimiento"}:
        d[c] = d[c].where(d[c].isna(), d[c].astype(str).str.strip())
    return d


def missing_required(df: pd.DataFrame) -> list[str]:
    cols = {str(c).replace("﻿", "").strip().lower() for c in df.columns}
    return [c for c in MODEL_FEATURES if c not in cols]


# ================= Predicción =================

def _predict_local(X: pd.DataFrame) -> pd.DataFrame:
    m = _local_model()
    if not m["ok"]:
        raise RuntimeError(m["error"])
    pipe = m["pipe"]
    Xl = X.copy()
    Xl["mora"] = Xl["mora"].map(lambda v: bool(v) if pd.notna(v) else False).astype(bool)
    proba = pipe.predict_proba(Xl)
    classes = m["classes"]
    out = pd.DataFrame(proba, columns=[f"proba_{c.lower()}" for c in classes], index=X.index)
    out["y_pred"] = np.array(classes)[proba.argmax(axis=1)]
    out["proba_pred"] = proba.max(axis=1)
    out["motor"] = f"Modelo local v{m['version']}"
    return out


def _predict_api(X: pd.DataFrame, url: str, progress=None) -> pd.DataFrame:
    import requests

    complete = X.dropna()  # la API descarta filas con nulos: se envían solo filas completas
    preds = pd.Series(index=X.index, dtype=object)
    version = ""
    idx = complete.index.tolist()
    for i in range(0, len(idx), BATCH):
        chunk = complete.loc[idx[i:i + BATCH]]
        payload = {"inputs": chunk.astype(object).where(chunk.notna(), None).to_dict(orient="records")}
        r = requests.post(f"{url}/api/v1/predict", json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")
        body = r.json()
        p = body.get("predictions") or []
        if len(p) != len(chunk):
            raise RuntimeError(f"La API devolvió {len(p)} predicciones para {len(chunk)} filas.")
        preds.loc[chunk.index] = p
        version = body.get("version", version)
        if progress:
            progress(min(1.0, (i + len(chunk)) / max(len(idx), 1)))
    out = pd.DataFrame({"y_pred": preds}, index=X.index)
    out["proba_pred"] = np.nan
    out["motor"] = f"API v{version}" if version else "API"
    return out


def predict(df: pd.DataFrame, engine: str | None = None, progress=None) -> pd.DataFrame:
    """Devuelve y_pred, proba_pred, proba_* (si hay) y motor, alineado al índice de ``df``."""
    X = prepare_inputs(df)
    status = engine_status()
    engine = engine or status["preferred"]
    if engine == "api":
        url = api_url()
        if not url:
            raise RuntimeError("No hay URL de API configurada.")
        return _predict_api(X, url, progress)
    if engine == "local":
        return _predict_local(X)
    raise RuntimeError("No hay motor de predicción disponible (configura la API o instala el modelo local).")
