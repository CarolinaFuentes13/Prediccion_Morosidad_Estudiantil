"""Filtros globales (DB-01).

* Se dibujan en la barra lateral desde ``app.py`` para las páginas que analizan la cartera.
* Persisten al navegar entre páginas (estado propio en ``st.session_state`` +
  callbacks, porque Streamlit borra el estado de widgets que no se dibujan).
* Son encadenados: las opciones de *Programa* dependen de Facultad/Segmento.
* Se pueden fijar en la URL para compartir una vista (``?riesgo=Alto&facultad=…``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.config import RISK_ORDER
from core.data import nrm
from core.theme import fmt_int, fmt_period

PKEY = "_sat_flt"
QP_FLAG = "_sat_qp_loaded"

# nombre → (etiqueta, columna)
MULTI = {
    "riesgo": ("Riesgo predicho", "y_pred"),
    "facultad": ("Facultad", "facultad"),
    "cluster": ("Segmento de programa", "programa_cluster"),
    "programa": ("Programa", "programa"),
    "nivel": ("Nivel académico", "nivel"),
    "sede": ("Sede", "sede"),
    "tipo_interes": ("Tipo de interés", "tipo_interes"),
    "cliente": ("Tipo de cliente", "cliente_limpio"),
    "tipo_estudiante": ("Tipo de estudiante", "tipo_estudiante"),
    "cohorte": ("Cohorte", "cohorte"),
    "departamento": ("Departamento", "departamento_limpio"),
    "ruta": ("Ruta de gestión", "ruta"),
}
MORA_OPTS = ["Todos", "Con mora", "Sin mora"]


def _periods(df: pd.DataFrame) -> list[str]:
    return sorted(p for p in df["periodo"].dropna().unique())


def default_state(df: pd.DataFrame) -> dict:
    per = _periods(df)
    st_ = {k: [] for k in MULTI}
    st_.update({"periodo": (per[0], per[-1]) if per else None, "mora": "Todos", "q": ""})
    return st_


def get_state(df: pd.DataFrame) -> dict:
    if PKEY not in st.session_state:
        st.session_state[PKEY] = default_state(df)
        _load_query_params(df)
    return st.session_state[PKEY]


def _load_query_params(df: pd.DataFrame) -> None:
    if st.session_state.get(QP_FLAG):
        return
    st.session_state[QP_FLAG] = True
    qp = st.query_params
    s = st.session_state[PKEY]
    for k in MULTI:
        if k in qp and qp[k]:
            s[k] = [v for v in qp[k].split("|") if v]
    if qp.get("mora") in MORA_OPTS:
        s["mora"] = qp["mora"]
    if qp.get("q"):
        s["q"] = qp["q"]
    per = _periods(df)
    d, h = qp.get("desde"), qp.get("hasta")
    if per and d in per and h in per:
        s["periodo"] = (min(d, h), max(d, h))


def share_view() -> None:
    """Escribe los filtros activos en la URL (para compartir la vista)."""
    s = st.session_state.get(PKEY, {})
    params = {k: "|".join(map(str, v)) for k, v in s.items() if k in MULTI and v}
    if s.get("mora") and s["mora"] != "Todos":
        params["mora"] = s["mora"]
    if s.get("q"):
        params["q"] = s["q"]
    if _valid_range(s.get("periodo")):
        params["desde"], params["hasta"] = s["periodo"]
    st.query_params.clear()
    st.query_params.update(params)


def reset_filters(df: pd.DataFrame | None = None) -> None:
    from core.data import get_active_df

    st.session_state[PKEY] = default_state(df if df is not None else get_active_df())
    st.query_params.clear()


def _valid_range(v) -> bool:
    return isinstance(v, (tuple, list)) and len(v) == 2 and all(isinstance(x, str) for x in v)


def _sync(name: str) -> None:
    v = st.session_state.get(f"flt_{name}")
    if name == "periodo":
        if not _valid_range(v):
            return
        v = (min(v), max(v))
    st.session_state[PKEY][name] = v


def _options(df: pd.DataFrame, col: str) -> list:
    if col == "y_pred":
        present = set(df["y_pred"].dropna().astype(str))
        return [r for r in RISK_ORDER if r in present]
    vals = df[col].dropna().astype(str).unique().tolist()
    return sorted(vals, key=lambda x: nrm(x))


def _multiselect(name: str, df: pd.DataFrame, opts_df: pd.DataFrame | None = None, placeholder: str = "Todos") -> None:
    label, col = MULTI[name]
    opts = _options(opts_df if opts_df is not None else df, col)
    s = st.session_state[PKEY]
    s[name] = [v for v in s[name] if v in opts]
    st.session_state[f"flt_{name}"] = s[name]
    st.multiselect(label, opts, key=f"flt_{name}", on_change=_sync, args=(name,), placeholder=placeholder)


def render_sidebar_filters(df: pd.DataFrame) -> None:
    """Dibuja todos los filtros en la barra lateral."""
    s = get_state(df)
    per = _periods(df)

    with st.sidebar:
        hdr, btn = st.columns([3, 2], vertical_alignment="center")
        hdr.markdown("**Filtros de cartera**")
        btn.button("↺ Limpiar", on_click=reset_filters, args=(df,), width="stretch", key="flt_reset",
                   help="Restablecer todos los filtros")

        # --- búsqueda ---
        st.session_state["flt_q"] = s["q"]
        st.text_input("Buscar estudiante", key="flt_q", on_change=_sync, args=("q",),
                      placeholder="Nombre, ID o llave del crédito", label_visibility="collapsed")

        # --- periodo ---
        if per:
            if not _valid_range(s["periodo"]) or s["periodo"][0] not in per or s["periodo"][1] not in per:
                s["periodo"] = (per[0], per[-1])
            st.session_state["flt_periodo"] = s["periodo"]
            st.select_slider("Periodo de aprobación", options=per, key="flt_periodo", on_change=_sync,
                             args=("periodo",), format_func=fmt_period)

        # --- riesgo + mora ---
        _multiselect("riesgo", df)
        st.session_state["flt_mora"] = s["mora"]
        st.segmented_control("Mora en Datacrédito", MORA_OPTS, key="flt_mora", on_change=_sync, args=("mora",),
                             selection_mode="single")

        with st.expander("🎓 Académico", expanded=bool(s["facultad"] or s["cluster"] or s["programa"])):
            _multiselect("facultad", df)
            _multiselect("cluster", df)
            sub = df
            if s["facultad"]:
                sub = sub[sub["facultad"].isin(s["facultad"])]
            if s["cluster"]:
                sub = sub[sub["programa_cluster"].isin(s["cluster"])]
            _multiselect("programa", df, opts_df=sub)
            _multiselect("nivel", df)
            _multiselect("sede", df)

        with st.expander("💳 Crédito y perfil", expanded=False):
            _multiselect("tipo_interes", df)
            _multiselect("cliente", df)
            _multiselect("tipo_estudiante", df)
            _multiselect("cohorte", df)
            _multiselect("departamento", df)
            _multiselect("ruta", df)

        fdf = apply_filters(df)
        share = len(fdf) / max(len(df), 1)
        st.progress(share, text=f"{fmt_int(len(fdf))} de {fmt_int(len(df))} créditos ({share * 100:.0f} %)")
        st.button("🔗 Fijar vista en la URL", on_click=share_view, width="stretch", key="flt_share",
                  help="Copia la URL del navegador después de pulsar para compartir esta vista filtrada")


def apply_filters(df: pd.DataFrame, state: dict | None = None, skip: set[str] | None = None) -> pd.DataFrame:
    """Aplica los filtros globales. ``skip`` permite ignorar algunos (p. ej. {'riesgo'})."""
    s = state if state is not None else st.session_state.get(PKEY)
    if not s:
        return df
    skip = skip or set()
    mask = np.ones(len(df), dtype=bool)
    for name, (_, col) in MULTI.items():
        vals = s.get(name) or []
        if vals and name not in skip:
            mask &= df[col].astype(str).isin([str(v) for v in vals]).to_numpy()
    if _valid_range(s.get("periodo")) and "periodo" not in skip:
        d, h = s["periodo"]
        p = df["periodo"].astype(str)
        mask &= ((p >= d) & (p <= h)).to_numpy()
    if s.get("mora") == "Con mora" and "mora" not in skip:
        mask &= (df["mora_flag"] == 1).to_numpy()
    elif s.get("mora") == "Sin mora" and "mora" not in skip:
        mask &= (df["mora_flag"] == 0).to_numpy()
    q = (s.get("q") or "").strip()
    if q and "q" not in skip:
        nq = nrm(q)
        hay = (df["nombre"].map(nrm) + " " + df["id_estudiante"].astype(str) + " " + df["llave2"].astype(str))
        mask &= hay.str.contains(nq, regex=False).to_numpy()
    return df[mask]


def active_chips(df: pd.DataFrame | None = None) -> list[tuple[str, str]]:
    s = st.session_state.get(PKEY) or {}
    chips = []
    per = _periods(df) if df is not None else []
    if _valid_range(s.get("periodo")) and per and tuple(s["periodo"]) != (per[0], per[-1]):
        chips.append(("Periodo", f"{fmt_period(s['periodo'][0])} – {fmt_period(s['periodo'][1])}"))
    for name, (label, _) in MULTI.items():
        v = s.get(name) or []
        if v:
            chips.append((label, ", ".join(map(str, v[:3])) + (f" +{len(v) - 3}" if len(v) > 3 else "")))
    if s.get("mora") and s["mora"] != "Todos":
        chips.append(("Mora", s["mora"]))
    if s.get("q"):
        chips.append(("Búsqueda", s["q"]))
    return chips
