"""Cargar y predecir — ¿cómo puntúa el SAT un archivo nuevo de créditos y lo lleva a todo el tablero?

Flujo guiado en cuatro pasos (DE-02 · ME-02 · DB-02):

1. **Plantilla y muestra**: plantilla CSV con las 42 variables del modelo o una muestra de ejemplo.
2. **Cargar archivo**: CSV (``,`` o ``;``; UTF-8 o Latin-1) o Excel ``.xlsx``.
3. **Validar**: esquema (columnas faltantes) + reglas de calidad automáticas (``core.quality``).
4. **Predecir y usar**: motor API / modelo local / ``y_pred`` del archivo → prioridad, ruta, descarga
   y activación del archivo como dataset de todo el tablero.

El estado vive en ``st.session_state`` con claves ``carga_*``. Esta página no usa filtros globales.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import badge, download_bar, esc, footer, insight, insight_row, kpi_card, kpi_row, page_header, section
from core.config import ACTION_ROUTES, BASE_DATA_PATH, MODEL_FEATURES, RISK_COLORS, RISK_ORDER
from core.data import enrich, get_active_meta, reset_active_df, set_active_df
from core.engine import engine_status, missing_required, predict
from core.nav import PAGES
from core.quality import quality_score, run_quality
from core.theme import fmt_cop, fmt_int, fmt_pct, show_fig

P = "carga"
PRED_COLS = ["y_pred", "y_true", "proba_pred", "proba_alto", "proba_medio", "proba_bajo", "motor", "y_categorica"]
EXTRA_SAMPLE = ["id_estudiante", "ciudad_norm", "departamento", "latitud", "longitud"]
QUALITY_OK = 0.90
MOTOR_TXT = {"api": "🌐 API del modelo", "local": "💻 Modelo local", "archivo": "📄 y_pred del archivo"}


# ================= Utilidades (cacheadas) =================

@st.cache_data(show_spinner=False)
def _raw_base() -> pd.DataFrame:
    """Base oficial tal como llega al modelo (sin datos de contacto ni objetivo)."""
    d = pd.read_csv(BASE_DATA_PATH)
    return d[[c for c in MODEL_FEATURES + EXTRA_SAMPLE if c in d.columns]]


@st.cache_data(show_spinner=False)
def _template_csv() -> bytes:
    return _raw_base()[MODEL_FEATURES].head(20).to_csv(index=False).encode("utf-8-sig")


def _sample() -> pd.DataFrame:
    return _raw_base().sample(300, random_state=42).reset_index(drop=True)


def _norm_cols(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d.columns = [str(c).replace("﻿", "").strip().lower() for c in d.columns]
    return d.loc[:, ~d.columns.duplicated()]


@st.cache_data(show_spinner="Leyendo el archivo…")
def _read_file(data: bytes, name: str) -> tuple[pd.DataFrame, str]:
    """Lee CSV (separador y codificación autodetectados) o Excel. Devuelve (df, descripción)."""
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data)), "Excel"
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        head = text.split("\n", 1)[0]
        sep = ";" if head.count(";") > head.count(",") else ","
        return pd.read_csv(io.StringIO(text), sep=sep, low_memory=False), f"CSV · separador «{sep}» · {enc.replace('-sig', '')}"
    raise ValueError("No se pudo decodificar el archivo (usa UTF-8 o Latin-1).")


def _has_ypred(d: pd.DataFrame) -> bool:
    if "y_pred" not in d.columns:
        return False
    s = d["y_pred"].astype(str).str.strip().str.capitalize()
    return bool(s.isin(RISK_ORDER).any())


def _reset(keep_upload: bool = False) -> None:
    for k in [k for k in st.session_state if str(k).startswith("carga_") and k not in ("carga_upl_n", "carga_upl_sig")]:
        del st.session_state[k]
    if not keep_upload:
        st.session_state["carga_upl_n"] = st.session_state.get("carga_upl_n", 0) + 1
        st.session_state.pop("carga_upl_sig", None)


def _set_raw(d: pd.DataFrame, name: str, fmt: str) -> None:
    _reset(keep_upload=True)
    d = _norm_cols(d)
    st.session_state.update({"carga_raw": d, "carga_name": name, "carga_fmt": fmt})
    res = run_quality(d)
    st.session_state["carga_quality"] = res
    st.session_state["carga_score"] = quality_score(res)


# ================= Estilos del flujo =================

def _css() -> None:
    st.markdown("""
<style>
.cg-steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:4px 0 6px 0}
.cg-step{display:flex;gap:12px;align-items:center;padding:12px 14px;border-radius:14px;
  border:1px solid var(--sat-border);background:var(--sat-surface);box-shadow:var(--sat-shadow)}
.cg-step .n{flex:0 0 34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-weight:800;font-family:'Space Grotesk',sans-serif;background:var(--sat-surface-2);color:var(--sat-muted);
  border:1px solid var(--sat-border)}
.cg-step .t{font-weight:700;font-size:14px;color:var(--sat-text);line-height:1.2}
.cg-step .s{font-size:11.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--sat-muted)}
.cg-step.done .n{background:var(--sat-bajo);color:#fff;border-color:var(--sat-bajo)}
.cg-step.done .s{color:var(--sat-bajo)}
.cg-step.now{border:2px solid var(--sat-accent)}
.cg-step.now .n{background:var(--sat-accent);color:#141414;border-color:var(--sat-accent)}
.cg-step.now .s{color:var(--sat-text)}
.cg-step.todo{opacity:.62}
.cg-head{display:flex;align-items:center;gap:10px;margin:22px 0 6px 0}
.cg-head .n{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-weight:800;background:var(--sat-accent);color:#141414;font-family:'Space Grotesk',sans-serif}
.cg-head.done .n{background:var(--sat-bajo);color:#fff}
.cg-head.todo .n{background:var(--sat-surface-2);color:var(--sat-muted);border:1px solid var(--sat-border)}
.cg-head h3{margin:0;font-size:20px;color:var(--sat-text);padding:0}
.cg-head .d{color:var(--sat-muted);font-size:13.5px;margin-left:4px}
.cg-light{display:flex;gap:14px;align-items:center;padding:14px 16px;border-radius:14px;border:1px solid var(--sat-border);
  background:var(--sat-surface)}
.cg-light .dot{flex:0 0 18px;height:18px;border-radius:50%}
.cg-light .ttl{font-weight:800;color:var(--sat-text);font-size:15px}
.cg-light .txt{color:var(--sat-muted);font-size:13.5px}
.cg-eng{padding:12px 14px;border-radius:14px;border:1px solid var(--sat-border);background:var(--sat-surface);min-height:96px}
.cg-eng .h{display:flex;justify-content:space-between;gap:8px;align-items:center;font-weight:700;color:var(--sat-text)}
.cg-eng .x{font-size:12.5px;color:var(--sat-muted);margin-top:6px;overflow-wrap:anywhere}
@media (max-width: 900px){.cg-steps{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>""", unsafe_allow_html=True)


STEP_NAMES = ["Plantilla y muestra", "Cargar archivo", "Validar", "Predecir y usar"]
STATE_TXT = {"done": "✓ Hecho", "now": "● Actual", "todo": "Pendiente"}


def _stepper(states: list[str]) -> None:
    items = "".join(
        f"<div class='cg-step {s}'><div class='n'>{'✓' if s == 'done' else i + 1}</div>"
        f"<div><div class='s'>{STATE_TXT[s]}</div><div class='t'>{esc(STEP_NAMES[i])}</div></div></div>"
        for i, s in enumerate(states))
    st.markdown(f"<div class='cg-steps'>{items}</div>", unsafe_allow_html=True)


def _step_head(i: int, state: str, desc: str) -> None:
    st.markdown(f"<div class='cg-head {state}'><div class='n'>{'✓' if state == 'done' else i + 1}</div>"
                f"<h3>Paso {i + 1} · {esc(STEP_NAMES[i])}</h3><span class='d'>{esc(desc)}</span></div>",
                unsafe_allow_html=True)


def _light(color: str, title: str, text: str) -> None:
    st.markdown(f"<div class='cg-light' style='border-left:5px solid {color}'>"
                f"<div class='dot' style='background:{color}'></div><div><div class='ttl'>{esc(title)}</div>"
                f"<div class='txt'>{text}</div></div></div>", unsafe_allow_html=True)


# ================= Estado =================
ss = st.session_state
raw: pd.DataFrame | None = ss.get("carga_raw")
result: pd.DataFrame | None = ss.get("carga_result")

missing = missing_required(raw) if raw is not None else []
has_pred = raw is not None and _has_ypred(raw)
score = (ss.get("carga_score") or {}).get("Global", np.nan)
schema_ok = raw is not None and not missing
can_go = schema_ok or has_pred
loaded = raw is not None

states = ["done" if loaded else "now",
          "done" if loaded else "todo",
          ("done" if can_go else "now") if loaded else "todo",
          ("done" if result is not None else ("now" if can_go else "todo"))]
if loaded and not can_go:
    states[3] = "todo"

meta = get_active_meta()
page_header(
    "Cargar y predecir",
    "Sube un archivo de créditos nuevos, valídalo con las reglas de calidad y obtén el riesgo, la prioridad "
    "y la ruta de gestión de cada crédito en menos de un minuto.",
    eyebrow="Operación · DE-02 · ME-02",
    highlight=f"Dataset activo: {meta['name']}",
)
_css()
_stepper(states)

# ================= Paso 1 · Plantilla y muestra =================
_step_head(0, states[0], "¿Primera vez? Descarga la plantilla o prueba con datos de ejemplo.")
with st.container(border=True):
    c1, c2, c3 = st.columns([1.25, 1, 1], vertical_alignment="center")
    with c1:
        st.markdown(f"**Formato esperado:** {len(MODEL_FEATURES)} columnas del modelo, una fila por crédito. "
                    "Si el archivo ya trae `y_pred` (Alto / Medio / Bajo) se puede usar sin motor.")
    with c2:
        st.download_button("⬇️ Descargar plantilla CSV", _template_csv(), file_name="plantilla_sat_42_variables.csv",
                           mime="text/csv", key=f"{P}_tpl", width="stretch",
                           help="42 columnas del modelo con 20 filas de ejemplo de la base oficial.")
    with c3:
        if st.button("🧪 Probar con una muestra de ejemplo", key=f"{P}_sample", type="primary", width="stretch",
                     help="300 créditos de la base oficial sin predicciones ni riesgo observado."):
            _set_raw(_sample(), "muestra_ejemplo_300.csv", "Muestra de la base oficial")
            st.rerun()
    with st.expander("Ver las 42 columnas de la plantilla"):
        st.markdown(" ".join(f"`{c}`" for c in MODEL_FEATURES))

# ================= Paso 2 · Cargar archivo =================
_step_head(1, states[1], "CSV (separador , o ;  ·  UTF-8 o Latin-1) o Excel .xlsx.")
with st.container(border=True):
    up = st.file_uploader("Arrastra aquí tu archivo de créditos", type=["csv", "xlsx"],
                          key=f"{P}_upl_{ss.get('carga_upl_n', 0)}", help="Máximo recomendado: 50.000 filas.")
    if up is not None:
        sig = f"{up.name}-{up.size}"
        if ss.get("carga_upl_sig") != sig:
            ss["carga_upl_sig"] = sig
            try:
                d, fmt = _read_file(up.getvalue(), up.name)
                if d.empty:
                    st.error("El archivo no tiene filas.")
                else:
                    _set_raw(d, up.name, fmt)
                    st.rerun()
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")
    if loaded:
        cA, cB = st.columns([4, 1], vertical_alignment="center")
        with cA:
            st.markdown(f"{badge('Cargado', 'bajo')} &nbsp;**{esc(ss['carga_name'])}** · "
                        f"{fmt_int(len(raw))} filas × {fmt_int(raw.shape[1])} columnas · {esc(ss.get('carga_fmt', ''))}",
                        unsafe_allow_html=True)
        with cB:
            if st.button("↺ Empezar de nuevo", key=f"{P}_reset", width="stretch"):
                _reset()
                st.rerun()
        st.dataframe(raw.head(8), width="stretch", height=250, hide_index=True)
    else:
        st.caption("Aún no hay archivo. Usa el paso 1 para probar con la muestra de ejemplo.")

# ================= Paso 3 · Validar =================
_step_head(2, states[2], "Esquema del modelo + controles de calidad automáticos (ME-02).")
with st.container(border=True):
    if not loaded:
        st.caption("La validación se ejecuta automáticamente al cargar un archivo.")
    else:
        qres: pd.DataFrame = ss["carga_quality"]
        fails = qres[qres["incumplen"] > 0].sort_values(["severidad", "incumplen"], ascending=[True, False])
        n_da02 = int(qres.loc[qres["id"] == "VA-02", "incumplen"].sum())
        # Semáforo
        if schema_ok and (np.isnan(score) or score >= QUALITY_OK):
            _light(RISK_COLORS["Bajo"], "Listo para predecir",
                   f"El archivo trae las {len(MODEL_FEATURES)} variables del modelo y la calidad global es "
                   f"{fmt_pct(score)}.")
        elif schema_ok:
            _light(RISK_COLORS["Medio"], "Se puede predecir, con advertencias",
                   f"Esquema completo, pero la calidad global ({fmt_pct(score)}) está por debajo de "
                   f"{fmt_pct(QUALITY_OK, 0)}. Revisa las reglas incumplidas antes de gestionar.")
        elif has_pred:
            _light(RISK_COLORS["Medio"], "Esquema incompleto, pero el archivo trae y_pred",
                   f"Faltan {len(missing)} variables del modelo: no se puede puntuar de nuevo, pero sí usar "
                   "las predicciones que ya vienen en el archivo.")
        else:
            _light(RISK_COLORS["Alto"], "Bloqueado: faltan columnas del modelo",
                   f"Faltan {len(missing)} de {len(MODEL_FEATURES)} variables y el archivo no trae y_pred. "
                   "Completa las columnas con la plantilla del paso 1.")
        st.write("")
        kpi_row([
            kpi_card("Columnas del modelo", f"{len(MODEL_FEATURES) - len(missing)} / {len(MODEL_FEATURES)}",
                     "completas" if not missing else f"faltan {len(missing)}", tone="bajo" if not missing else "alto",
                     bar=(len(MODEL_FEATURES) - len(missing)) / len(MODEL_FEATURES)),
            kpi_card("Calidad global", fmt_pct(score), f"meta ≥ {fmt_pct(QUALITY_OK, 0)} · {len(qres)} reglas",
                     tone="bajo" if score >= QUALITY_OK else "medio", bar=score if not np.isnan(score) else 0),
            kpi_card("Reglas con incumplimientos", fmt_int(len(fails)), f"de {len(qres)} ejecutadas",
                     tone="medio" if len(fails) else "bajo"),
            kpi_card("Financiación > 85 % (DA-02)", fmt_int(n_da02), "créditos a remediar",
                     tone="alto" if n_da02 else "bajo"),
        ])
        if missing:
            st.error("**Columnas faltantes:** " + ", ".join(f"`{c}`" for c in missing))
        dims = {k: v for k, v in (ss.get("carga_score") or {}).items() if k != "Global"}
        if dims:
            st.write("")
            st.caption("Cumplimiento por dimensión: " + " · ".join(f"**{k}** {fmt_pct(v)}" for k, v in dims.items()))
        with st.expander(f"Ver reglas con incumplimientos ({len(fails)})", expanded=False):
            if fails.empty:
                st.success("Todas las reglas se cumplen al 100 %.")
            else:
                st.dataframe(
                    fails[["id", "dimension", "campo", "regla", "severidad", "incumplen", "evaluados", "cumplimiento"]],
                    hide_index=True, width="stretch",
                    column_config={
                        "id": "Regla", "dimension": "Dimensión", "campo": "Campo", "regla": "Descripción",
                        "severidad": "Severidad", "incumplen": st.column_config.NumberColumn("Incumplen", format="%d"),
                        "evaluados": st.column_config.NumberColumn("Evaluados", format="%d"),
                        "cumplimiento": st.column_config.ProgressColumn("Cumplimiento", min_value=0, max_value=1,
                                                                        format="percent"),
                    })

# ================= Paso 4 · Predecir y usar =================
_step_head(3, states[3], "Elige el motor, puntúa y lleva el resultado a todo el tablero.")
status = engine_status()
api, local = status["api"], status["local"]
with st.container(border=True):
    e1, e2, e3 = st.columns(3)
    with e1:
        if api.get("ok"):
            b, x = badge("En línea", "bajo"), (f"{esc(api.get('url'))} · modelo v{esc(api.get('model_version', '?'))}"
                                                f" · {fmt_int(api.get('latency_ms'))} ms")
        elif api.get("configured"):
            b, x = badge("Sin respuesta", "alto"), f"{esc(api.get('url'))} · {esc(api.get('error', ''))[:120]}"
        else:
            b, x = badge("No configurada", "neutral"), "Define <code>[api] url</code> en secrets o SAT_API_URL."
        st.markdown(f"<div class='cg-eng'><div class='h'>🌐 API FastAPI {b}</div><div class='x'>{x}</div></div>",
                    unsafe_allow_html=True)
    with e2:
        b = badge("Instalado", "bajo") if local["installed"] else badge("No instalado", "neutral")
        x = ("Pipeline empaquetado: entrega probabilidades por clase." if local["installed"]
             else "Instala <code>requirements-model.txt</code> para puntuar sin API.")
        st.markdown(f"<div class='cg-eng'><div class='h'>💻 Modelo local {b}</div><div class='x'>{x}</div></div>",
                    unsafe_allow_html=True)
    with e3:
        b = badge("Disponible", "bajo") if has_pred else badge("No aplica", "neutral")
        x = ("El archivo trae la columna <code>y_pred</code>." if has_pred
             else "Sirve si el archivo ya viene puntuado (columna <code>y_pred</code>).")
        st.markdown(f"<div class='cg-eng'><div class='h'>📄 y_pred del archivo {b}</div><div class='x'>{x}</div></div>",
                    unsafe_allow_html=True)
    with st.expander("¿Cómo habilitar un motor de predicción?"):
        st.markdown("**API** — en `.streamlit/secrets.toml` (o variable de entorno `SAT_API_URL`):")
        st.code('[api]\nurl = "https://mi-api-morosidad.example.com"', language="toml")
        st.markdown("**Modelo local** — instala el paquete del modelo (≈ 550 MB en memoria):")
        st.code("pip install -r Dashboard/alerta_temprana/requirements-model.txt", language="bash")

    options = []
    if schema_ok and api.get("ok"):
        options.append("api")
    if schema_ok and local["installed"]:
        options.append("local")
    if has_pred:
        options.append("archivo")

    if not loaded:
        st.caption("Carga un archivo (paso 2) para habilitar la predicción.")
    elif not options:
        st.warning("No hay un motor disponible para este archivo: habilita la API o el modelo local, "
                   "o sube un archivo que ya traiga la columna `y_pred`.")
    else:
        pref = status["preferred"] if status["preferred"] in options else options[0]
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        with c1:
            motor = st.segmented_control("Motor de predicción", options, default=pref, key=f"{P}_motor",
                                         format_func=lambda o: MOTOR_TXT[o]) or pref
        with c2:
            go_btn = st.button(f"▶ Predecir {fmt_int(len(raw))} créditos", key=f"{P}_run", type="primary",
                               width="stretch")
        if motor == "api":
            n_null = int(raw.reindex(columns=MODEL_FEATURES).isna().any(axis=1).sum())
            if n_null:
                st.caption(f"ℹ️ La API solo puntúa filas completas: {fmt_int(n_null)} filas con vacíos quedarán sin predicción.")
        if go_btn:
            drop = [] if motor == "archivo" else [c for c in PRED_COLS if c != "y_true"]
            base = raw.drop(columns=drop, errors="ignore")
            try:
                if motor == "archivo":
                    label = "Predicción del archivo"
                    out = base
                else:
                    bar = st.progress(0.0, text="Enviando lotes al motor…")
                    pr = predict(base, engine=motor, progress=lambda f: bar.progress(f, text=f"Puntuando… {f:.0%}"))
                    bar.progress(1.0, text="Predicción completa")
                    label = str(pr["motor"].iloc[0]) if len(pr) else MOTOR_TXT[motor]
                    out = base.join(pr)
                res = enrich(out)
                res["origen"] = ss["carga_name"]
                res["motor"] = label
                ss["carga_result"], ss["carga_engine"] = res, label
                ss.pop("carga_applied", None)
                st.rerun()
            except Exception as exc:
                st.error(f"**La predicción falló** con el motor {MOTOR_TXT[motor]}: {exc}")
                st.caption("Revisa que la API esté en línea o prueba con otro motor.")

# ---------- Resultado ----------
if result is not None:
    scored = result[result["y_pred"].notna()]
    n, ns = len(result), len(scored)
    cnt = scored["y_pred"].value_counts().reindex(RISK_ORDER, fill_value=0)
    section("Resultado de la predicción",
            f"{fmt_int(ns)} de {fmt_int(n)} créditos puntuados con {ss.get('carga_engine', '')}.", kicker="Paso 4")
    kpi_row([
        kpi_card("Créditos puntuados", fmt_int(ns), f"{fmt_int(n - ns)} sin predicción" if n > ns else "100 % del archivo",
                 tone="ink", bar=ns / max(n, 1)),
        kpi_card("Riesgo Alto", fmt_int(cnt["Alto"]), fmt_pct(cnt["Alto"] / max(ns, 1)) + " de lo puntuado", tone="alto",
                 bar=cnt["Alto"] / max(ns, 1)),
        kpi_card("Riesgo Medio", fmt_int(cnt["Medio"]), fmt_pct(cnt["Medio"] / max(ns, 1)), tone="medio",
                 bar=cnt["Medio"] / max(ns, 1)),
        kpi_card("Riesgo Bajo", fmt_int(cnt["Bajo"]), fmt_pct(cnt["Bajo"] / max(ns, 1)), tone="bajo",
                 bar=cnt["Bajo"] / max(ns, 1)),
        kpi_card("Monto en Alto", fmt_cop(scored.loc[scored["y_pred"] == "Alto", "valor_financiacion"].sum()),
                 "valor financiado a proteger", tone="accent"),
    ])
    g1, g2 = st.columns(2)
    with g1, st.container(border=True):
        st.markdown("**Riesgo predicho** · qué parte del archivo requiere gestión")
        fig = go.Figure(go.Bar(
            y=RISK_ORDER, x=cnt.values, orientation="h", marker_color=[RISK_COLORS[r] for r in RISK_ORDER],
            text=[f"{fmt_int(v)} · {fmt_pct(v / max(ns, 1))}" for v in cnt.values], textposition="auto",
            customdata=[fmt_int(v) for v in cnt.values], hovertemplate="%{y}: %{customdata} créditos<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="Créditos")
        show_fig(fig, key=f"{P}_dist", height=300, legend=False)
    with g2, st.container(border=True):
        st.markdown("**Rutas de gestión** · carga de trabajo por ruta y SLA")
        rc = scored["ruta"].value_counts().reindex(list(ACTION_ROUTES), fill_value=0)
        names = [f"{r} · {ACTION_ROUTES[r]['nombre']}" for r in rc.index]
        fig = go.Figure(go.Bar(
            y=names, x=rc.values, orientation="h", marker_color=[ACTION_ROUTES[r]["color"] for r in rc.index],
            text=[fmt_int(v) for v in rc.values], textposition="auto",
            customdata=[[ACTION_ROUTES[r]["sla"], fmt_int(v)] for r, v in rc.items()],
            hovertemplate="%{y}<br>%{customdata[1]} créditos · SLA %{customdata[0]}<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title="Créditos")
        show_fig(fig, key=f"{P}_rutas", height=300, legend=False)

    tips = [insight("Primero, contacto inmediato",
                    f"<b>{fmt_int(rc.get('R1', 0))}</b> créditos van a la ruta R1 (SLA {ACTION_ROUTES['R1']['sla']}): "
                    f"{esc(ACTION_ROUTES['R1']['accion'])}", tone="alto", icon="📞")]
    if result["has_truth"].any():
        acc = float(result["acierto"].mean())
        tips.append(insight("El archivo trae riesgo observado",
                            f"La concordancia entre predicho y observado es <b>{fmt_pct(acc)}</b> "
                            f"en {fmt_int(int(result['acierto'].notna().sum()))} créditos.", tone="info", icon="🎯"))
    if result["proba_pred"].isna().all():
        tips.append(insight("Sin probabilidades",
                            "El motor entrega solo la clase: la prioridad usa una confianza neutra (0,5) "
                            "combinada con el monto y la mora histórica.", tone="accent", icon="ℹ️"))
    insight_row(tips)

    st.write("")
    st.markdown("**Créditos ordenados por prioridad** (0–100: riesgo 60 %, exposición 25 %, mora 15 %)")
    cols = ["prioridad", "ruta", "y_pred", "proba_pred", "nombre", "llave2", "programa", "valor_financiacion",
            "cuotas", "mora_txt"]
    table = scored.sort_values("prioridad", ascending=False)[cols].copy()
    table["y_pred"] = table["y_pred"].astype(str)
    if table["proba_pred"].isna().all():
        table = table.drop(columns="proba_pred")
    table["ruta"] = table["ruta"].map(lambda r: f"{r} · {ACTION_ROUTES[r]['nombre']}")
    st.dataframe(
        table, hide_index=True, width="stretch", height=380,
        column_config={
            "prioridad": st.column_config.ProgressColumn("Prioridad", min_value=0, max_value=100, format="%.0f"),
            "ruta": "Ruta", "y_pred": "Riesgo", "nombre": "Estudiante", "llave2": "Crédito", "programa": "Programa",
            "proba_pred": st.column_config.NumberColumn("Confianza", format="percent"),
            "valor_financiacion": st.column_config.NumberColumn("Valor financiado (COP)", format="localized"),
            "cuotas": st.column_config.NumberColumn("Cuotas", format="%d"), "mora_txt": "Mora histórica",
        })
    export = scored.sort_values("prioridad", ascending=False)[
        ["llave2", "id_estudiante", "nombre", "programa", "valor_financiacion", "y_pred", "proba_pred",
         "proba_alto", "proba_medio", "proba_bajo", "prioridad", "ruta", "motor"]].copy()
    export["ruta_nombre"] = export["ruta"].map(lambda r: ACTION_ROUTES[r]["nombre"])
    export = export.dropna(axis=1, how="all")
    download_bar(export, f"predicciones_{ss['carga_name'].rsplit('.', 1)[0]}", key=f"{P}_dl", label="Descargar")

    with st.container(border=True):
        a1, a2 = st.columns([2.2, 1.3], vertical_alignment="center")
        with a1:
            if ss.get("carga_applied"):
                st.markdown(f"✅ **{esc(ss['carga_name'])}** es ahora el dataset activo: Resumen, Cola de gestión, "
                            "Segmentos y Modelo muestran estos créditos.", unsafe_allow_html=True)
            else:
                st.markdown("**¿Listo?** Activa este resultado para que todas las páginas (resumen, cola de gestión, "
                            "segmentos) trabajen con estos créditos en lugar de la base oficial.")
        with a2:
            if st.button("🚀 Usar estos datos en todo el tablero", key=f"{P}_apply", type="primary", width="stretch",
                         disabled=bool(ss.get("carga_applied"))):
                set_active_df(result, name=ss["carga_name"], source="upload", engine=ss.get("carga_engine", ""))
                ss["carga_applied"] = True
                st.rerun()
        if ss.get("carga_applied"):
            l1, l2, _ = st.columns([1, 1, 2])
            with l1:
                st.page_link(PAGES["resumen"][0], label="Ir al Resumen ejecutivo", icon=":material/dashboard:")
            with l2:
                st.page_link(PAGES["cola"][0], label="Ir a la Cola de gestión", icon=":material/checklist:")

if meta.get("source") == "upload":
    st.write("")
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    with c1:
        st.caption(f"El tablero usa hoy «{meta['name']}» ({fmt_int(meta.get('rows'))} filas · {meta.get('engine', '')}).")
    with c2:
        if st.button("↩ Volver a la base oficial", key=f"{P}_base", width="stretch"):
            reset_active_df()
            ss.pop("carga_applied", None)
            st.rerun()

footer()
