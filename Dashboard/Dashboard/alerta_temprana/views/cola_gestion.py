"""Cola de gestión — ¿a quién contactamos esta semana, por qué ruta y cuánto riesgo cubre la capacidad del equipo?

Convierte las predicciones del modelo en una cola priorizada con rutas de cobro preventivo (R1–R4), muestra
cuánto riesgo real captura la capacidad semanal frente a un orden aleatorio y exporta la cola (DB-02) y el
archivo de intercambio con el sistema de gestión de cartera (DB-04).
"""
from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import (badge, empty_state, esc, filter_chips, footer, kpi_card, page_header, risk_badge,
                             section)
from core.config import ACTION_ROUTES, RECAUDO_BASELINE, RISK_COLORS, RISK_ICONS
from core.data import DEFAULT_WEIGHTS, compute_priority, get_active_df, to_excel_bytes
from core.filters import active_chips, apply_filters
from core.theme import fmt_cop, fmt_delta_pp, fmt_int, fmt_num, fmt_pct, pal, show_fig, theme_mode

ROUTES = list(ACTION_ROUTES)                     # ["R1", "R2", "R3", "R4"]
QUEUE_ROUTES = ["R1", "R2"]                      # rutas que requieren gestor por diseño
ROUTE_SHORT = {"R1": "Inmediato", "R2": "Preventivo", "R3": "Automático", "R4": "Monitoreo"}
ROUTE_EMOJI = {"R1": "🟥", "R2": "🟧", "R3": "🟦", "R4": "🟩"}

_CFG_KEY = "_cola_cfg"
_DEFAULTS = {
    "w_riesgo": int(round(DEFAULT_WEIGHTS["riesgo"] * 100)),
    "w_exposicion": int(round(DEFAULT_WEIGHTS["exposicion"] * 100)),
    "w_mora": int(round(DEFAULT_WEIGHTS["mora"] * 100)),
    "capacidad": 200, "gestores": 4, "vista": "Plan de la semana",
    "sc_semanas": 4, "sc_contacto": 70, "sc_ef_R1": 40, "sc_ef_R2": 30,
}

# Esquema fijo del archivo de intercambio con el sistema de gestión de cartera (DB-04).
DB04_SCHEMA = [
    ("id_estudiante", "texto", "Sí", "Identificador institucional del estudiante."),
    ("llave2", "texto", "Sí", "Llave única del crédito (llave de integración)."),
    ("nombre", "texto", "Sí", "Nombre del estudiante (anonimizado en este prototipo)."),
    ("programa", "texto", "Sí", "Programa académico asociado al crédito."),
    ("riesgo_predicho", "texto {Alto, Medio, Bajo}", "Sí", "Clase de riesgo predicha por el modelo."),
    ("confianza", "decimal [0, 1]", "No", "Probabilidad de la clase predicha."),
    ("prioridad", "decimal [0, 100]", "Sí", "Índice de prioridad con los pesos vigentes."),
    ("ruta", "texto {R1, R2, R3, R4}", "Sí", "Ruta de gestión preventiva asignada."),
    ("accion", "texto", "Sí", "Acción recomendada para la ruta."),
    ("sla", "texto", "Sí", "Tiempo máximo de atención de la ruta."),
    ("valor_financiado", "entero (COP)", "Sí", "Valor financiado del crédito."),
    ("gestor_asignado", "texto 'Gestor k'", "Sí", "Gestor responsable según el reparto de la semana."),
    ("fecha_asignacion", "fecha ISO (AAAA-MM-DD)", "Sí", "Fecha en que se genera la asignación."),
    ("estado", "texto {PENDIENTE}", "Sí", "Estado inicial; lo actualiza el sistema de cartera."),
]
DB04_COLUMNS = [c[0] for c in DB04_SCHEMA]


# ============================================================================================
# Estado persistente de los controles (sobrevive a la navegación entre páginas)
# ============================================================================================
def _cfg() -> dict:
    cfg = st.session_state.setdefault(_CFG_KEY, {})
    for k, v in _DEFAULTS.items():
        cfg.setdefault(k, v)
    return cfg


def _sync(name: str) -> None:
    st.session_state[_CFG_KEY][name] = st.session_state.get(f"cola_{name}")


def _bind(name: str) -> dict:
    k = f"cola_{name}"
    st.session_state[k] = _cfg()[name]
    return {"key": k, "on_change": _sync, "args": (name,)}


def _reset_weights() -> None:
    for n in ("w_riesgo", "w_exposicion", "w_mora"):
        _cfg()[n] = _DEFAULTS[n]


# ============================================================================================
# Utilidades visuales
# ============================================================================================
def _dark() -> bool:
    return theme_mode() == "dark"


def _rc(r: str) -> str:
    """Color de marca de la ruta."""
    return "#7C9CFF" if (r == "R3" and _dark()) else ACTION_ROUTES[r]["color"]


def _ri(r: str) -> str:
    """Color de texto legible de la ruta."""
    d = _dark()
    return {"R1": "#FF6B70" if d else "#D13438", "R2": "#FFB547" if d else "#A8650A",
            "R3": "#8FA8FF" if d else "#3E63DD", "R4": "#46C487" if d else "#218358"}[r]


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _nb(text: str) -> str:
    return str(text).replace(" ", "&nbsp;")


def _grid(items: list[str], min_px: int = 200) -> None:
    st.markdown(f"<div class='cola-grid' style='--min:{min_px}px'>" + "".join(f"<div>{h}</div>" for h in items)
                + "</div>", unsafe_allow_html=True)


def _css() -> None:
    st.markdown(
        """
        <style>
        .cola-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(var(--min,200px),100%),1fr));
          gap:12px;align-items:stretch;}
        .cola-grid > div{min-width:0;}
        .cola-mini{font-size:12.5px;color:var(--sat-muted);margin:-2px 0 6px 0;line-height:1.5;}
        .cola-mini b{color:var(--sat-text);}
        .cola-wbar{display:inline-flex;width:110px;height:8px;border-radius:6px;overflow:hidden;gap:2px;
          background:var(--sat-surface-2);vertical-align:middle;margin:0 6px;}
        .cola-wbar i{display:block;height:100%;}

        .cola-route{position:relative;background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:16px;
          padding:14px 16px 14px 20px;box-shadow:var(--sat-shadow);height:100%;overflow:hidden;}
        .cola-route:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--rc);}
        .cola-route .top{display:flex;justify-content:space-between;align-items:center;gap:8px;}
        .cola-route .code{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:12.5px;letter-spacing:.04em;
          padding:2px 9px;border-radius:7px;color:var(--ri);background:color-mix(in srgb, var(--rc) 15%, transparent);}
        .cola-route .sla{font-size:11.5px;font-weight:700;color:var(--sat-muted);border:1px solid var(--sat-border);
          padding:2px 9px;border-radius:999px;white-space:nowrap;}
        .cola-route .name{font-weight:700;font-size:14px;margin-top:8px;color:var(--sat-text);white-space:nowrap;
          overflow:hidden;text-overflow:ellipsis;}
        .cola-route .big{font-family:'Space Grotesk','Inter',sans-serif;font-size:30px;font-weight:700;color:var(--sat-text);
          line-height:1.1;margin-top:4px;letter-spacing:-.02em;}
        .cola-route .big small{font-family:'Inter',sans-serif;font-size:12.5px;font-weight:500;color:var(--sat-muted);
          margin-left:6px;letter-spacing:0;}
        .cola-route .money{font-size:13px;color:var(--sat-muted);margin-top:2px;}
        .cola-route .money b{color:var(--sat-text);}
        .cola-route .obs{margin-top:12px;padding-top:10px;border-top:1px solid var(--sat-border);}
        .cola-route .row{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:12.5px;
          color:var(--sat-muted);}
        .cola-route .row b{color:var(--sat-text);font-size:15px;}
        .cola-route .lift{font-weight:800;color:var(--ri);white-space:nowrap;}
        .cola-route .meter{position:relative;height:8px;border-radius:6px;background:var(--sat-surface-2);
          margin:7px 0 3px 0;border:1px solid var(--sat-border);}
        .cola-route .meter i{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:var(--rc);}
        .cola-route .meter u{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--sat-text);border-radius:2px;}
        .cola-route .legend{font-size:11px;color:var(--sat-subtle);}

        .cola-stat{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:14px;padding:12px 14px;
          box-shadow:var(--sat-shadow);height:100%;}
        .cola-stat .k{font-size:12px;font-weight:600;color:var(--sat-muted);}
        .cola-stat .v{font-family:'Space Grotesk','Inter',sans-serif;font-size:26px;font-weight:700;color:var(--sat-text);
          line-height:1.15;letter-spacing:-.02em;}
        .cola-stat .s{font-size:12px;color:var(--sat-muted);margin-top:2px;}
        .cola-stat .s b{color:var(--sat-text);}
        .cola-stat.hl{border-left:4px solid var(--sat-accent);}

        .cola-case{background:var(--sat-surface);border:1px solid var(--sat-border);border-left:5px solid var(--rc);
          border-radius:16px;padding:14px 18px;box-shadow:var(--sat-shadow);margin:6px 0 14px 0;}
        .cola-case .hd{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;}
        .cola-case .nm{font-family:'Space Grotesk','Inter',sans-serif;font-size:19px;font-weight:700;color:var(--sat-text);}
        .cola-case .meta{font-size:12.5px;color:var(--sat-muted);margin-top:2px;}
        .cola-case .chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center;}
        .cola-case .g3{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px;}
        .cola-case .g3 > div{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:12px;padding:8px 10px;}
        .cola-case .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--sat-subtle);font-weight:700;}
        .cola-case .v{font-size:16px;font-weight:700;color:var(--sat-text);margin-top:2px;}
        .cola-case .act{margin-top:10px;font-size:13px;color:var(--sat-muted);line-height:1.5;}
        .cola-case .act b{color:var(--sat-text);}

        .cola-banner{display:flex;gap:12px;align-items:flex-start;border-radius:14px;padding:10px 14px;
          background:color-mix(in srgb, var(--sat-medio) 11%, var(--sat-surface));
          border:1px solid color-mix(in srgb, var(--sat-medio) 45%, transparent);color:var(--sat-text);font-size:13px;
          line-height:1.5;margin-bottom:10px;}
        .cola-banner .tag{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;
          background:var(--sat-medio);color:#141414;padding:3px 9px;border-radius:999px;white-space:nowrap;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================================
# Cálculos
# ============================================================================================
def _weights(cfg: dict) -> tuple[dict, bool]:
    """Pesos normalizados y si difieren de los de por defecto."""
    raw = {"riesgo": cfg["w_riesgo"], "exposicion": cfg["w_exposicion"], "mora": cfg["w_mora"]}
    total = sum(float(v or 0) for v in raw.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS), False
    w = {k: float(v or 0) / total for k, v in raw.items()}
    return w, any(abs(w[k] - DEFAULT_WEIGHTS[k]) > 1e-6 for k in w)


def _working_frame(df_all: pd.DataFrame, dff: pd.DataFrame, w: dict, custom: bool) -> pd.DataFrame:
    """Cartera filtrada con la prioridad vigente, ordenada de mayor a menor prioridad."""
    cols = ["llave2", "id_estudiante", "nombre", "programa", "y_pred", "y_true", "has_truth", "proba_pred",
            "valor_financiacion", "mora_flag", "exposicion_riesgo", "ruta", "prioridad"]
    W = dff[[c for c in cols if c in dff.columns]].copy()
    if custom:
        W["prioridad"] = compute_priority(df_all, w).reindex(W.index)
    W["exposicion_riesgo"] = W["exposicion_riesgo"].fillna(0.0)
    W["valor_financiacion"] = W["valor_financiacion"].fillna(0.0)
    W["_alto_obs"] = (W["has_truth"] & (W["y_true"].astype(str) == "Alto")).astype(int)
    return W.sort_values(["prioridad", "exposicion_riesgo", "llave2"], ascending=[False, False, True],
                         kind="mergesort").reset_index(drop=True)


def _assign(queue: pd.DataFrame, capacity: int, n_gestores: int) -> pd.DataFrame:
    """Programa la cola por semanas (capacidad) y la reparte entre gestores en round-robin."""
    q = queue.copy()
    pos = np.arange(len(q))
    q["_pos"] = pos
    q["semana"] = pos // capacity + 1
    q["gestor"] = "Gestor " + ((pos % capacity) % n_gestores + 1).astype(str)
    return q


def _route_stats(W: pd.DataFrame, truth: bool) -> pd.DataFrame:
    base = W["_alto_obs"].sum() / max(W["has_truth"].sum(), 1) if truth else np.nan
    rows = []
    for r in ROUTES:
        s = W[W["ruta"] == r]
        nt = int(s["has_truth"].sum())
        rate = s["_alto_obs"].sum() / nt if (truth and nt) else np.nan
        rows.append({"ruta": r, "n": len(s), "share": len(s) / max(len(W), 1),
                     "monto": s["valor_financiacion"].sum(), "exp_riesgo": s["exposicion_riesgo"].sum(),
                     "alto_rate": rate, "base": base, "lift": rate / base if (truth and base) else np.nan,
                     "alto_pred": (s["y_pred"].astype(str) == "Alto").mean() if len(s) else np.nan})
    return pd.DataFrame(rows)


def _db04_frame(plan: pd.DataFrame, today: date) -> pd.DataFrame:
    s = lambda c: plan[c].astype("string").fillna("").astype(str).to_numpy()  # noqa: E731
    return pd.DataFrame({
        "id_estudiante": s("id_estudiante"), "llave2": s("llave2"), "nombre": s("nombre"), "programa": s("programa"),
        "riesgo_predicho": s("y_pred"), "confianza": plan["proba_pred"].astype(float).round(4).to_numpy(),
        "prioridad": plan["prioridad"].astype(float).round(1).to_numpy(), "ruta": s("ruta"),
        "accion": plan["ruta"].map(lambda r: ACTION_ROUTES[r]["accion"]).to_numpy(),
        "sla": plan["ruta"].map(lambda r: ACTION_ROUTES[r]["sla"]).to_numpy(),
        "valor_financiado": pd.array(plan["valor_financiacion"].round(0).to_numpy(), dtype="Int64"),
        "gestor_asignado": s("gestor"), "fecha_asignacion": today.isoformat(), "estado": "PENDIENTE",
    }, columns=DB04_COLUMNS)


def _db04_json(ex: pd.DataFrame, meta: dict) -> bytes:
    recs = json.loads(ex.to_json(orient="records", force_ascii=False))
    payload = {"esquema": "SAT-cola-gestion", "version": "1.0", "generado": datetime.now().isoformat(timespec="seconds"),
               **meta, "total_registros": len(recs), "campos": DB04_COLUMNS, "registros": recs}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# ============================================================================================
# Página
# ============================================================================================
df_all = get_active_df()
dff = apply_filters(df_all)
cfg = _cfg()
_css()
p = pal()

page_header(
    "Cola de gestión",
    "Convierte las predicciones en una cola priorizada con rutas de cobro preventivo, la dimensiona contra la "
    "capacidad del equipo y la envía al sistema de cartera.",
    eyebrow="Gestión de cartera",
    highlight=f"{fmt_int(int((dff['ruta'] == 'R1').sum()) if len(dff) else 0)} casos para contacto inmediato",
)
filter_chips(active_chips(df_all))

if dff.empty:
    empty_state("La cartera filtrada está vacía", "Ajusta o limpia los filtros de la barra lateral para armar la cola.")
    footer()
    st.stop()

# ---- Controles compactos -------------------------------------------------------------------
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.05, 0.85, 0.85, 2.5], vertical_alignment="bottom", gap="medium")
    with c1:
        st.number_input("Capacidad semanal", min_value=10, max_value=20000, step=10,
                        help="Gestiones que el equipo puede realizar por semana.", **_bind("capacidad"))
    with c2:
        st.number_input("Gestores", min_value=1, max_value=50, step=1, **_bind("gestores"))
    with c3:
        with st.popover("⚙️ Pesos", width="stretch", help="Pesos del índice de prioridad"):
            st.markdown("**Índice de prioridad (0–100)** = riesgo, exposición y mora ponderados "
                        "(los pesos se normalizan a 100 %).")
            st.slider("Riesgo del modelo (clase + confianza)", 0, 100, step=5, format="%d %%", **_bind("w_riesgo"))
            st.slider("Exposición (percentil del monto)", 0, 100, step=5, format="%d %%", **_bind("w_exposicion"))
            st.slider("Mora en Datacrédito", 0, 100, step=5, format="%d %%", **_bind("w_mora"))
            st.button("↺ Restablecer pesos", key="cola_reset_w", on_click=_reset_weights, width="stretch")
    w, custom_w = _weights(cfg)
    capacity = int(max(10, cfg["capacidad"] or 10))
    n_gest = int(max(1, cfg["gestores"] or 1))
    wcol = {"riesgo": "var(--sat-text)", "exposicion": "var(--sat-accent)", "mora": "#3E63DD"}
    wbar = "".join(f"<i style='width:{w[k] * 100:.1f}%;background:{wcol[k]}'></i>" for k in w if w[k] > 0)
    with c4:
        tag = badge("Pesos personalizados", "accent") if custom_w else badge("Pesos por defecto", "neutral")
        st.markdown(
            f"<div class='cola-mini' style='margin:0 0 6px 0'>{tag}<span class='cola-wbar'>{wbar}</span><br>"
            f"Riesgo <b>{fmt_pct(w['riesgo'], 0)}</b> · Exposición <b>{fmt_pct(w['exposicion'], 0)}</b> · "
            f"Mora <b>{fmt_pct(w['mora'], 0)}</b><br>Cola de gestores: <b>R1 + R2</b> · "
            f"≈ <b>{fmt_num(capacity / n_gest, 1)}</b> gestiones por gestor a la semana</div>",
            unsafe_allow_html=True)

# ---- Cálculo central -----------------------------------------------------------------------
W = _working_frame(df_all, dff, w, custom_w)
truth = bool(W["has_truth"].sum() >= 30 and W["_alto_obs"].sum() > 0)
n_all = len(W)
tot_exp = float(W["exposicion_riesgo"].sum())
tot_alto = int(W["_alto_obs"].sum())
queue = _assign(W[W["ruta"].isin(QUEUE_ROUTES)], capacity, n_gest)
plan = queue[queue["semana"] == 1]
stats = _route_stats(W, truth).set_index("ruta")

# ============================================================================================
# 1 · Rutas de cobro
# ============================================================================================
section("Rutas de cobro preventivo", "Cada crédito recibe una ruta según riesgo predicho, confianza y mora. Si la "
        "priorización funciona, el riesgo observado se concentra en R1.", "1 · Rutas")
max_rate = float(np.nanmax(stats["alto_rate"])) if truth and stats["alto_rate"].notna().any() else 0.0
scale = max(0.4, max_rate * 1.15)
cards = []
for r in ROUTES:
    s, a = stats.loc[r], ACTION_ROUTES[r]
    if truth and s["n"] > 0 and pd.notna(s["alto_rate"]):
        obs = (f"<div class='obs'><div class='row'><span>Alto observado</span><span class='lift'>"
               f"{fmt_num(s['lift'], 1)}× la base</span></div><div class='row'><b>{fmt_pct(s['alto_rate'])}</b>"
               f"<span>base {fmt_pct(s['base'])}</span></div><div class='meter'>"
               f"<i style='width:{min(s['alto_rate'] / scale, 1) * 100:.1f}%'></i>"
               f"<u style='left:{min(s['base'] / scale, 1) * 100:.1f}%'></u></div>"
               f"<div class='legend'>▮ tasa de la ruta · │ tasa base</div></div>")
    else:
        obs = (f"<div class='obs'><div class='row'><span>Alto predicho</span><b>{fmt_pct(s['alto_pred'])}</b></div>"
               f"<div class='legend'>Sin riesgo observado (y_true) en este dataset.</div></div>")
    cards.append(
        f"<div class='cola-route' style='--rc:{_rc(r)};--ri:{_ri(r)}'>"
        f"<div class='top'><span class='code'>{r}</span><span class='sla'>SLA · {esc(a['sla'])}</span></div>"
        f"<div class='name' title='{esc(a['nombre'])}'>{esc(a['nombre'])}</div>"
        f"<div class='big'>{fmt_int(s['n'])}<small>créditos · {fmt_pct(s['share'])}</small></div>"
        f"<div class='money'>Financiado <b>{_nb(fmt_cop(s['monto']))}</b> · en riesgo "
        f"<b>{_nb(fmt_cop(s['exp_riesgo']))}</b></div>{obs}</div>")
_grid(cards, min_px=225)

# ============================================================================================
# 2 · Curva de cobertura por capacidad
# ============================================================================================
section("¿Cuánto riesgo cubre la capacidad del equipo?",
        "Si se gestionan primero los créditos de mayor prioridad, ¿qué parte del riesgo queda cubierta con N gestiones, "
        "frente a contactar al azar? La distancia entre las curvas es el valor del modelo.", "2 · Cobertura")
exp_arr = W["exposicion_riesgo"].to_numpy(float)
cum_exp = np.cumsum(exp_arr) / tot_exp if tot_exp else np.zeros(n_all)
cum_alto = np.cumsum(W["_alto_obs"].to_numpy(int)) / tot_alto if tot_alto else np.zeros(n_all)
k_cap = min(capacity, n_all)
cap_exp = float(cum_exp[k_cap - 1])
cap_alto = float(cum_alto[k_cap - 1]) if truth else np.nan
rand = k_cap / n_all

cc1, cc2 = st.columns([2.35, 1], gap="medium")
with cc1:
    with st.container(border=True):
        xmax = int(min(max(capacity * 12, 10), n_all))
        idx = np.unique(np.concatenate([np.linspace(0, xmax - 1, min(xmax, 400)).astype(int), [k_cap - 1]]))
        xs = np.concatenate([[0], idx + 1])
        fig = go.Figure()
        if truth:
            fig.add_trace(go.Scatter(x=xs, y=np.concatenate([[0], cum_alto[idx]]) * 100, mode="lines",
                                     name="Alto observados capturados", line=dict(color=RISK_COLORS["Alto"], width=3),
                                     hovertemplate="Alto observados: %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=np.concatenate([[0], cum_exp[idx]]) * 100, mode="lines",
                                 name="Exposición en riesgo capturada", line=dict(color=p["text"], width=2.5),
                                 hovertemplate="Exposición en riesgo: %{y:.1f} %<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=xs / n_all * 100, mode="lines", name="Orden aleatorio",
                                 line=dict(color=p["subtle"], width=1.5, dash="dot"),
                                 hovertemplate="Orden aleatorio: %{y:.1f} %<extra></extra>"))
        for kk in range(2, int(xmax // capacity) + 1):
            fig.add_vline(x=kk * capacity, line=dict(color=p["grid"], width=1))
        fig.add_vline(x=k_cap, line=dict(color="#FFD100", width=2.5))
        fig.add_annotation(x=k_cap, y=1.0, yref="paper", text="<b>Semana 1</b>", showarrow=False, yanchor="bottom",
                           font=dict(size=11, color=p["text"]))
        lab = dict(showarrow=False, xanchor="left", xshift=9, borderpad=2, bgcolor=p["surface"])
        fig.add_annotation(x=k_cap, y=cap_exp * 100, text=f"<b>{fmt_pct(cap_exp)}</b>", yshift=10,
                           font=dict(color=p["text"], size=12), **lab)
        if truth:
            fig.add_annotation(x=k_cap, y=cap_alto * 100, text=f"<b>{fmt_pct(cap_alto)}</b>", yshift=-10,
                               font=dict(color=_ri("R1"), size=12), **lab)
        fig.update_layout(hovermode="x unified", margin=dict(l=8, r=16, t=48, b=8), legend=dict(y=1.12),
                          xaxis=dict(title="Gestiones realizadas (créditos en orden de prioridad)",
                                     range=[0, xmax * 1.02], tickformat=",d", hoverformat=",d"),
                          yaxis=dict(title="% capturado del total", ticksuffix=" %", range=[0, 104]))
        show_fig(fig, key="cola_curva", height=400)
        st.markdown("<div class='cola-mini' style='margin:0'>Líneas verticales: semanas de gestión a la capacidad actual "
                    "(la amarilla es la primera); horizonte de 12 semanas.</div>", unsafe_allow_html=True)

with cc2:
    blocks = [f"<div class='cola-stat hl'><div class='k'>Semana 1 · {fmt_int(k_cap)} gestiones</div>"
              f"<div class='v'>{fmt_pct(rand)}</div><div class='s'>de la cartera filtrada "
              f"({fmt_int(n_all)} créditos)</div></div>"]
    if truth:
        k50 = int(np.searchsorted(cum_alto, 0.5) + 1)
        blocks.append(f"<div class='cola-stat'><div class='k'>Alto observados capturados</div>"
                      f"<div class='v'>{fmt_pct(cap_alto)}</div><div class='s'><b>{fmt_num(cap_alto / rand, 1)}×</b> un "
                      f"orden aleatorio · {fmt_int(round(cap_alto * tot_alto))} de {fmt_int(tot_alto)}</div></div>")
        blocks.append(f"<div class='cola-stat'><div class='k'>Para capturar la mitad de los Alto</div>"
                      f"<div class='v'>{fmt_int(k50)}</div><div class='s'>gestiones ≈ <b>{fmt_num(k50 / capacity, 1)} "
                      f"semanas</b> a la capacidad actual</div></div>")
    blocks.append(f"<div class='cola-stat'><div class='k'>Exposición en riesgo capturada</div>"
                  f"<div class='v'>{fmt_pct(cap_exp)}</div><div class='s'><b>{fmt_num(cap_exp / rand, 1)}×</b> un orden "
                  f"aleatorio · {_nb(fmt_cop(cap_exp * tot_exp))}</div></div>")
    _grid(blocks, min_px=200)

# ============================================================================================
# 3 · Cola priorizada
# ============================================================================================
section("Cola priorizada de gestión", "Casos R1 y R2 ordenados por índice de prioridad y asignados a gestores. "
        "Selecciona una fila para ver el detalle del estudiante.", "3 · Cola")


def _detail_card(rec: pd.Series) -> None:
    r = str(rec["ruta"])
    a = ACTION_ROUTES[r]
    conf = fmt_pct(rec["proba_pred"], 0) if pd.notna(rec["proba_pred"]) else "—"
    st.markdown(
        f"<div class='cola-case' style='--rc:{_rc(r)}'><div class='hd'><div><div class='nm'>{esc(rec['nombre'])}</div>"
        f"<div class='meta'>{esc(rec['programa'])} · crédito {esc(rec['llave2'])}</div></div>"
        f"<div class='chips'>{risk_badge(str(rec['y_pred']))}<span class='sat-badge' style='color:{_ri(r)};"
        f"background:{_rgba(_rc(r), .14)};border-color:{_rgba(_rc(r), .4)}'>{r} · {esc(a['nombre'])}</span></div></div>"
        f"<div class='g3'><div><div class='k'>Confianza del modelo</div><div class='v'>{conf}</div></div>"
        f"<div><div class='k'>Prioridad</div><div class='v'>{fmt_num(rec['prioridad'], 1)} / 100</div></div>"
        f"<div><div class='k'>Monto financiado</div><div class='v'>{fmt_cop(rec['valor_financiacion'], compact=False)}"
        f"</div></div></div><div class='act'><b>Acción ({esc(a['sla'])}):</b> {esc(a['accion'])} · "
        f"<b>{esc(rec['gestor'])}</b>, semana {int(rec['semana'])}</div></div>",
        unsafe_allow_html=True)


@st.fragment
def _queue_block(queue: pd.DataFrame) -> None:
    if queue.empty:
        empty_state("La cola de gestores está vacía", "No hay créditos R1 ni R2 con los filtros actuales.", icon="📭")
        return
    v1, v2 = st.columns([1.3, 2.4], vertical_alignment="center")
    with v1:
        st.segmented_control("Vista de la cola", ["Plan de la semana", "Cola completa"], selection_mode="single",
                             width="stretch", label_visibility="collapsed", **_bind("vista"))
    vista = _cfg()["vista"] or "Plan de la semana"
    view = (queue[queue["semana"] == 1] if vista == "Plan de la semana" else queue).reset_index(drop=True)
    with v2:
        st.markdown(f"<div class='cola-mini' style='text-align:right;margin:0'><b>{fmt_int(len(view))}</b> casos · "
                    f"<b>{_nb(fmt_cop(view['valor_financiacion'].sum()))}</b> financiados · "
                    f"{_nb(fmt_cop(view['exposicion_riesgo'].sum()))} en riesgo</div>", unsafe_allow_html=True)
    tbl = pd.DataFrame({
        "#": view["_pos"] + 1,
        "Prioridad": view["prioridad"].astype(float),
        "Riesgo": view["y_pred"].astype(str).map(lambda x: f"{RISK_ICONS.get(x, '⚪')} {x}"),
        "Confianza": (view["proba_pred"] * 100).round(0),
        "Estudiante": view["nombre"],
        "Ruta": view["ruta"].map(lambda r: f"{ROUTE_EMOJI[r]} {r} · {ROUTE_SHORT[r]}"),
        "Gestor": view["gestor"],
        "Semana": "S" + view["semana"].astype(str),
        "Financiado": view["valor_financiacion"].round(0).astype("int64"),
        "Programa": view["programa"],
    })
    ev = st.dataframe(
        tbl, key=f"cola_tabla_{vista}_{len(view)}", hide_index=True, on_select="rerun", selection_mode="single-row",
        height=390,
        column_config={
            "#": st.column_config.NumberColumn("#", format="%d", width=48),
            "Prioridad": st.column_config.ProgressColumn("Prioridad", min_value=0, max_value=100, format="%.0f",
                                                         width=105),
            "Riesgo": st.column_config.TextColumn("Riesgo", width=82),
            "Confianza": st.column_config.NumberColumn("Confianza", format="%d %%", width=88,
                                                       help="Probabilidad de la clase predicha"),
            "Estudiante": st.column_config.TextColumn("Estudiante", width=190, help="Nombre anonimizado"),
            "Ruta": st.column_config.TextColumn("Ruta", width=130),
            "Gestor": st.column_config.TextColumn("Gestor", width=82),
            "Semana": st.column_config.TextColumn("Semana", width=70),
            "Financiado": st.column_config.NumberColumn("Financiado (COP)", format="localized", width=125),
            "Programa": st.column_config.TextColumn("Programa", width=260),
        },
    )
    rows = list(getattr(getattr(ev, "selection", None), "rows", []) or [])
    if rows:
        _detail_card(view.iloc[int(rows[0])])
    else:
        st.markdown("<div class='cola-mini'>☝️ Selecciona una fila (casilla de la izquierda) para ver el detalle del "
                    "estudiante.</div>", unsafe_allow_html=True)


_queue_block(queue)

# ============================================================================================
# 4 · Exportar (DB-02) e intercambio con cartera (DB-04)
# ============================================================================================
section("Exportar y enviar al sistema de cartera", "DB-02: descarga de la cola priorizada · DB-04: archivo de "
        "intercambio con esquema fijo para el sistema de gestión de cartera.", "4 · Exportar")
today = date.today()
stamp = today.strftime("%Y%m%d")
e1, e2 = st.columns(2, gap="medium")
with e1:
    with st.container(border=True):
        st.markdown(f"{badge('DB-02', 'accent')} &nbsp;**Cola priorizada completa**", unsafe_allow_html=True)
        st.markdown(f"<div class='cola-mini'>{fmt_int(len(queue))} casos con posición, semana, gestor, ruta, SLA y acción. "
                    f"El Excel agrega el resumen de rutas y los parámetros de la corrida.</div>", unsafe_allow_html=True)
        exp_df = pd.DataFrame({
            "posicion": queue["_pos"] + 1, "semana": queue["semana"], "gestor": queue["gestor"],
            "prioridad": queue["prioridad"].round(1), "ruta": queue["ruta"],
            "sla": queue["ruta"].map(lambda r: ACTION_ROUTES[r]["sla"]),
            "accion": queue["ruta"].map(lambda r: ACTION_ROUTES[r]["accion"]),
            "riesgo_predicho": queue["y_pred"].astype(str), "confianza": queue["proba_pred"].round(4),
            "id_estudiante": queue["id_estudiante"].astype(str), "llave2": queue["llave2"], "nombre": queue["nombre"],
            "programa": queue["programa"], "valor_financiado": queue["valor_financiacion"].round(0),
            "exposicion_riesgo": queue["exposicion_riesgo"].round(0),
        })
        rutas_df = stats.reset_index()[["ruta", "n", "share", "monto", "exp_riesgo", "alto_rate", "base", "lift"]]
        params_df = pd.DataFrame({
            "parametro": ["fecha", "creditos_filtrados", "rutas_en_cola", "capacidad_semanal", "gestores",
                          "peso_riesgo", "peso_exposicion", "peso_mora"],
            "valor": [today.isoformat(), n_all, "R1, R2", capacity, n_gest, round(w["riesgo"], 4),
                      round(w["exposicion"], 4), round(w["mora"], 4)]}).astype(str)
        b1, b2 = st.columns(2)
        b1.download_button("CSV", lambda: exp_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"cola_gestion_{stamp}.csv", mime="text/csv", key="cola_dl_csv", width="stretch",
                           type="primary", icon=":material/download:", on_click="ignore")
        b2.download_button("Excel", lambda: to_excel_bytes({"cola": exp_df, "rutas": rutas_df, "parametros": params_df}),
                           file_name=f"cola_gestion_{stamp}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="cola_dl_xlsx", width="stretch", icon=":material/table_view:", on_click="ignore")
with e2:
    with st.container(border=True):
        st.markdown(f"{badge('DB-04', 'accent')} &nbsp;**Archivo de intercambio · plan de la semana**",
                    unsafe_allow_html=True)
        if plan.empty:
            st.markdown("<div class='cola-mini'>Sin casos asignados esta semana: el archivo se genera cuando la cola "
                        "tiene casos.</div>", unsafe_allow_html=True)
        else:
            ex = _db04_frame(plan, today)
            st.markdown(f"<div class='cola-mini'>{fmt_int(len(ex))} registros · {len(DB04_COLUMNS)} campos en orden "
                        f"fijo · CSV con «;» (UTF-8 con BOM) y JSON con metadatos. Llave de integración: "
                        f"<b>llave2</b>.</div>", unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            d1.download_button("CSV ; (intercambio)", ex.to_csv(sep=";", index=False).encode("utf-8-sig"),
                               file_name=f"sat_intercambio_cartera_{stamp}.csv", mime="text/csv", key="cola_db04_csv",
                               width="stretch", type="primary", icon=":material/sync_alt:", on_click="ignore")
            d2.download_button("JSON (intercambio)",
                               _db04_json(ex, {"fecha_asignacion": today.isoformat(), "capacidad_semanal": capacity,
                                               "gestores": n_gest, "rutas": QUEUE_ROUTES}),
                               file_name=f"sat_intercambio_cartera_{stamp}.json", mime="application/json",
                               key="cola_db04_json", width="stretch", icon=":material/data_object:", on_click="ignore")

with st.expander("📘 DB-04 · Esquema del archivo de intercambio (14 campos, orden fijo)"):
    dic = pd.DataFrame(DB04_SCHEMA, columns=["Campo", "Tipo", "Obligatorio", "Descripción"])
    st.dataframe(dic, hide_index=True, key="cola_db04_dic", height=38 + 35 * len(dic),
                 column_config={"Campo": st.column_config.TextColumn(width=140),
                                "Tipo": st.column_config.TextColumn(width=190),
                                "Obligatorio": st.column_config.TextColumn(width=90),
                                "Descripción": st.column_config.TextColumn(width=380)})
    st.caption("`estado` inicia en PENDIENTE y lo actualiza el sistema de cartera (p. ej. CONTACTADO, ACUERDO).")

# ============================================================================================
# 5 · Escenario ilustrativo: recaudo y reparto por gestor
# ============================================================================================
with st.expander("💵 Escenario ilustrativo · impacto en recaudo y reparto por gestor"):
    st.markdown(
        "<div class='cola-banner'><span class='tag'>Escenario ilustrativo</span><div>La base no contiene datos de "
        "recaudo: esto <b>no es una predicción</b>, sino una simulación con supuestos editables frente a la línea base "
        f"institucional de {fmt_pct(RECAUDO_BASELINE, 0)} (según el reporte).</div></div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4, gap="medium")
    s1.slider("Horizonte (semanas)", 1, 26, step=1, **_bind("sc_semanas"))
    s2.slider("Contactabilidad", 0, 100, step=5, format="%d %%", **_bind("sc_contacto"))
    s3.slider("Efectividad R1", 0, 100, step=5, format="%d %%", **_bind("sc_ef_R1"))
    s4.slider("Efectividad R2", 0, 100, step=5, format="%d %%", **_bind("sc_ef_R2"))
    H = int(cfg["sc_semanas"])
    ef = {"R1": cfg["sc_ef_R1"] / 100, "R2": cfg["sc_ef_R2"] / 100}
    managed = queue.head(capacity * H)
    gain = float((managed["exposicion_riesgo"] * (cfg["sc_contacto"] / 100) * managed["ruta"].map(ef).fillna(0)).sum())
    V = float(W["valor_financiacion"].sum())
    rec_proj = RECAUDO_BASELINE + (min(gain / V, 1 - RECAUDO_BASELINE) if V > 0 else 0.0)
    _grid([
        kpi_card("Recaudo proyectado", fmt_pct(rec_proj), f"{fmt_delta_pp(rec_proj - RECAUDO_BASELINE)} sobre la línea "
                 f"base de {fmt_pct(RECAUDO_BASELINE, 0)}", tone="bajo", icon="📈", bar=rec_proj),
        kpi_card("Recaudo adicional estimado", fmt_cop(gain), f"en {H} semana{'s' if H != 1 else ''} de gestión",
                 tone="accent", icon="💵"),
        kpi_card("Casos gestionados", fmt_int(len(managed)), f"de {fmt_int(len(queue))} en la cola R1 + R2",
                 tone="ink", icon="📞"),
    ], min_px=200)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.caption("Fórmula: recaudo = 85 % + Σ (monto × intensidad de riesgo × contactabilidad × efectividad de la ruta) / "
               "valor financiado de la cartera filtrada, sobre los primeros capacidad × semanas casos de la cola "
               "(tope 100 %).")
    if not plan.empty:
        gcols = [f"Gestor {i}" for i in range(1, n_gest + 1)]
        load = plan.groupby(["gestor", "ruta"]).size().unstack(fill_value=0).reindex(index=gcols, fill_value=0)
        expg = plan.groupby("gestor")["exposicion_riesgo"].sum().reindex(gcols, fill_value=0.0)
        st.markdown("<div class='cola-mini' style='margin-top:8px'><b>Reparto de la semana por gestor</b> (round-robin) · "
                    "casos por ruta; la línea punteada es la capacidad individual.</div>", unsafe_allow_html=True)
        figl = go.Figure()
        for r in QUEUE_ROUTES:
            if r in load.columns and load[r].sum() > 0:
                figl.add_trace(go.Bar(x=load.index, y=load[r], name=f"{r} · {ROUTE_SHORT[r]}",
                                      marker=dict(color=_rc(r), line=dict(color=p["surface"], width=1.5)),
                                      customdata=expg.map(fmt_cop).to_numpy(),
                                      hovertemplate="%{x}<br>" + r + ": %{y:,d} casos<br>En riesgo del gestor: "
                                                    "%{customdata}<extra></extra>"))
        tot = load.sum(axis=1)
        figl.add_trace(go.Scatter(x=load.index, y=tot, mode="text", text=[fmt_int(v) for v in tot],
                                  textposition="top center", textfont=dict(color=p["text"], size=12),
                                  showlegend=False, hoverinfo="skip", cliponaxis=False))
        figl.add_hline(y=capacity / n_gest, line=dict(color=p["muted"], width=1.5, dash="dash"))
        figl.update_layout(barmode="stack", bargap=0.45, margin=dict(l=8, r=8, t=36, b=8),
                           yaxis=dict(title="Casos asignados", range=[0, max(float(tot.max()), capacity / n_gest) * 1.25]),
                           xaxis=dict(title=None, tickangle=0 if n_gest <= 8 else -45))
        show_fig(figl, key="cola_carga", height=280)

footer()
