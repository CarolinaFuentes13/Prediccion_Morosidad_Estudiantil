"""Segmentos y territorio — ¿qué grupos y qué regiones concentran el riesgo de morosidad?

Materializa el *enfoque descriptivo* del reporte en tres pestañas:

* **Explorador de segmentos**: ranking por dimensión con IC de Wilson 95 %, promedio y lift.
* **Perfil de riesgo**: gradiente por rango de scoring externo y cruce tipo de interés × cuotas.
* **Territorio**: mapa por ciudad, top de departamentos y Bogotá frente al resto del país.

Todas las cifras respetan los filtros globales de la barra lateral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.components import (download_bar, empty_state, esc, filter_chips, footer, insight, insight_row, kpi_card,
                             kpi_row, page_header, section)
from core.config import RISK_COLORS, SCORE_ORDER, SCORE_SHORT
from core.data import get_active_df, nrm
from core.filters import active_chips, apply_filters
from core.theme import SEQ_RISK, fmt_int, fmt_num, fmt_pct, pal, show_fig, theme_mode

P = "segmentos"
Z95 = 1.959964

DIMS = {
    "programa_cluster": "Segmento de programa", "facultad": "Facultad", "nivel": "Nivel académico",
    "tipo_interes": "Tipo de interés", "cuotas": "Número de cuotas", "cohorte": "Cohorte",
    "tipo_estudiante": "Tipo de estudiante", "score_corto": "Scoring externo", "rango_edad": "Rango de edad",
    "sede": "Sede",
}
ORDINAL = {"cuotas", "cohorte", "score_corto", "rango_edad"}
_SCORE_LABELS = [SCORE_SHORT[s] for s in SCORE_ORDER]  # peor → mejor
_AGE_ORDER = ["≤ 24", "25–29", "30–34", "35–44", "45 +"]
METRICS = {  # nombre → (columna de la marca 0/1, requiere y_true)
    "% Alto predicho": ("_ap", False),
    "% Alto observado": ("_ao", True),
    "% mora": ("_m", False),
}

_CSS = """
<style>
.sg-note{font-size:12.5px;color:var(--sat-muted);line-height:1.5;margin:4px 0 2px 0}
.sg-note b{color:var(--sat-text)}
.sg-legend{display:flex;gap:6px 16px;flex-wrap:wrap;font-size:12px;color:var(--sat-muted);margin:0 0 2px 0}
.sg-legend span{display:inline-flex;align-items:center;gap:6px}
.sg-legend i{display:inline-block;width:11px;height:11px;border-radius:3px;background:var(--c)}
.sg-legend i.ln{height:3px;width:16px;border-radius:2px}
div[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--sat-border);flex-wrap:wrap}
div[data-testid="stTabs"] button[role="tab"]{padding:8px 14px;border-radius:10px 10px 0 0}
div[data-testid="stTabs"] button[role="tab"] p{font-size:15px;font-weight:600}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{background:var(--sat-accent-soft)}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background-color:var(--sat-accent) !important;height:3px}
</style>
"""


# ======================================================================================
# Utilidades
# ======================================================================================
def _wilson(k, n, z: float = Z95):
    """Proporción e intervalo de Wilson (vectorizado). Devuelve (p, lo, hi)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        den = 1 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / den
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def _x(v, d: int = 1) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{fmt_num(v, d)}×"


def _short(s, n: int = 32) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _note(html_text: str) -> None:
    st.markdown(f"<div class='sg-note'>{html_text}</div>", unsafe_allow_html=True)


def _legend(items: list[tuple[str, str, str]]) -> None:
    st.markdown("<div class='sg-legend'>" + "".join(
        f"<span><i class='{shape}' style='--c:{c}'></i>{esc(t)}</span>" for c, t, shape in items) + "</div>",
        unsafe_allow_html=True)


def _labels(d: pd.DataFrame, col: str) -> pd.Series:
    """Etiquetas legibles (str) de una dimensión, con «Sin dato» para faltantes."""
    if col == "cuotas":
        v = pd.to_numeric(d[col], errors="coerce")
        return v.map(lambda x: f"{int(x)} cuotas" if pd.notna(x) else "Sin dato").astype(str)
    out = d[col].astype(str).str.strip()
    out = out.where(~out.str.lower().isin({"nan", "none", "<na>", "", "nat"}), "Sin dato")
    if col == "nivel":
        out = out.map(lambda s: s if s == "Sin dato" else s[:1] + s[1:].lower())
    return out


def _order_key(col: str):
    if col == "score_corto":
        return lambda x: _SCORE_LABELS.index(x) if x in _SCORE_LABELS else 99
    if col == "rango_edad":
        return lambda x: _AGE_ORDER.index(x) if x in _AGE_ORDER else 99
    if col in ("cuotas", "cohorte"):
        return lambda x: int("".join(ch for ch in x if ch.isdigit()) or 999)
    return lambda x: nrm(x)


def _flags(d: pd.DataFrame) -> pd.DataFrame:
    """Marcas 0/1 por crédito: Alto predicho, Alto observado (NaN sin verdad) y mora."""
    t = d["has_truth"].to_numpy(dtype=bool)
    return pd.DataFrame({
        "_ap": (d["y_pred"].astype(str) == "Alto").to_numpy(dtype=float),
        "_ao": np.where(t, (d["y_true"].astype(str) == "Alto").to_numpy(), np.nan),
        "_m": pd.to_numeric(d["mora_flag"], errors="coerce").fillna(0).to_numpy(dtype=float),
    }, index=d.index)


def _rate_table(d: pd.DataFrame, keys: pd.Series | list, flag: str) -> pd.DataFrame:
    """Por grupo: créditos, % de la marca ``flag`` con IC de Wilson y % de las demás métricas."""
    f = _flags(d)
    if not isinstance(keys, list):
        keys = pd.Series(np.asarray(keys), index=d.index, name="seg")
    g = f.groupby(keys, observed=True, sort=False).agg(
        n=("_ap", "size"), k=(flag, "sum"), nv=(flag, "count"), pct_pred=("_ap", "mean"), pct_obs=("_ao", "mean"),
        pct_mora=("_m", "mean")).reset_index()
    g["val"], g["lo"], g["hi"] = _wilson(g["k"], g["nv"])
    return g


def _rate(d: pd.DataFrame, flag: str) -> float:
    s = _flags(d)[flag]
    return float(s.mean()) if s.notna().any() else float("nan")


# ======================================================================================
# Página
# ======================================================================================
st.markdown(_CSS, unsafe_allow_html=True)
df_all = get_active_df()
dff = apply_filters(df_all)
has_truth = bool(dff["has_truth"].any()) if len(dff) else False

page_header(
    "Segmentos y territorio",
    "Qué programas, condiciones del crédito, perfiles de scoring y regiones concentran el riesgo Alto: "
    "criterios para focalizar la gestión preventiva de cartera.",
    eyebrow="Análisis de riesgo · enfoque descriptivo",
    highlight=f"{fmt_int(len(dff))} créditos analizados" if len(dff) else None,
    meta=["IC de Wilson 95 %"],
)
filter_chips(active_chips(df_all))
if dff.empty:
    empty_state("Sin créditos para analizar", "Los filtros actuales no dejan créditos. Ajusta o limpia los filtros "
                "de la barra lateral.")
    footer()
    st.stop()

AVG_PRED, AVG_OBS, AVG_MORA = _rate(dff, "_ap"), _rate(dff, "_ao"), _rate(dff, "_m")
BASIS, BASIS_TXT = ("_ao", "observado") if has_truth else ("_ap", "predicho")
AVG_BASIS = AVG_OBS if has_truth else AVG_PRED

# ---------------------------------------------------------------- KPI
_n_est = dff["id_estudiante"].nunique()
kpi_row([
    kpi_card("Créditos analizados", fmt_int(len(dff)), f"{fmt_int(_n_est)} estudiantes · "
             f"{dff['departamento_limpio'].nunique()} departamentos", tone="ink", icon="📄"),
    kpi_card("Riesgo Alto predicho", fmt_pct(AVG_PRED), f"{fmt_int(round(AVG_PRED * len(dff)))} créditos marcados "
             "por el modelo", tone="alto", icon="🔴", bar=min(AVG_PRED / 0.25, 1)),
    kpi_card("Riesgo Alto observado", fmt_pct(AVG_OBS) if has_truth else "—",
             "Riesgo real registrado en la base" if has_truth else "El dataset activo no trae y_true",
             tone="medio", icon="🎯"),
    kpi_card("Mora Datacrédito", fmt_pct(AVG_MORA), "Créditos con mora reportada en el scoring externo",
             tone="accent", icon="📉"),
])


# ---------------------------------------------------------------- Hallazgos
def _top_segment(col: str, min_n: int = 50, only: list[str] | None = None) -> pd.Series | None:
    t = _rate_table(dff, _labels(dff, col).to_numpy(), BASIS)
    t = t[(t["nv"] >= min_n) & (t["seg"] != "Sin dato")]
    if only is not None:
        t = t[t["seg"].isin(only)]
    if len(t) < 2 or not np.isfinite(AVG_BASIS) or AVG_BASIS <= 0:
        return None
    r = t.sort_values("val", ascending=False).iloc[0].copy()
    r["lift"] = r["val"] / AVG_BASIS
    r["best"] = t.sort_values("val").iloc[0]
    return r


def _findings() -> list[str]:
    cards = []
    specs = [("programa_cluster", "Programa con más riesgo", "🎓", None),
             ("tipo_interes", "Condición del crédito", "💳", None),
             ("score_corto", "Scoring externo", "📊", _SCORE_LABELS[1:])]
    for col, title, icon, only in specs:
        r = _top_segment(col, only=only)
        if r is None:
            continue
        b = r["best"]
        sig = "significativo" if r["lo"] > AVG_BASIS else "no concluyente"
        cards.append(insight(
            title,
            f"<b>{esc(r['seg'])}</b>: <b>{fmt_pct(r['val'])}</b> de Alto {BASIS_TXT}, <b>{_x(r['lift'])}</b> el "
            f"promedio ({fmt_pct(AVG_BASIS)}; IC 95 % {sig}, n = {fmt_int(r['nv'])}).<br>"
            f"Menor valor: {esc(b['seg'])} con {fmt_pct(b['val'])}.",
            tone="alto" if r["lo"] > AVG_BASIS else "medio", icon=icon))
    return cards


section("Hallazgos clave", f"Se recalculan con cada filtro (riesgo Alto {BASIS_TXT}, segmentos con ≥ 50 créditos).",
        kicker="Lectura rápida")
_cards = _findings()
if _cards:
    insight_row(_cards)


# ======================================================================================
# Pestaña 1 · Explorador de segmentos
# ======================================================================================
def _rank_fig(t: pd.DataFrame, mname: str, ref: float) -> go.Figure:
    p = pal()
    t = t.iloc[::-1].reset_index(drop=True)
    sig = np.where(t["lo"] > ref, "sobre", np.where(t["hi"] < ref, "bajo", "nc"))
    cmap = {"sobre": RISK_COLORS["Alto"], "bajo": RISK_COLORS["Bajo"], "nc": p["subtle"]}
    lift = t["val"] / ref if ref > 0 else np.full(len(t), np.nan)
    cd = np.c_[[esc(s) for s in t["seg"]], [fmt_int(v) for v in t["n"]], [fmt_pct(v) for v in t["val"]],
               [f"{fmt_pct(a)} – {fmt_pct(b)}" for a, b in zip(t["lo"], t["hi"])], [_x(v, 2) for v in lift],
               [{"sobre": "sobre el promedio", "bajo": "bajo el promedio", "nc": "no concluyente"}[s] for s in sig]]
    labels = [_short(s) for s in t["seg"]]
    fig = go.Figure(go.Bar(
        y=labels, x=t["val"], orientation="h", marker=dict(color=[cmap[s] for s in sig], line=dict(width=0)),
        customdata=cd,
        error_x=dict(type="data", symmetric=False, array=np.nan_to_num(t["hi"] - t["val"]),
                     arrayminus=np.nan_to_num(t["val"] - t["lo"]), color=p["muted"], thickness=1.3, width=4),
        hovertemplate=(f"<b>%{{customdata[0]}}</b><br>{esc(mname)}: <b>%{{customdata[2]}}</b>"
                       "<br>IC 95 %: %{customdata[3]}<br>Lift: %{customdata[4]} · %{customdata[5]}"
                       "<br>Créditos: %{customdata[1]}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(x=t["hi"], y=labels, mode="text", text=[f"<b>{fmt_pct(v)}</b> · {_x(l)}"
                                                                      for v, l in zip(t["val"], lift)],
                             textposition="middle right", textfont=dict(size=11.5, color=p["text"]),
                             hoverinfo="skip", cliponaxis=False, showlegend=False))
    fig.add_vline(x=ref, line=dict(color=p["text"], width=1.2, dash="dot"), layer="below")
    fig.add_annotation(x=ref, y=1, yref="paper", text=f"Promedio {fmt_pct(ref)}", showarrow=False, xanchor="left",
                       yanchor="bottom", xshift=4, font=dict(size=11, color=p["muted"]))
    fig.update_xaxes(range=[0, max(float(np.nanmax(t["hi"])) * 1.3, 0.01)], tickformat=".0%", title_text=mname,
                     gridcolor=p["grid"])
    fig.update_yaxes(showgrid=False, ticksuffix="  ", tickfont=dict(size=12, color=p["text"]))
    fig.update_layout(bargap=0.35, barcornerradius=4, margin=dict(l=8, r=16, t=28, b=8), showlegend=False)
    return fig


def _tab_explorer() -> None:
    section("Ranking de segmentos", "Elige una dimensión y una métrica. Rojo = supera el promedio con significancia "
            "(todo su IC 95 % por encima); verde = por debajo; gris = diferencia no concluyente.", kicker="Explorador")
    opts = [m for m, (_, truth) in METRICS.items() if has_truth or not truth]
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.3, 1.9, 1.1], vertical_alignment="bottom")
        dim = c1.selectbox("Dimensión", list(DIMS), format_func=DIMS.get, key=f"{P}_dim")
        mname = c2.segmented_control("Métrica", opts, default=opts[0], key=f"{P}_metric", required=True) or opts[0]
        nmin = c3.select_slider("n mínimo por segmento", [10, 30, 50, 100, 200], value=30, key=f"{P}_nmin")
        flag = METRICS[mname][0]
        ref = _rate(dff, flag)
        tab = _rate_table(dff, _labels(dff, dim).to_numpy(), flag)
        hidden = int((tab["nv"] < nmin).sum())
        tab = tab[tab["nv"] >= nmin].copy()
        if tab.empty or not np.isfinite(ref):
            empty_state("Ningún segmento cumple el n mínimo", "Reduce el n mínimo o amplía los filtros.", icon="📏")
            return
        if dim in ORDINAL:
            tab = tab.assign(_k=tab["seg"].map(_order_key(dim))).sort_values("_k").drop(columns="_k")
        else:
            tab = tab.sort_values(["val", "n"], ascending=False)
        tab = tab.head(20)
        _legend([(RISK_COLORS["Alto"], "Sobre el promedio (significativo)", "sq"), (pal()["subtle"], "No concluyente", "sq"),
                 (RISK_COLORS["Bajo"], "Bajo el promedio (significativo)", "sq"), (pal()["muted"], "IC 95 % de Wilson", "ln")])
        show_fig(_rank_fig(tab, mname, ref), key=f"{P}_rank", legend=False, height=int(np.clip(40 * len(tab) + 90, 280, 820)))
        n_up = int((tab["lo"] > ref).sum())
        _note(f"<b>{n_up}</b> de {len(tab)} segmentos superan el promedio de <b>{fmt_pct(ref)}</b> con significancia. "
              f"Lift = tasa del segmento ÷ promedio de la cartera filtrada."
              + (f" {hidden} segmento(s) con menos de {nmin} créditos quedan fuera." if hidden else "")
              + (" Orden natural de la dimensión." if dim in ORDINAL else ""))

    table = pd.DataFrame({
        DIMS[dim]: tab["seg"], "Créditos": tab["n"].astype(int), mname: tab["val"], "IC inferior": tab["lo"],
        "IC superior": tab["hi"], "Lift": tab["val"] / ref if ref > 0 else np.nan, "% Alto predicho": tab["pct_pred"],
        "% Alto observado": tab["pct_obs"], "% mora": tab["pct_mora"],
    })
    table = table.loc[:, ~table.columns.duplicated()]
    if not has_truth:
        table = table.drop(columns=["% Alto observado"], errors="ignore")
    with st.expander(f"Tabla y descarga · {len(table)} segmentos ({DIMS[dim].lower()})"):
        pct = st.column_config.NumberColumn(format="percent")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            c: pct for c in table.columns if c.startswith("%") or c.startswith("IC")} | {
            "Lift": st.column_config.NumberColumn(format="%.2f×")})
        download_bar(table, f"segmentos_{dim}", key=f"{P}_dl_rank")


# ======================================================================================
# Pestaña 2 · Perfil de riesgo
# ======================================================================================
def _score_fig(t: pd.DataFrame, avg: float, obs: bool) -> go.Figure:
    p = pal()
    vmax = max(float(t["val"].max()), 1e-3)
    colors = px.colors.sample_colorscale(SEQ_RISK, list(np.clip(t["val"] / vmax, 0.15, 1)))
    cd = np.c_[[fmt_pct(v) for v in t["val"]], [f"{fmt_pct(a)} – {fmt_pct(b)}" for a, b in zip(t["lo"], t["hi"])],
               [fmt_int(v) for v in t["n"]]]
    fig = go.Figure(go.Bar(
        x=t["seg"], y=t["val"], marker=dict(color=colors, line=dict(width=0)), customdata=cd,
        error_y=dict(type="data", symmetric=False, array=t["hi"] - t["val"], arrayminus=t["val"] - t["lo"],
                     color=p["muted"], thickness=1.3, width=5),
        text=[f"<b>{fmt_pct(v)}</b>" for v in t["val"]], textposition="inside", insidetextanchor="start",
        textfont=dict(color=["#FFFFFF" if v / vmax > 0.72 else "#141414" for v in t["val"]], size=12.5),
        cliponaxis=False,
        hovertemplate=(f"<b>Scoring %{{x}}</b><br>% Alto {'observado' if obs else 'predicho'}: "
                       "<b>%{customdata[0]}</b><br>IC 95 %: %{customdata[1]}<br>Créditos: %{customdata[2]}"
                       "<extra></extra>"),
    ))
    fig.add_hline(y=avg, line=dict(color=p["text"], width=1.2, dash="dot"),
                  annotation=dict(text=f"Promedio {fmt_pct(avg)}", font=dict(size=11, color=p["muted"])),
                  annotation_position="top right")
    fig.update_xaxes(tickvals=t["seg"], ticktext=[f"{s}<br><span style='font-size:10px'>n = {fmt_int(n)}</span>"
                                                  for s, n in zip(t["seg"], t["n"])],
                     title_text="Peor score  →  mejor score")
    fig.update_yaxes(tickformat=".0%", title_text=f"% Alto {'observado' if obs else 'predicho'}",
                     range=[0, float(t["hi"].max()) * 1.2])
    fig.update_layout(bargap=0.3, barcornerradius=4, showlegend=False, margin=dict(l=8, r=8, t=24, b=8))
    return fig


def _heatmap_fig(d: pd.DataFrame, flag: str, gmin: int, avg: float, mname: str) -> go.Figure:
    p = pal()
    t = _rate_table(d, [_labels(d, "tipo_interes").rename("r"), _labels(d, "cuotas").rename("c")], flag)
    t = t[t["n"] >= 1]
    ctot, rtot = t.groupby("c")["n"].sum(), t.groupby("r")["n"].sum()
    cols = sorted(ctot[ctot >= gmin].index, key=_order_key("cuotas"))
    rows = sorted(rtot[rtot >= gmin].index)
    z = t.pivot(index="r", columns="c", values="val").reindex(index=rows, columns=cols)
    n = t.pivot(index="r", columns="c", values="nv").reindex(index=rows, columns=cols).fillna(0)
    lo = t.pivot(index="r", columns="c", values="lo").reindex(index=rows, columns=cols)
    hi = t.pivot(index="r", columns="c", values="hi").reindex(index=rows, columns=cols)
    zmask = z.where(n >= gmin)
    text = [[(f"<b>{fmt_pct(zmask.iat[i, j])}</b><br><span style='font-size:10px'>n = {fmt_int(n.iat[i, j])}</span>"
              if pd.notna(zmask.iat[i, j]) else (f"n = {fmt_int(n.iat[i, j])}" if n.iat[i, j] else ""))
             for j in range(len(cols))] for i in range(len(rows))]
    cd = np.dstack([[[r] * len(cols) for r in rows], [cols] * len(rows),
                    [[fmt_pct(v) if pd.notna(v) else "n bajo" for v in row] for row in zmask.to_numpy()],
                    [[f"{fmt_pct(a)} – {fmt_pct(b)}" for a, b in zip(ra, rb)] for ra, rb in zip(lo.to_numpy(), hi.to_numpy())],
                    [[fmt_int(v) for v in row] for row in n.to_numpy()]])
    zmax = max(float(np.nanmax(zmask.to_numpy())) if zmask.notna().any().any() else 0, avg * 1.5, 0.01)
    fig = go.Figure(go.Heatmap(
        z=zmask.to_numpy(), x=cols, y=rows, text=text, texttemplate="%{text}", textfont=dict(size=12.5),
        customdata=cd, zmin=0, zmax=zmax, colorscale=SEQ_RISK, xgap=4, ygap=4, hoverongaps=False,
        colorbar=dict(title=dict(text=mname, side="right", font=dict(size=11, color=p["muted"])), tickformat=".0%",
                      thickness=10, len=0.9, outlinewidth=0, tickfont=dict(color=p["muted"], size=10)),
        hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}<br>" + esc(mname) + ": <b>%{customdata[2]}</b>"
                       "<br>IC 95 %: %{customdata[3]}<br>Créditos: %{customdata[4]}<extra></extra>"),
    ))
    fig.update_xaxes(side="top", showgrid=False, tickangle=0, tickfont=dict(size=12, color=p["text"]))
    fig.update_yaxes(showgrid=False, ticksuffix="  ", tickfont=dict(size=12, color=p["text"]))
    fig.update_layout(margin=dict(l=8, r=8, t=36, b=8), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def _tab_profile() -> None:
    opts = ["Predicho", "Observado"] if has_truth else ["Predicho"]
    c1, c2, _ = st.columns([1.3, 1.3, 2], vertical_alignment="bottom")
    basis = c1.segmented_control("Riesgo Alto", opts, default=opts[-1], key=f"{P}_prof_basis", required=True) or opts[-1]
    gmin = c2.select_slider("Ocultar celdas con menos de", [10, 30, 50, 100], value=30, key=f"{P}_prof_min",
                            format_func=lambda v: f"{v} créditos")
    obs = basis == "Observado"
    flag = "_ao" if obs else "_ap"
    avg = AVG_OBS if obs else AVG_PRED
    left, right = st.columns([1.35, 1], gap="medium")
    with left, st.container(border=True):
        section("Riesgo por rango de scoring externo", "Rangos de Datacrédito del peor al mejor. Si el scoring "
                "discrimina, las barras bajan de izquierda a derecha.", kicker="Scoring")
        t = _rate_table(dff, dff["score_corto"].astype(str).to_numpy(), flag)
        t = t[t["seg"].isin(_SCORE_LABELS) & (t["nv"] >= gmin)].copy()
        if len(t) < 2:
            _note("No hay suficientes rangos de scoring con el n mínimo en el filtro actual.")
        else:
            t = t.assign(_k=t["seg"].map(_SCORE_LABELS.index)).sort_values("_k")
            show_fig(_score_fig(t, avg, obs), key=f"{P}_score", height=400, legend=False)
            w, b = t.loc[t["val"].idxmax()], t.iloc[-1]
            _note(f"El rango <b>{esc(w['seg'])}</b> tiene <b>{fmt_pct(w['val'])}</b> de Alto frente a "
                  f"<b>{fmt_pct(b['val'])}</b> en <b>{esc(b['seg'])}</b> ({_x(w['val'] / b['val'] if b['val'] else np.nan)})."
                  " «≤ 400» puede reflejar historiales de crédito escasos más que mal comportamiento.")
    with right, st.container(border=True):
        section("Tipo de interés × número de cuotas", f"% Alto {basis.lower()} en cada combinación; las celdas con "
                f"menos de {gmin} créditos se ocultan.", kicker="Condiciones del crédito")
        show_fig(_heatmap_fig(dff, flag, gmin, avg, f"% Alto {basis.lower()}"), key=f"{P}_heat", height=400, legend=False)
        _note("Un color más oscuro indica más riesgo. Compara filas: la modalidad «Anualidad Vencida» concentra el riesgo "
              "según el reporte; aquí se verifica con el filtro actual.")

    with st.expander("⚖️ Nota de equidad: variables sensibles solo para auditoría"):
        st.markdown("Género, edad, estado civil y grupo étnico **no se usan para focalizar la gestión**: solo sirven "
                    "para auditar que el modelo no marque de forma desproporcionada a un grupo. La tabla compara la tasa "
                    "de alerta del modelo con el riesgo observado por grupo.")
        rows = []
        for col, lbl in [("genero_txt", "Género"), ("rango_edad", "Rango de edad")]:
            if col not in dff.columns:
                continue
            t = _rate_table(dff, _labels(dff, col).to_numpy(), "_ap").rename(columns={"seg": "g"})
            t = t[(t["n"] >= 30) & (t["g"] != "Sin dato")]
            for _, r in t.sort_values("g", key=lambda s: s.map(_order_key(col))).iterrows():
                rows.append({"Atributo": lbl, "Grupo": r["g"], "Créditos": int(r["n"]), "% Alto predicho": r["pct_pred"],
                             "% Alto observado": r["pct_obs"], "Razón vs. promedio": r["pct_pred"] / AVG_PRED
                             if AVG_PRED else np.nan})
        if rows:
            ft = pd.DataFrame(rows)
            if not has_truth:
                ft = ft.drop(columns=["% Alto observado"])
            st.dataframe(ft, hide_index=True, width="stretch", column_config={
                "% Alto predicho": st.column_config.NumberColumn(format="percent"),
                "% Alto observado": st.column_config.NumberColumn(format="percent"),
                "Razón vs. promedio": st.column_config.NumberColumn(format="%.2f×",
                                                                    help="Aceptable entre 0,80× y 1,25×")})


# ======================================================================================
# Pestaña 3 · Territorio
# ======================================================================================
@st.cache_data(show_spinner=False)
def _city_table(d: pd.DataFrame) -> pd.DataFrame:
    x = d[["ciudad_norm", "departamento_limpio", "latitud", "longitud"]].copy()
    x["key"] = x["ciudad_norm"].astype(str).map(nrm).str.replace(r"[^a-z ]", "", regex=True).str.replace(
        r"\s+", " ", regex=True).str.strip()
    x = x.join(_flags(d))
    x = x[pd.to_numeric(x["latitud"], errors="coerce").notna() & pd.to_numeric(x["longitud"], errors="coerce").notna()]
    g = x.groupby("key").agg(ciudad=("ciudad_norm", lambda s: s.value_counts().index[0]),
                             depto=("departamento_limpio", lambda s: s.value_counts().index[0]), lat=("latitud", "median"),
                             lon=("longitud", "median"), n=("_ap", "size"), pct_pred=("_ap", "mean"),
                             pct_obs=("_ao", "mean")).reset_index(drop=True)
    g["ciudad"] = g["ciudad"].astype(str).str.title().str.replace("D.C.", "D. C.", regex=False)
    return g


def _tab_territory() -> None:
    opts = ["Predicho", "Observado"] if has_truth else ["Predicho"]
    c1, c2, _ = st.columns([1.3, 1.3, 2], vertical_alignment="bottom")
    basis = c1.segmented_control("Riesgo Alto", opts, default=opts[0], key=f"{P}_geo_basis", required=True) or opts[0]
    dmin = c2.select_slider("n mínimo por departamento", [30, 50, 100, 200], value=50, key=f"{P}_geo_min")
    obs = basis == "Observado"
    flag, vcol = ("_ao", "pct_obs") if obs else ("_ap", "pct_pred")
    avg = AVG_OBS if obs else AVG_PRED
    p = pal()

    # KPI Bogotá vs resto
    is_bog = dff["departamento_limpio"].astype(str).map(nrm).str.contains("bogota")
    bog, rest = dff[is_bog], dff[~is_bog]
    rb, rr = (_rate(bog, flag) if len(bog) else np.nan), (_rate(rest, flag) if len(rest) else np.nan)
    diff = (rb - rr) * 100 if np.isfinite(rb) and np.isfinite(rr) else np.nan
    kpi_row([
        kpi_card("Bogotá D. C.", fmt_pct(rb) if np.isfinite(rb) else "—",
                 f"% Alto {basis.lower()} · {fmt_int(len(bog))} créditos ({fmt_pct(len(bog) / len(dff))} de la cartera)",
                 tone="accent", icon="🏙️",
                 delta=(f"{'+' if diff >= 0 else '−'}{fmt_num(abs(diff), 1)} pp vs. resto" if np.isfinite(diff) else None),
                 delta_dir="up" if np.isfinite(diff) and diff > 0.05 else "down" if np.isfinite(diff) and diff < -0.05 else "flat"),
        kpi_card("Resto del país", fmt_pct(rr) if np.isfinite(rr) else "—",
                 f"% Alto {basis.lower()} · {fmt_int(len(rest))} créditos en "
                 f"{rest['departamento_limpio'].nunique()} departamentos", tone="ink", icon="🗺️",
                 delta="Referencia de comparación", delta_dir="flat"),
    ])

    st.write("")
    left, right = st.columns([1.25, 1], gap="medium")
    with left, st.container(border=True):
        section("Mapa por ciudad", "Tamaño = número de créditos; color = % de riesgo Alto. Pasa el cursor para ver "
                "cada municipio.", kicker="Territorio")
        g = _city_table(dff)
        g = g[g[vcol].notna()]
        if g.empty:
            _note("No hay coordenadas para los créditos del filtro actual.")
        else:
            cap = float(max(np.nanquantile(g.loc[g["n"] >= 20, vcol], 0.95) if (g["n"] >= 20).any() else 0,
                            avg * 2, 0.05))
            g = g.assign(hover_n=g["n"].map(fmt_int), hover_p=g[vcol].map(fmt_pct)).sort_values("n", ascending=False)
            fig = px.scatter_map(g, lat="lat", lon="lon", size="n", color=vcol, size_max=38,
                                 color_continuous_scale=SEQ_RISK, range_color=[0, cap], zoom=4.6,
                                 center={"lat": 5.4, "lon": -74.0}, custom_data=["ciudad", "depto", "hover_n", "hover_p"],
                                 map_style="carto-darkmatter" if theme_mode() == "dark" else "carto-positron")
            fig.update_traces(marker=dict(opacity=0.82), hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>Créditos: %{customdata[2]}"
                f"<br>% Alto {basis.lower()}: <b>%{{customdata[3]}}</b><extra></extra>"))
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(
                title=dict(text="% Alto", font=dict(size=11, color=p["muted"])), tickformat=".0%", thickness=10,
                len=0.7, outlinewidth=0, tickfont=dict(color=p["muted"], size=10)))
            show_fig(fig, key=f"{P}_map", height=520, legend=False)
            _note(f"{fmt_int(len(g))} municipios con créditos. La escala de color se satura en {fmt_pct(cap, 0)} para "
                  "que las ciudades pequeñas con tasas extremas no dominen la lectura.")
    with right, st.container(border=True):
        section("Top 10 departamentos", f"% Alto {basis.lower()} con al menos {dmin} créditos; la línea punteada es el "
                "promedio nacional.", kicker="Ranking territorial")
        t = _rate_table(dff, dff["departamento_limpio"].astype(str).to_numpy(), flag)
        t = t[(t["nv"] >= dmin) & (~t["seg"].str.lower().isin({"nan", "sin dato", ""}))]
        if len(t) < 2:
            _note("Pocos departamentos cumplen el n mínimo con el filtro actual.")
        else:
            t = t.sort_values("val", ascending=False).head(10)
            show_fig(_rank_fig(t, f"% Alto {basis.lower()}", avg), key=f"{P}_deptos", height=520, legend=False)
            top = t.iloc[0]
            _note(f"<b>{esc(top['seg'])}</b> lidera con <b>{fmt_pct(top['val'])}</b> ({_x(top['val'] / avg if avg else np.nan)} "
                  f"el promedio nacional; IC 95 %: {fmt_pct(top['lo'])} – {fmt_pct(top['hi'])}).")


# ======================================================================================
# Pestañas
# ======================================================================================
_TABS = ["🧭 Explorador de segmentos", "📊 Perfil de riesgo", "🗺️ Territorio"]
tabs = st.tabs(_TABS, key=f"{P}_tab", on_change="rerun")
_any_open = any(t.open for t in tabs)
for i, (tb, fn) in enumerate(zip(tabs, [_tab_explorer, _tab_profile, _tab_territory])):
    with tb:
        if tb.open or (not _any_open and i == 0):
            fn()

footer()
