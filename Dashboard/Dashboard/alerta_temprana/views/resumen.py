"""Resumen ejecutivo — ¿cuánta cartera está en riesgo, dónde se concentra y qué hacer primero?

Vista para Dirección Financiera y Gestión de Cartera. DB-03: créditos, riesgo Alto (n y %), mora
Datacrédito, monto total y monto por categoría de riesgo. ME-03: reporte ejecutivo descargable.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.components import download_bar, empty_state, esc, filter_chips, footer, insight, kpi_card, page_header, section
from core.config import APP_NAME, PROGRAMA_ACADEMICO, RECALL_ALTO_TARGET, RISK_COLORS, RISK_ORDER, UNIVERSIDAD
from core.data import get_active_df, get_active_meta, segment_summary
from core.filters import active_chips, apply_filters
from core.metrics import confusion, load_artifacts
from core.nav import PAGES, can_access
from core.theme import fmt_cop, fmt_delta_pp, fmt_int, fmt_num, fmt_pct, fmt_period, pal, show_fig

P = "resumen"
NAN = float("nan")
N_CMP = 6  # meses de la ventana de comparación de los deltas
MIN_N = 30  # n mínimo por segmento
_DIMS = {"Segmento": "programa_cluster", "Facultad": "facultad", "Tipo de interés": "tipo_interes",
         "Scoring": "score_corto", "Cohorte": "cohorte"}
_LINKS = [
    ("cola", "Priorizar casos", "Lista ordenada por prioridad con rutas R1–R4 y exportación."),
    ("segmentos", "Segmentos y territorio", "Perfiles de riesgo por programa, scoring y mapa por departamento."),
    ("modelo", "Auditar el modelo", "Métricas por clase, curvas ROC y umbral de alerta para Alto."),
    ("carga", "Cargar y predecir", "Puntuar un archivo nuevo de créditos con el modelo."),
]

_CSS = """
<style>
.rs-spark{width:100%;height:30px;display:block;margin-top:10px;overflow:visible}
.rs-risk{background:var(--sat-surface);border:1px solid var(--sat-border);border-radius:var(--sat-radius);
  padding:14px 18px 12px 20px;box-shadow:var(--sat-shadow);position:relative;overflow:hidden}
.rs-risk:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--rc)}
.rs-risk-top{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--sat-muted);font-weight:600}
.rs-dot{width:10px;height:10px;border-radius:50%;background:var(--rc);box-shadow:0 0 0 3px color-mix(in srgb,var(--rc) 22%,transparent)}
.rs-risk-name{color:var(--sat-text);font-weight:700}
.rs-pill{margin-left:auto;font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;
  background:color-mix(in srgb,var(--rc) 13%,transparent);color:var(--sat-text);white-space:nowrap}
.rs-risk-val{font-family:'Space Grotesk','Inter',sans-serif;font-size:28px;font-weight:700;color:var(--sat-text);margin-top:6px;line-height:1.1}
.rs-risk-val small{font-family:'Inter',sans-serif;font-size:13px;color:var(--sat-muted);font-weight:500}
.rs-risk-money{font-size:13px;color:var(--sat-muted);margin-top:3px}
.rs-risk-money b{color:var(--sat-text)}
.rs-meter{height:7px;border-radius:7px;background:color-mix(in srgb,var(--rc) 14%,transparent);margin-top:9px;overflow:hidden}
.rs-meter i{display:block;height:100%;background:var(--rc);border-radius:7px}
.rs-legend{display:flex;flex-direction:column;gap:6px}
.rs-legend-row{display:grid;grid-template-columns:14px 1fr auto auto;gap:10px;align-items:center;font-size:13px;
  color:var(--sat-muted);padding:6px 10px;border-radius:10px;background:var(--sat-surface-2);border:1px solid var(--sat-border)}
.rs-legend-row .sw{width:10px;height:10px;border-radius:3px}
.rs-legend-row .nm{color:var(--sat-text);font-weight:600}
.rs-legend-row .v{color:var(--sat-text);font-weight:700;font-variant-numeric:tabular-nums}
.rs-legend-row .s{font-variant-numeric:tabular-nums;min-width:52px;text-align:right}
.rs-note{font-size:12.5px;color:var(--sat-muted);margin:-4px 0 6px 0;line-height:1.45}
.rs-t{font-weight:700;color:var(--sat-text);font-size:15px}
.rs-d{font-size:12.8px;color:var(--sat-muted);line-height:1.4;min-height:54px}
.rs-mini{background:var(--sat-surface-2);border:1px solid var(--sat-border);border-radius:14px;padding:11px 14px;height:100%}
.rs-mini .l{font-size:12px;color:var(--sat-muted);font-weight:600}
.rs-mini .v{font-family:'Space Grotesk','Inter',sans-serif;font-size:24px;font-weight:700;color:var(--sat-text);line-height:1.15}
.rs-mini .s{font-size:12px;color:var(--sat-muted)}
.rs-target{position:relative;height:8px;border-radius:8px;background:color-mix(in srgb,var(--sat-alto) 14%,transparent);margin:8px 0 4px}
.rs-target i{position:absolute;left:0;top:0;bottom:0;border-radius:8px;background:var(--sat-alto)}
.rs-target b{position:absolute;top:-4px;width:2px;height:16px;background:var(--sat-text);border-radius:2px}
.rs-head{display:flex;gap:14px;align-items:flex-start;background:var(--sat-surface);border:1px solid var(--sat-border);
  border-radius:var(--sat-radius);padding:16px 20px;box-shadow:var(--sat-shadow);position:relative;overflow:hidden;margin-bottom:12px}
.rs-head:before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--sat-accent)}
.rs-head .q{font-family:'Space Grotesk',sans-serif;font-size:40px;line-height:.8;color:var(--sat-accent);font-weight:700}
.rs-head .k{font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--sat-muted);margin-bottom:3px}
.rs-head .t{font-size:15px;color:var(--sat-text);line-height:1.55}
[data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(> [class^="rs-"], > .sat-insight){margin-bottom:0}
.rs-kpi .sat-kpi .val{white-space:nowrap}
.rs-kpi .sat-kpi .sub{min-height:38px;line-height:1.5}
@media (max-width:1180px){
  [data-testid="stMain"] [data-testid="stColumn"]{min-width:calc(50% - 16px)}
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has([data-testid="stPlotlyChart"]) > [data-testid="stColumn"]{min-width:100%}
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]{row-gap:14px}
}
@media (max-width:1000px){
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.sat-insight) > [data-testid="stColumn"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.rs-risk) > [data-testid="stColumn"]{min-width:100%}
}
</style>
"""


# ---------------------------------------------------------------- cálculo
def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _div(a, b) -> float:
    try:
        return float(a) / float(b) if b else NAN
    except (TypeError, ValueError):
        return NAN


def _x(v) -> str:
    return "—" if _isnan(v) else f"{fmt_num(v, 1)}×"


def _nb(s) -> str:
    return str(s).replace(" ", " ")


def _stats(d: pd.DataFrame) -> dict:
    yp, monto, mora = d["y_pred"].astype(str), d["valor_financiacion"].fillna(0), d["mora_flag"].fillna(0)
    out = {"n": len(d), "est": int(d["id_estudiante"].nunique()), "monto": float(monto.sum()), "n_mora": int(mora.sum())}
    for r in RISK_ORDER:
        m = (yp == r).to_numpy()
        out[f"n_{r}"], out[f"monto_{r}"] = int(m.sum()), float(monto[m].sum())
        out[f"mora_{r}"] = float(mora[m].mean()) if m.any() else NAN
    out["pct_alto"], out["pct_mora"] = _div(out["n_Alto"], out["n"]), _div(out["n_mora"], out["n"])
    out["pct_monto_alto"], out["ticket"] = _div(out["monto_Alto"], out["monto"]), _div(out["monto"], out["n"])
    return out


def _monthly(d: pd.DataFrame) -> pd.DataFrame:
    t = d.dropna(subset=["periodo"])
    t = t[t["periodo"].astype(str) != "nan"].assign(_a=(t["y_pred"].astype(str) == "Alto").astype(int),
                                                    _m=t["valor_financiacion"].fillna(0))
    return (t.groupby("periodo").agg(n=("llave2", "size"), alto=("_a", "mean"), mora=("mora_flag", "mean"),
                                     monto=("_m", "sum")).reset_index().sort_values("periodo"))


def _seg_table(d: pd.DataFrame, col: str, min_n: int, g: float) -> pd.DataFrame:
    s = segment_summary(d, col, min_n=min_n)
    if s.empty:
        return s
    s = s.assign(segmento=s[col].astype(str), n_alto=(s["pct_alto"] * s["creditos"]).round().astype(int),
                 lift=s["pct_alto"] / g if g and not _isnan(g) else NAN)
    return s.sort_values(["pct_alto", "creditos"], ascending=[False, False]).reset_index(drop=True)


def _compare(d: pd.DataFrame, ma, mb) -> dict:
    a = (d["y_pred"].astype(str) == "Alto").to_numpy()
    ma, mb = np.asarray(ma, dtype=bool), np.asarray(mb, dtype=bool)
    pa = float(a[ma].mean()) if ma.any() else NAN
    pb = float(a[mb].mean()) if mb.any() else NAN
    return {"na": int(ma.sum()), "nb": int(mb.sum()), "pa": pa, "pb": pb, "alto_a": int(a[ma].sum()), "lift": _div(pa, pb)}


def _truth(d: pd.DataFrame) -> dict | None:
    t = d[d["has_truth"] & d["y_pred"].notna()]
    if t.empty:
        return None
    cm = confusion(t["y_true"], t["y_pred"])
    tp, real, pred = int(cm[0, 0]), int(cm[0].sum()), int(cm[:, 0].sum())
    return {"cm": cm, "n": int(cm.sum()), "acc": _div(np.trace(cm), cm.sum()), "rec": _div(tp, real),
            "prec": _div(tp, pred), "tp": tp, "real_alto": real, "pred_alto": pred}


def _headline(s: dict, truth: dict | None) -> str:
    txt = (f"De <b>{fmt_int(s['n'])}</b> créditos filtrados, <b>{fmt_int(s['n_Alto'])} ({fmt_pct(s['pct_alto'])}) están "
           f"en riesgo Alto</b> y comprometen <b>{fmt_cop(s['monto_Alto'])}</b> ({fmt_pct(s['pct_monto_alto'])} del monto "
           f"financiado). La mora real reportada por Datacrédito es {fmt_pct(s['pct_mora'])}.")
    if truth and not _isnan(truth["rec"]) and truth["rec"] < RECALL_ALTO_TARGET:
        txt += (f" Ojo: la regla actual anticipa solo {fmt_pct(truth['rec'])} de los Alto observados; el umbral debe "
                f"ajustarse para cumplir DE-03.")
    return txt


def _findings(d: pd.DataFrame, s: dict) -> list[dict]:
    """Tres hallazgos automáticos: segmento más riesgoso, antigüedad y scoring externo."""
    out, g = [], s["pct_alto"]
    top = next(((c, t.iloc[0]) for c in ["programa_cluster", "programa", "facultad"]
                if len(t := _seg_table(d, c, MIN_N, g)) >= 2), None)
    if top is not None and g > 0:
        c, row = top
        out.append({"title": "Segmento que más concentra riesgo", "tone": "alto", "icon": "🎯",
                    "html": f"<b>{esc(row['segmento'])}</b> tiene <b>{fmt_pct(row['pct_alto'])}</b> de sus créditos en "
                            f"Alto: <b>{_x(row['lift'])}</b> el promedio del filtro ({fmt_pct(g)}). Son "
                            f"{fmt_int(row['n_alto'])} alertas y {fmt_cop(row['exposicion_alto'])} financiados en Alto."})
    ant = d["antiguedad_meses"]
    c = _compare(d, (ant < 12).fillna(False), (ant >= 12).fillna(False))
    if c["na"] >= 10 and c["nb"] >= 10:
        up = c["pa"] > c["pb"]
        out.append({"title": "Los créditos recientes pesan más" if up else "Antigüedad del crédito",
                    "tone": "medio" if up else "bajo", "icon": "⏱️",
                    "html": f"Los créditos con <b>menos de 12 meses</b> tienen <b>{fmt_pct(c['pa'])}</b> en Alto frente a "
                            f"{fmt_pct(c['pb'])} del resto (<b>{_x(c['lift'])}</b>) y aportan "
                            f"{fmt_pct(_div(c['alto_a'], max(s['n_Alto'], 1)))} de las alertas."})
    sc = d["score_corto"].astype(str)
    c = _compare(d, sc == "≤ 400", sc == "≥ 721")
    if c["na"] >= 10 and c["nb"] >= 10:
        out.append({"title": "Scoring externo (Datacrédito)", "tone": "accent", "icon": "📉",
                    "html": f"Con scoring <b>≤ 400</b> el {fmt_pct(c['pa'])} está en Alto frente a {fmt_pct(c['pb'])} con "
                            f"<b>≥ 721</b> (<b>{_x(c['lift'])}</b>)"
                            + (": peor historial crediticio, más riesgo." if c["pa"] > c["pb"] else
                               ". En este filtro el scoring no separa el riesgo en la dirección esperada.")})
    return out[:3] or [{"title": "Sin contrastes suficientes", "tone": "info", "icon": "ℹ️",
                        "html": "El filtro actual deja muy pocos créditos para comparar segmentos."}]


# ---------------------------------------------------------------- gráficas
def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _evolution_fig(d: pd.DataFrame) -> go.Figure | None:
    """Barras apiladas por riesgo (créditos por mes) + franja superior con el % en Alto."""
    p = pal()
    t = d.dropna(subset=["periodo"])
    t = t[t["periodo"].astype(str) != "nan"]
    if t.empty:
        return None
    piv = (t.groupby(["periodo", "y_pred"], observed=False).size().unstack("y_pred")
           .reindex(columns=RISK_ORDER).fillna(0).sort_index())
    tot = piv.sum(axis=1)
    share = (piv["Alto"] / tot.replace(0, np.nan)).astype(float)
    idx = [str(x) for x in piv.index]
    labels = [fmt_period(x) for x in idx]
    avg = _div(piv["Alto"].sum(), tot.sum())
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.30, 0.70], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(
        x=labels, y=share, mode="lines", line=dict(color=RISK_COLORS["Alto"], width=2, shape="spline", smoothing=0.5),
        fill="tozeroy", fillcolor=_alpha(RISK_COLORS["Alto"], 0.10), showlegend=False,
        customdata=[fmt_pct(v) for v in share], hovertemplate="<b>% en Alto</b>: %{customdata}<extra></extra>"),
        row=1, col=1)
    if not _isnan(avg):
        fig.add_hline(y=avg, line=dict(color=p["muted"], width=1, dash="dot"), row=1, col=1)
        fig.add_annotation(x=0, xref="x domain", y=avg, yref="y", text=f"Promedio {fmt_pct(avg)}", showarrow=False,
                           xanchor="left", yanchor="bottom", yshift=2, font=dict(size=11, color=p["muted"]))
    valid = np.where(share.notna().to_numpy())[0]
    if len(tot) >= 4 and tot.iloc[-1] < 0.35 * float(tot.iloc[:-1].tail(12).median()):
        valid = valid[valid < len(share) - 1]  # último mes incompleto: no se anota
    if len(valid):
        i = int(valid[np.argmax(share.to_numpy()[valid])])
        fig.add_annotation(x=labels[i], y=share.iloc[i], xref="x", yref="y", text=f"<b>máx. {fmt_pct(share.iloc[i])}</b>"
                           f" · {labels[i]}", showarrow=False, xanchor="right", yanchor="bottom", xshift=-6,
                           font=dict(size=11, color=p["text"]))
    for r in RISK_ORDER:
        pct = (piv[r] / tot.replace(0, np.nan)).fillna(0)
        fig.add_trace(go.Bar(
            x=labels, y=piv[r], name=r, marker=dict(color=RISK_COLORS[r] if r != "Bajo" else _alpha(RISK_COLORS["Bajo"], .62),
                                                     line=dict(width=0)),
            customdata=np.c_[[fmt_int(v) for v in piv[r]], [fmt_pct(x) for x in pct]],
            hovertemplate=f"<b>{r}</b>: %{{customdata[0]}} créditos (%{{customdata[1]}})<extra></extra>"), row=2, col=1)
    yr = [i for i, x in enumerate(idx) if x.endswith("-01")]
    if len(labels) > 12 and len(yr) >= 2:
        tv, tt = [labels[i] for i in yr], [idx[i][:4] for i in yr]
    else:
        tv = tt = labels[::max(1, math.ceil(len(labels) / 8))]
    fig.update_xaxes(type="category", tickmode="array", tickvals=tv, ticktext=tt, tickangle=0, row=2, col=1)
    fig.update_xaxes(type="category", showticklabels=False, row=1, col=1)
    ymax = float(np.nanmax(share.to_numpy())) if share.notna().any() else 0.1
    fig.update_yaxes(tickformat=".0%", range=[0, max(ymax * 1.35, 0.02)], nticks=3, title_text="% Alto", row=1, col=1)
    fig.update_yaxes(title_text="Créditos", row=2, col=1)
    fig.update_layout(barmode="stack", bargap=0.22, hovermode="x unified", margin=dict(l=8, r=16, t=26, b=8),
                      legend=dict(y=1.0, yanchor="bottom", x=0, traceorder="normal"))
    return fig


def _donut_fig(s: dict) -> go.Figure:
    p = pal()
    vals = [s[f"n_{r}"] for r in RISK_ORDER]
    fig = go.Figure(go.Pie(
        labels=RISK_ORDER, values=vals, hole=0.72, sort=False, direction="clockwise", textinfo="none",
        marker=dict(colors=[RISK_COLORS[r] for r in RISK_ORDER], line=dict(color=p["bg"], width=3)),
        customdata=[fmt_pct(_div(v, s["n"])) for v in vals],
        hovertemplate="<b>%{label}</b><br>%{customdata} de los créditos<extra></extra>"))
    fig.add_annotation(text=f"<b>{fmt_int(s['n'])}</b>", x=.5, y=.54, showarrow=False, font=dict(size=26, color=p["text"]))
    fig.add_annotation(text="créditos", x=.5, y=.41, showarrow=False, font=dict(size=12, color=p["muted"]))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    return fig


def _top_fig(t: pd.DataFrame, col: str, g: float) -> go.Figure:
    p = pal()
    t = t.iloc[::-1]
    lab = [f"Scoring {v}" if col == "score_corto" else (v if len(v) <= 34 else v[:33] + "…") for v in t["segmento"]]
    fig = go.Figure(go.Bar(
        y=lab, x=t["pct_alto"], orientation="h", cliponaxis=False, textposition="outside",
        marker=dict(color=[RISK_COLORS["Alto"] if (not _isnan(l) and l > 1) else p["subtle"] for l in t["lift"]],
                    line=dict(width=0)),
        text=[f"<b>{fmt_pct(a)}</b> · {_x(l)}" for a, l in zip(t["pct_alto"], t["lift"])],
        textfont=dict(color=p["text"], size=12),
        customdata=np.c_[[fmt_int(v) for v in t["creditos"]], [fmt_int(v) for v in t["n_alto"]],
                         [fmt_cop(v) for v in t["exposicion_alto"]]],
        hovertemplate="<b>%{y}</b><br>Créditos: %{customdata[0]} · en Alto: %{customdata[1]}"
                      "<br>Monto en Alto: %{customdata[2]}<extra></extra>"))
    if not _isnan(g):
        fig.add_vline(x=g, line=dict(color=p["text"], width=1.2, dash="dot"), layer="below")
        fig.add_annotation(x=g, y=1, yref="paper", text=f"Promedio {fmt_pct(g)}", showarrow=False, xanchor="left",
                           yanchor="bottom", xshift=4, font=dict(size=11, color=p["muted"]))
    fig.update_xaxes(range=[0, max(float(t["pct_alto"].max()) * 1.3, 0.02)], tickformat=".0%",
                     title_text="% de créditos en riesgo Alto")
    fig.update_yaxes(showgrid=False, ticksuffix="  ")
    fig.update_layout(bargap=0.38, margin=dict(l=8, r=8, t=30, b=8))
    return fig


def _risk_card(r: str, s: dict) -> str:
    pct_m = _div(s[f"monto_{r}"], s["monto"])
    w = 0 if _isnan(pct_m) else max(0.0, min(1.0, pct_m)) * 100
    return (f"<div class='rs-risk' style='--rc:{RISK_COLORS[r]}'><div class='rs-risk-top'><span class='rs-dot'></span>"
            f"<span class='rs-risk-name'>Riesgo {r}</span><span class='rs-pill'>{fmt_pct(_div(s[f'n_{r}'], s['n']))} de los "
            f"créditos</span></div><div class='rs-risk-val'>{fmt_int(s[f'n_{r}'])} <small>créditos</small></div>"
            f"<div class='rs-risk-money'><b>{fmt_cop(s[f'monto_{r}'])}</b> financiados · {fmt_pct(pct_m)} del monto · "
            f"mora DC {fmt_pct(s[f'mora_{r}'])}</div><div class='rs-meter'><i style='width:{w:.1f}%'></i></div></div>")


def _spark(vals, color: str) -> str:
    v = [float(x) for x in vals if not _isnan(x)]
    if len(v) < 3:
        return ""
    lo, rng = min(v), (max(v) - min(v)) or 1.0
    line = " ".join(f"{i * 100 / (len(v) - 1):.2f},{27 - (x - lo) / rng * 23:.2f}" for i, x in enumerate(v))
    return (f"<svg class='rs-spark' viewBox='0 0 100 30' preserveAspectRatio='none' aria-hidden='true'>"
            f"<polygon points='0,30 {line} 100,30' fill='{color}' fill-opacity='0.10'/><polyline points='{line}' "
            f"fill='none' stroke='{color}' stroke-width='1.8' vector-effect='non-scaling-stroke'/></svg>")


# ---------------------------------------------------------------- reporte ejecutivo (ME-03)
def _report_html(s: dict, kpis: list[dict], finds: list[dict], top: pd.DataFrame, dim: str, truth: dict | None,
                 chips: list, period: str, headline: str) -> str:
    k_html = "".join(f"<div class='k'><div class='kl'>{esc(k['label'])}</div><div class='kv'>{esc(k['value'])}</div>"
                     f"<div class='ks'>{esc(k['sub'])}</div></div>" for k in kpis)
    risk = "".join(f"<tr><td><span class='sw' style='background:{RISK_COLORS[x]}'></span>{x}</td><td>{fmt_int(s[f'n_{x}'])}"
                   f"</td><td>{fmt_pct(_div(s[f'n_{x}'], s['n']))}</td><td>{fmt_cop(s[f'monto_{x}'])}</td>"
                   f"<td>{fmt_pct(s[f'mora_{x}'])}</td></tr>" for x in RISK_ORDER)
    seg = "".join(f"<tr><td>{esc(r['segmento'])}</td><td>{fmt_int(r['creditos'])}</td><td>{fmt_int(r['n_alto'])}</td>"
                  f"<td><b>{fmt_pct(r['pct_alto'])}</b></td><td>{_x(r['lift'])}</td><td>{fmt_cop(r['exposicion_alto'])}</td>"
                  f"</tr>" for _, r in top.iterrows()) or "<tr><td colspan='6'>Sin segmentos con el n mínimo.</td></tr>"
    f_html = "".join(f"<div class='f {f['tone']}'><b>{esc(f['icon'])} {esc(f['title'])}</b><div>{f['html']}</div></div>"
                     for f in finds)
    t_html = (f"Sobre {fmt_int(truth['n'])} créditos con riesgo observado: concordancia <b>{fmt_pct(truth['acc'])}</b>, "
              f"recall de Alto <b>{fmt_pct(truth['rec'])}</b> (meta DE-03 {fmt_pct(RECALL_ALTO_TARGET, 0)}) y precisión "
              f"de Alto <b>{fmt_pct(truth['prec'])}</b>." if truth else "El dataset activo no incluye riesgo observado.")
    ch = "".join(f"<span>{esc(k)}: {esc(v)}</span>" for k, v in chips) or "<span>Sin filtros: toda la cartera</span>"
    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Reporte ejecutivo · {esc(APP_NAME)}</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#F6F6F3;color:#141414;font:14px/1.45 Inter,'Segoe UI',Arial,sans-serif}}
.w{{max-width:1000px;margin:0 auto;padding:28px 24px 40px}} .hero{{background:linear-gradient(120deg,#0B0B0C,#26251F);color:#F7F7F2;border-radius:18px;padding:24px 28px}}
.ey{{color:#FFD100;font-size:11px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}} h1{{font-size:26px;margin:6px 0 4px}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}} .chips span{{font-size:11.5px;padding:3px 9px;border-radius:999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.16)}}
h2{{font-size:17px;margin:26px 0 10px;border-left:4px solid #FFD100;padding-left:8px}}
.kg{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}} .k{{background:#fff;border:1px solid #E7E5DF;border-radius:14px;padding:12px 14px}}
.kl,.ks{{font-size:11.5px;color:#6B6A64}} .kv{{font-size:22px;font-weight:800;margin-top:4px}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #E7E5DF;font-size:13px}} th{{text-align:left;background:#F1EFE8;padding:8px 10px;font-size:12px}}
td{{padding:7px 10px;border-top:1px solid #EFEDE7}} td:not(:first-child),th:not(:first-child){{text-align:right}}
.sw{{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:7px}}
.head,.card,.f{{background:#fff;border:1px solid #E7E5DF;border-left:5px solid #FFD100;border-radius:12px;padding:12px 15px;margin-top:14px}}
.fg{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}} .f{{margin:0;font-size:13px}} .f.alto{{border-left-color:#E5484D}} .f.medio{{border-left-color:#F5A524}}
.muted{{color:#6B6A64;font-size:12px}} @media (max-width:760px){{.kg,.fg{{grid-template-columns:1fr 1fr}}}}
@media print{{.hero{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body><div class="w"><div class="hero"><div class="ey">◆ {esc(APP_NAME)} · Reporte ejecutivo</div>
<h1>Riesgo de morosidad de la cartera estudiantil</h1><div class="chips"><span>Generado {datetime.now():%d/%m/%Y %H:%M}</span>
<span>Fuente: {esc(get_active_meta().get('name', ''))}</span><span>{esc(period)}</span></div><div class="chips">{ch}</div></div>
<div class="head"><b>En síntesis.</b> {headline}</div>
<h2>Indicadores clave (DB-03)</h2><div class="kg">{k_html}</div>
<h2>Distribución y monto por riesgo predicho</h2>
<table><tr><th>Riesgo</th><th>Créditos</th><th>%</th><th>Monto</th><th>Mora DC</th></tr>{risk}</table>
<h2>Hallazgos automáticos</h2><div class="fg">{f_html}</div>
<h2>Segmentos con mayor % en Alto · {esc(dim)}</h2>
<table><tr><th>Segmento</th><th>Créditos</th><th>En Alto</th><th>% Alto</th><th>Lift</th><th>Monto en Alto</th></tr>{seg}</table>
<p class="muted">Segmentos con al menos {MIN_N} créditos. Lift = % Alto del segmento ÷ % Alto del filtro.</p>
<h2>Predicho vs. observado</h2><div class="card">{t_html}</div>
<p class="muted" style="margin-top:24px">Riesgo predicho por el Random Forest seleccionado (clase más probable). Mora según
Datacrédito. Nombres anonimizados; sin datos personales. {esc(PROGRAMA_ACADEMICO)} · {esc(UNIVERSIDAD)}.</p>
</div></body></html>"""


@st.dialog("Vista previa · Reporte ejecutivo", width="large")
def _preview_dialog(doc: str, fname: str) -> None:
    st.iframe(doc, height=620)
    st.download_button("Descargar reporte (HTML)", doc.encode("utf-8"), file_name=fname, mime="text/html",
                       icon=":material/download:", type="primary", key=f"{P}_report_dl_dialog", width="stretch")


# ================================================================ página
st.markdown(_CSS, unsafe_allow_html=True)
df_all = get_active_df()
dff = apply_filters(df_all)
p = pal()

periods = sorted(str(x) for x in dff["periodo"].dropna().unique() if str(x) != "nan")
period_txt = f"Aprobaciones {fmt_period(periods[0])} – {fmt_period(periods[-1])}" if periods else "Sin fechas de aprobación"
page_header("Resumen ejecutivo", "Cuánta cartera está en riesgo según el modelo Random Forest, dónde se concentra y qué "
            "hacer primero.", eyebrow="Visión general · Dirección y Cartera", meta=[period_txt],
            highlight=f"{fmt_int((dff['y_pred'].astype(str) == 'Alto').sum())} créditos en alerta alta")
filter_chips(active_chips(df_all))
if dff.empty:
    empty_state("Sin créditos para este filtro", "Ningún crédito cumple los filtros de la barra lateral. "
                "Amplía el periodo o limpia algunos criterios.")
    footer()
    st.stop()

S = _stats(dff)
truth = _truth(dff)

# ---- 1. KPI (DB-03) con delta contra los 6 meses previos ----
rec_s = prev_s = None
if periods:
    end = pd.Period(periods[-1], "M")
    pp = (base := apply_filters(df_all, skip={"periodo"}))["periodo"].astype(str)
    rec_df = base[(pp > str(end - N_CMP)) & (pp <= str(end))]
    prev_df = base[(pp > str(end - 2 * N_CMP)) & (pp <= str(end - N_CMP))]
    if len(rec_df) >= 20 and len(prev_df) >= 20:
        rec_s, prev_s = _stats(rec_df), _stats(prev_df)


def _delta(key: str, rel: bool) -> tuple[str, str]:
    if not rec_s or _isnan(rec_s[key]) or _isnan(prev_s[key]):
        return "■ sin comparación", "flat"
    d = rec_s[key] / prev_s[key] - 1 if rel else rec_s[key] - prev_s[key]
    arrow = "▲" if d > 0.0005 else ("▼" if d < -0.0005 else "■")
    txt = f"{'+' if d >= 0 else '−'}{fmt_num(abs(d) * 100, 1)} %" if rel else fmt_delta_pp(d)
    return f"{arrow} {txt} · {N_CMP} m", "flat" if rel or abs(d) < 0.0005 else ("up" if d > 0 else "down")


mon = _monthly(dff).tail(12)
kpis = [
    {"label": "Créditos filtrados", "value": fmt_int(S["n"]), "tone": "ink", "icon": "📄", "d": _delta("n", True),
     "sub": f"{fmt_int(S['est'])} estudiantes · {fmt_pct(_div(S['n'], len(df_all)), 0)} de la cartera", "sp": mon["n"],
     "c": p["text"]},
    {"label": "En riesgo Alto (predicho)", "value": fmt_int(S["n_Alto"]), "tone": "alto", "icon": "🔴",
     "sub": f"{fmt_pct(S['pct_alto'])} de los créditos filtrados", "d": _delta("pct_alto", False), "sp": mon["alto"],
     "c": RISK_COLORS["Alto"]},
    {"label": "Mora real · Datacrédito", "value": fmt_pct(S["pct_mora"]), "tone": "medio", "icon": "🏦",
     "sub": f"{fmt_int(S['n_mora'])} créditos con reporte de mora", "d": _delta("pct_mora", False), "sp": mon["mora"],
     "c": RISK_COLORS["Medio"]},
    {"label": "Monto financiado total", "value": fmt_cop(S["monto"]), "tone": "accent", "icon": "💰",
     "sub": f"Ticket promedio {_nb(fmt_cop(S['ticket']))} · {_nb(fmt_cop(S['monto_Alto']))} en Alto",
     "d": _delta("monto", True), "sp": mon["monto"], "c": p["text"]},
]
for c_, k in zip(st.columns(4, gap="small"), kpis):
    h = kpi_card(k["label"], k["value"], k["sub"], tone=k["tone"], delta=k["d"][0], delta_dir=k["d"][1], icon=k["icon"],
                 help=f"Delta: últimos {N_CMP} meses vs. {N_CMP} meses previos (mismos filtros). Línea: últimos 12 meses.")
    c_.markdown(f"<div class='rs-kpi'>{h[:-6] + _spark(k['sp'], k['c']) + '</div>'}</div>", unsafe_allow_html=True)
st.write("")
for c_, r in zip(st.columns(3, gap="small"), RISK_ORDER):
    c_.markdown(_risk_card(r, S), unsafe_allow_html=True)

# ---- 2. En síntesis + hallazgos ----
F = _findings(dff, S)
HEADLINE = _headline(S, truth)
section("En síntesis", "Lectura automática de la cartera filtrada: se recalcula con cada cambio de filtro.",
        kicker="Lectura ejecutiva")
st.markdown(f"<div class='rs-head'><div class='q'>“</div><div><div class='k'>En síntesis</div>"
            f"<div class='t'>{HEADLINE}</div></div></div>", unsafe_allow_html=True)
for c_, f in zip(st.columns(len(F), gap="small"), F):
    c_.markdown(insight(f["title"], f["html"], tone=f["tone"], icon=f["icon"]), unsafe_allow_html=True)

# ---- 3. Evolución mensual + dona ----
section("Evolución de la cartera por riesgo predicho", "Créditos aprobados por mes según su riesgo y, arriba, la "
        "proporción en Alto: mire si la franja roja sube por encima del promedio.", kicker="Tendencia")
cc1, cc2 = st.columns([2.15, 1], gap="medium")
with cc1, st.container(border=True):
    fig_ev = _evolution_fig(dff)
    if fig_ev is None:
        empty_state("Sin fechas de aprobación", "El dataset activo no trae fechas válidas para la serie.", "🗓️")
    else:
        show_fig(fig_ev, key=f"{P}_evolucion", height=400)
with cc2, st.container(border=True):
    st.markdown("<div class='rs-t'>Distribución por riesgo</div><div class='rs-note' style='margin:2px 0 0'>Créditos "
                "según riesgo predicho.</div>", unsafe_allow_html=True)
    show_fig(_donut_fig(S), key=f"{P}_dona", height=230, legend=False)
    st.markdown("<div class='rs-legend'>" + "".join(
        f"<div class='rs-legend-row'><span class='sw' style='background:{RISK_COLORS[r]}'></span><span class='nm'>{r}</span>"
        f"<span class='v'>{fmt_int(S[f'n_{r}'])}</span><span class='s'>{fmt_pct(_div(S[f'n_{r}'], S['n']))}</span></div>"
        for r in RISK_ORDER) + "</div>", unsafe_allow_html=True)

# ---- 4. Top segmentos ----
section("Dónde se concentra el riesgo", f"Segmentos con al menos {MIN_N} créditos, ordenados por % en riesgo Alto. "
        "En rojo, los que superan el promedio del filtro (lift > 1×).", kicker="Concentración")
with st.container(border=True):
    dim_lbl = st.segmented_control("Dimensión de análisis", list(_DIMS), default="Segmento", key=f"{P}_dim",
                                   label_visibility="collapsed") or "Segmento"
    dim_col = _DIMS[dim_lbl]
    seg_all = _seg_table(dff, dim_col, MIN_N, S["pct_alto"])
    seg_top = seg_all.head(8)
    if seg_top.empty:
        empty_state("Ningún segmento cumple el n mínimo", f"No hay valores de {dim_lbl.lower()} con al menos {MIN_N} "
                    "créditos en el filtro.", "🧩")
    else:
        show_fig(_top_fig(seg_top, dim_col, S["pct_alto"]), key=f"{P}_top_{dim_col}",
                 height=max(240, 44 + 34 * len(seg_top)), legend=False)
        with st.expander(f"Ver detalle · tabla de {fmt_int(len(seg_all))} grupos y exportación", icon=":material/table_view:"):
            tbl = pd.DataFrame({dim_lbl: seg_all["segmento"], "Créditos": seg_all["creditos"].astype(int),
                                "En Alto": seg_all["n_alto"], "% Alto": (seg_all["pct_alto"] * 100).round(2),
                                "Lift": seg_all["lift"].round(2), "% mora Datacrédito": (seg_all["pct_mora"] * 100).round(2),
                                "Monto en Alto (M COP)": (seg_all["exposicion_alto"] / 1e6).round(1)})
            st.dataframe(tbl, hide_index=True, width="stretch", height=min(380, 38 + 35 * len(tbl)), column_config={
                "% Alto": st.column_config.ProgressColumn(format="%.1f %%", min_value=0,
                                                          max_value=max(float(tbl["% Alto"].max() or 1), 1)),
                "Lift": st.column_config.NumberColumn(format="%.2f×"),
                "% mora Datacrédito": st.column_config.NumberColumn(format="%.1f %%"),
                "Monto en Alto (M COP)": st.column_config.NumberColumn(format="$ %.1f M")})
            download_bar(tbl, f"segmentos_{dim_col}", key=f"{P}_seg_dl", label="Exportar")
        if can_access("segmentos"):
            st.page_link(PAGES["segmentos"][0], label="Profundizar en Segmentos y territorio",
                         icon=":material/arrow_forward:")

# ---- 5. Predicho vs observado ----
section("Predicho vs. observado", "¿Cuántos de los créditos que realmente resultaron Alto anticipó el modelo?",
        kicker="Confiabilidad")
if truth is None:
    st.markdown(insight("Sin riesgo observado en el dataset activo", "El archivo cargado no trae la columna <b>y_true</b>: "
                        "la concordancia no se puede calcular.", tone="info", icon="ℹ️"), unsafe_allow_html=True)
else:
    rec = truth["rec"]
    thr = (((load_artifacts() or {}).get("alto") or {}).get("rf") or {}).get("umbral_objetivo") or {}
    with st.container(border=True):
        v1, v2, v3 = st.columns([1, 1, 2.1], gap="medium")
        v1.markdown(f"<div class='rs-mini'><div class='l'>Recall de Alto · meta {fmt_pct(RECALL_ALTO_TARGET, 0)}</div>"
                    f"<div class='v'>{fmt_pct(rec)}</div><div class='rs-target'><i style='width:"
                    f"{0 if _isnan(rec) else min(rec, 1) * 100:.1f}%'></i><b style='left:{RECALL_ALTO_TARGET * 100:.0f}%'>"
                    f"</b></div><div class='s'>{fmt_int(truth['tp'])} de {fmt_int(truth['real_alto'])} Alto observados "
                    f"anticipados</div></div>", unsafe_allow_html=True)
        v2.markdown(f"<div class='rs-mini'><div class='l'>Concordancia global</div><div class='v'>"
                    f"{fmt_pct(truth['acc'])}</div><div class='s'>Precisión de Alto {fmt_pct(truth['prec'])} · "
                    f"{fmt_int(truth['n'])} créditos con riesgo observado</div></div>", unsafe_allow_html=True)
        with v3:
            low = not _isnan(rec) and rec < RECALL_ALTO_TARGET
            thr_txt = (f" Alertar cuando <b>P(Alto) ≥ {fmt_num(thr.get('umbral', 0.3), 2)}</b> eleva el recall a "
                       f"<b>{fmt_pct(thr.get('recall'))}</b> en prueba (precisión {fmt_pct(thr.get('precision'))}, alerta al "
                       f"{fmt_pct(thr.get('pct_alertas'), 0)} de la cartera).") if thr else ""
            st.markdown(insight("La regla argmax subdetecta el riesgo Alto" if low else "Recall de Alto dentro de la meta",
                                f"Hay <b>{fmt_int(truth['real_alto'] - truth['tp'])}</b> créditos Alto sin alerta con la "
                                f"clase más probable.{thr_txt}", tone="alto" if low else "bajo",
                                icon="⚠️" if low else "✅"), unsafe_allow_html=True)
            if can_access("modelo"):
                st.page_link(PAGES["modelo"][0], label="Ajustar el umbral de alerta en Desempeño del modelo",
                             icon=":material/tune:")

# ---- 6. Reporte + accesos rápidos ----
section("Siguientes pasos", "Descargue el reporte ejecutivo con los filtros actuales o continúe en las vistas "
        "operativas.", kicker="Acción")
doc = _report_html(S, [{"label": k["label"], "value": k["value"], "sub": k["sub"]} for k in kpis], F, seg_top, dim_lbl,
                   truth, active_chips(df_all), period_txt, HEADLINE)
fname = f"reporte_ejecutivo_SAT_{datetime.now():%Y%m%d_%H%M}.html"
with st.container(border=True):
    r1, r2, r3 = st.columns([3, 1, 1], vertical_alignment="center", gap="small")
    r1.markdown("<div class='rs-t'>📄 Reporte ejecutivo (ME-03)</div><div class='rs-d' style='min-height:0'>HTML "
                "autocontenido con KPI, hallazgos, top segmentos, filtros y fecha. Se imprime a PDF desde el "
                "navegador.</div>", unsafe_allow_html=True)
    r2.download_button("Descargar", doc.encode("utf-8"), file_name=fname, mime="text/html", icon=":material/download:",
                       type="primary", key=f"{P}_report_dl", width="stretch")
    if r3.button("Vista previa", icon=":material/visibility:", key=f"{P}_report_preview", width="stretch"):
        _preview_dialog(doc, fname)
links = [q for q in _LINKS if can_access(q[0])]
for c_, (k_, ttl, desc) in zip(st.columns(4, gap="small"), links):
    with c_, st.container(border=True):
        st.markdown(f"<div class='rs-t'>{esc(ttl)}</div><div class='rs-d'>{esc(desc)}</div>", unsafe_allow_html=True)
        st.page_link(PAGES[k_][0], label="Abrir", icon=PAGES[k_][2], width="stretch")

footer()
