"""Desempeño del modelo — ¿qué tan bien clasifica el riesgo y con qué regla conviene alertar?

Página de auditoría del clasificador (EV-02, DE-03, MO-02, MO-03). Se organiza en cuatro
pestañas pensadas para exponerse en orden:

1. Comparación de modelos: por qué se eligió el Random Forest.
2. Reporte por clase: precisión, recall, F1, matriz de confusión y curvas ROC.
3. Umbral de alerta: cómo cambia el recall de la clase Alto al mover el corte de P(Alto).
4. Variables: importancia, alerta de gobierno y ficha técnica del modelo.

No usa los filtros globales: todas las cifras salen de los artefactos precalculados sobre la
partición de prueba (``core.metrics.load_artifacts()``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.components import badge, esc, footer, insight, insight_row, kpi_card, kpi_row, page_header, section
from core.config import (BORUTA_FEATURES, FIELD_LABELS, ID_LIKE_FEATURES, MLFLOW_REPORTED, MODEL_LABELS,
                         PACKAGED_MODEL, RECALL_ALTO_TARGET, RISK_COLORS, RISK_ORDER, SELECTED_MODEL)
from core.metrics import artifacts_or_warn
from core.theme import fmt_int, fmt_num, fmt_pct, pal, show_fig

P = "modelo"

_CSS = """
<style>
.md-note{font-size:12.5px;color:var(--sat-muted);line-height:1.5;margin:2px 0 8px}
.md-note b{color:var(--sat-text)}
.md-box{border:1px dashed var(--sat-border);border-radius:12px;padding:12px 16px;background:var(--sat-surface-2);
  font-size:13px;line-height:1.55;color:var(--sat-text)}
.md-box .ttl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--sat-muted);font-weight:700;
  margin-bottom:6px}
.md-chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}
.md-chip{font-size:12px;padding:4px 10px;border-radius:999px;border:1px solid var(--sat-border);
  background:var(--sat-surface);color:var(--sat-text)}
.md-chip.id{border-color:#E5484D;color:#E5484D;font-weight:600}
.md-policy{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px}
.md-policy>div{border:1px solid var(--sat-border);border-radius:12px;padding:12px 14px;background:var(--sat-surface)}
.md-policy .lv{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}
.md-policy .rule{font-size:18px;font-weight:700;color:var(--sat-text);margin:2px 0}
.md-policy .txt{font-size:12.5px;color:var(--sat-muted);line-height:1.45}
@media (max-width:760px){.md-policy{grid-template-columns:1fr}}
</style>
"""

_METRICS = [  # (clave en artefacto, etiqueta)
    ("auc_macro", "AUC macro"),
    ("accuracy", "Accuracy"),
    ("precision_macro", "Precisión macro"),
    ("recall_macro", "Recall macro"),
    ("f1_macro", "F1 macro"),
    ("recall_alto", "Recall Alto"),
    ("auc_alto", "AUC Alto"),
]


def _n(x, d: int = 3) -> str:
    """Número con coma decimal (0,824)."""
    try:
        return "—" if x is None or not np.isfinite(float(x)) else fmt_num(float(x), d)
    except (TypeError, ValueError):
        return "—"


def _label(var: str) -> str:
    return FIELD_LABELS.get(var, var.replace("_", " ").capitalize())


# ======================================================================================
# Datos
# ======================================================================================
st.markdown(_CSS, unsafe_allow_html=True)
art = artifacts_or_warn()
if not art or not art.get("modelos"):
    page_header("Desempeño del modelo", "No hay artefactos del modelo para mostrar.", eyebrow="Modelo")
    footer()
    st.stop()

models = {m["key"]: m for m in art["modelos"]}
sel_key = art.get("modelo_seleccionado", SELECTED_MODEL)
if sel_key not in models:
    sel_key = next(iter(models))
sel = models[sel_key]
sm = sel["metricas"]
sel_name = MODEL_LABELS.get(sel_key, sel["nombre"])
alto = (art.get("alto") or {}).get(sel_key)

page_header(
    "Desempeño del modelo",
    "Qué tan bien separa el modelo los niveles de riesgo y qué regla de alerta cumple el compromiso con Cartera.",
    eyebrow="Modelo · Evaluación",
    highlight=f"{sel_name} · AUC macro {_n(sm['auc_macro'])}",
    meta=[f"Prueba: {fmt_int(sel['n_test'])} créditos", "Partición estratificada 80/20"],
)

# ======================================================================================
# KPI del modelo seleccionado
# ======================================================================================
gap_alto = sm["recall_alto"] - RECALL_ALTO_TARGET
kpi_row([
    kpi_card("AUC macro", _n(sm["auc_macro"]), "Capacidad de ordenar el riesgo", tone="accent", bar=sm["auc_macro"],
             help="Área bajo la curva ROC, promedio one-vs-rest de las 3 clases."),
    kpi_card("Accuracy", _n(sm["accuracy"]), "Aciertos sobre el total", bar=sm["accuracy"]),
    kpi_card("Precisión macro", _n(sm["precision_macro"]), "De lo alertado, cuánto es real", bar=sm["precision_macro"]),
    kpi_card("Recall macro", _n(sm["recall_macro"]), "De lo real, cuánto se detecta", bar=sm["recall_macro"]),
    kpi_card("F1 macro · MO-02", _n(sm["f1_macro"]), "Balance precisión–recall", bar=sm["f1_macro"]),
    kpi_card("Recall Alto · DE-03", _n(sm["recall_alto"]), f"Meta {_n(RECALL_ALTO_TARGET, 2)} con regla argmax",
             tone="alto" if gap_alto < 0 else "bajo", bar=sm["recall_alto"],
             delta=f"{fmt_num(gap_alto * 100, 1)} pp vs meta", delta_dir="up" if gap_alto < 0 else "down"),
])
st.markdown(
    "<div class='md-note'>Cifras del modelo seleccionado sobre la partición de prueba. El recall de Alto con la "
    "regla <b>argmax</b> (clase más probable) no alcanza la meta; la pestaña <b>Umbral de alerta</b> muestra "
    "cómo se cumple bajando el corte de P(Alto).</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["⚖️ Comparación de modelos", "📋 Reporte por clase (EV-02)",
                                  "🎚️ Umbral de alerta (DE-03)", "🧬 Variables (MO-03)"])

# ======================================================================================
# 1. Comparación de modelos
# ======================================================================================
with tab1:
    section("Cinco modelos, una misma partición de prueba", "El mejor valor de cada métrica aparece resaltado.",
            kicker="Comparación")
    rows = []
    for k, m in models.items():
        name = MODEL_LABELS.get(k, m["nombre"])
        rows.append({"Modelo": ("★ " if k == sel_key else "") + name,
                     **{lab: m["metricas"].get(key, np.nan) for key, lab in _METRICS}})
    cmp = pd.DataFrame(rows)
    num_cols = [lab for _, lab in _METRICS]
    p = pal()

    def _hl(col: pd.Series):
        best = col.max()
        return [f"background-color:{p['accent_soft']};font-weight:700" if v == best else "" for v in col]

    styled = (cmp.style.apply(_hl, subset=num_cols)
              .format({c: (lambda v: _n(v)) for c in num_cols}))
    st.dataframe(styled, hide_index=True, width="stretch")
    st.caption("★ modelo seleccionado. Recall Alto y AUC Alto evalúan solo la clase Alto (one-vs-rest).")

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("<div class='md-note'>Mire la barra roja: los modelos XGBoost/LightGBM ordenan bien "
                        "(AUC alto) pero casi no detectan a los estudiantes de riesgo Alto.</div>",
                        unsafe_allow_html=True)
            names = [MODEL_LABELS.get(k, m["nombre"]) for k, m in models.items()]
            fig = go.Figure()
            for key, lab, color in [("auc_macro", "AUC macro", p["text"]), ("recall_macro", "Recall macro", "#8E8C84"),
                                    ("recall_alto", "Recall Alto", RISK_COLORS["Alto"])]:
                vals = [m["metricas"][key] for m in models.values()]
                fig.add_bar(x=names, y=vals, name=lab, marker_color=color, text=[_n(v, 2) for v in vals],
                            textposition="outside", cliponaxis=False,
                            customdata=[_n(v) for v in vals],
                            hovertemplate="%{x}<br>" + lab + ": %{customdata}<extra></extra>")
            fig.add_scatter(x=[names[0], names[-1]], y=[RECALL_ALTO_TARGET] * 2, mode="lines",
                            name="Meta recall Alto 0,75", hoverinfo="skip",
                            line=dict(color=RISK_COLORS["Alto"], dash="dash", width=1.5))
            fig.update_layout(barmode="group", bargap=0.25, yaxis=dict(title="Valor de la métrica", range=[0, 1.05]),
                              xaxis_title=None)
            show_fig(fig, key=f"{P}_cmp_bars", height=380)
    with c2:
        xgb = models.get("xgb", {}).get("metricas", {})
        why = (f"El <b>{esc(sel_name)}</b> logra el mejor <b>recall macro ({_n(sm['recall_macro'])})</b> y el mejor "
               f"<b>recall Alto ({_n(sm['recall_alto'])})</b>, con un AUC macro competitivo ({_n(sm['auc_macro'])}).")
        if xgb:
            why += (f" XGBoost tiene más AUC ({_n(xgb['auc_macro'])}) pero detecta solo el "
                    f"<b>{fmt_pct(xgb['recall_alto'])}</b> de los casos Alto: no sirve para alertar.")
        st.markdown(insight("¿Por qué Random Forest?", why, tone="accent", icon="🏆"), unsafe_allow_html=True)
        st.markdown(insight("Accuracy engaña con clases desbalanceadas",
                            "Con ~88 % de créditos Bajo, un modelo que casi siempre dice «Bajo» ya tiene accuracy "
                            "alto. Por eso se decide con recall macro y recall Alto.", tone="info", icon="⚠️"),
                    unsafe_allow_html=True)
        rep = MLFLOW_REPORTED.get(sel_key, {})
        if rep:
            st.markdown(
                "<div class='md-box'><div class='ttl'>Databricks + MLflow · según el reporte</div>"
                f"RF optimizado: accuracy <b>{_n(rep.get('accuracy'))}</b> · AUC macro <b>{_n(rep.get('auc_macro'))}"
                f"</b> · precisión macro <b>{_n(rep.get('precision_macro'))}</b>. Cifras reportadas por el equipo en "
                "otra corrida; no se recalculan aquí.</div>",
                unsafe_allow_html=True,
            )

# ======================================================================================
# 2. Reporte por clase
# ======================================================================================
with tab2:
    section("¿En qué clase acierta y en cuál se equivoca?", "Métricas one-vs-rest por nivel de riesgo.",
            kicker="EV-02 · Reporte por clase")
    pc = pd.DataFrame(sel["por_clase"])
    pc = pc.set_index("clase").reindex(RISK_ORDER).reset_index()
    pc_view = pd.DataFrame({
        "Clase": pc["clase"],
        **{lab: [_n(v) for v in pc[col]] for col, lab in [("precision", "Precisión"), ("recall", "Recall"),
                                                          ("f1", "F1"), ("auc", "AUC")]},
        "Soporte (real)": [fmt_int(v) for v in pc["soporte"]], "Predichos": [fmt_int(v) for v in pc["predichos"]],
    })
    st.dataframe(pc_view, hide_index=True, width="stretch")

    by = {r["clase"]: r for r in sel["por_clase"]}
    if all(c in by for c in RISK_ORDER):
        insight_row([
            insight("Bajo: sólido", f"F1 {_n(by['Bajo']['f1'])}; es la clase mayoritaria "
                    f"({fmt_int(by['Bajo']['soporte'])} casos).", tone="bajo", icon="🟢"),
            insight("Alto: el reto", f"Detecta el {fmt_pct(by['Alto']['recall'])} de los Alto y acierta en el "
                    f"{fmt_pct(by['Alto']['precision'])} de sus alertas.", tone="alto", icon="🔴"),
            insight("Medio: muy escaso", f"Solo {fmt_int(by['Medio']['soporte'])} casos en prueba; AUC "
                    f"{_n(by['Medio']['auc'])} pero métricas inestables.", tone="medio", icon="🟠"),
        ])
        st.write("")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            mode = st.segmented_control("Ver matriz como", ["Conteos", "% por fila"], default="% por fila",
                                        key=f"{P}_cm_mode") or "% por fila"
            cm = np.asarray(sel["confusion"], dtype=float)
            row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            z = row_pct if mode == "% por fila" else cm
            txt = [[(fmt_pct(row_pct[i, j]) if mode == "% por fila" else fmt_int(cm[i, j])) for j in range(3)]
                   for i in range(3)]
            hover = [[f"Real {RISK_ORDER[i]} → predicho {RISK_ORDER[j]}<br>{fmt_int(cm[i, j])} créditos "
                      f"({fmt_pct(row_pct[i, j])} de la fila)" for j in range(3)] for i in range(3)]
            fig = go.Figure(go.Heatmap(
                z=row_pct, x=[f"Pred. {c}" for c in RISK_ORDER], y=[f"Real {c}" for c in RISK_ORDER],
                text=txt, texttemplate="%{text}", customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                colorscale=[[0, p["surface_2"]], [0.5, "#FFE27A"], [1, "#F5873A"]], zmin=0, zmax=1,
                showscale=False, xgap=3, ygap=3, textfont=dict(size=15)))
            fig.update_yaxes(autorange="reversed")
            show_fig(fig, key=f"{P}_cm", height=340, legend=False)
            st.markdown("<div class='md-note'>La diagonal son aciertos. El color siempre refleja el % por fila "
                        "(recall), para que la clase Bajo no opaque a las demás.</div>", unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown("<div class='md-note'>Curvas ROC one-vs-rest: cuanto más se acercan a la esquina superior "
                        "izquierda, mejor separa esa clase del resto.</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", hoverinfo="skip",
                            line=dict(color=p["subtle"], dash="dot", width=1))
            for c in RISK_ORDER:
                r = sel.get("roc", {}).get(c)
                if not r:
                    continue
                fig.add_scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{c} · AUC {_n(r['auc'])}",
                                line=dict(color=RISK_COLORS[c], width=2.5),
                                hovertemplate=f"{c}<br>FPR %{{x:.2f}} · TPR %{{y:.2f}}<extra></extra>")
            fig.update_layout(xaxis=dict(title="Tasa de falsos positivos", range=[0, 1]),
                              yaxis=dict(title="Tasa de verdaderos positivos (recall)", range=[0, 1.02]))
            show_fig(fig, key=f"{P}_roc", height=392)

# ======================================================================================
# 3. Umbral de alerta (DE-03)
# ======================================================================================
with tab3:
    section("Mover el corte de P(Alto) para cumplir DE-03",
            f"Se alerta un crédito si P(Alto) ≥ umbral. Meta del compromiso: recall Alto ≥ {_n(RECALL_ALTO_TARGET, 2)}.",
            kicker="DE-03 · Umbral de alerta")
    if not alto:
        st.info("No hay barrido de umbrales para este modelo en los artefactos.", icon="ℹ️")
    else:
        sw = pd.DataFrame(alto["barrido"])
        sw_plot = sw.copy()  # con < 30 alertas la precisión es ruido: se oculta en la curva
        sw_plot.loc[sw_plot["alertas"] < 30, ["precision", "f1"]] = np.nan
        obj = alto.get("umbral_objetivo") or {}
        best_f1 = alto.get("umbral_f1") or {}
        argmax = alto.get("regla_argmax") or {}
        t_def = float(obj.get("umbral", 0.30))
        t = st.slider("Umbral de P(Alto)", min_value=0.02, max_value=0.98, value=round(t_def, 2), step=0.01,
                      key=f"{P}_thr", help="Probabilidad mínima de la clase Alto para generar una alerta.")
        row = sw.iloc[int((sw["umbral"] - t).abs().idxmin())]
        ok = row["recall"] >= RECALL_ALTO_TARGET
        kpi_row([
            kpi_card("Recall Alto", _n(row["recall"]), f"{fmt_int(row['tp'])} de {fmt_int(row['tp'] + row['fn'])} "
                     "casos Alto detectados", tone="bajo" if ok else "alto", bar=row["recall"],
                     delta="Cumple DE-03" if ok else f"Faltan {fmt_num((RECALL_ALTO_TARGET - row['recall']) * 100, 1)} pp",
                     delta_dir="down" if ok else "up"),
            kpi_card("Precisión Alto", _n(row["precision"]), f"{fmt_int(row['fp'])} falsas alarmas",
                     bar=row["precision"]),
            kpi_card("F1 Alto", _n(row["f1"]), f"Máximo {_n(best_f1.get('f1'))} con {_n(best_f1.get('umbral'), 2)}",
                     bar=row["f1"]),
            kpi_card("Cartera alertada", fmt_pct(row["pct_alertas"]), f"{fmt_int(row['alertas'])} de "
                     f"{fmt_int(alto['n'])} créditos", tone="accent", bar=row["pct_alertas"]),
        ])

        c1, c2 = st.columns([1.5, 1.1], gap="medium")
        with c1:
            with st.container(border=True):
                st.markdown("<div class='md-note'>Al bajar el umbral sube el recall (se detectan más Alto) pero cae "
                            "la precisión y crece la cartera alertada. La línea roja es la meta DE-03.</div>",
                            unsafe_allow_html=True)
                fig = go.Figure()
                for col, lab, color, dash in [("recall", "Recall Alto", RISK_COLORS["Alto"], "solid"),
                                              ("precision", "Precisión Alto", p["text"], "solid"),
                                              ("f1", "F1 Alto", "#3E63DD", "dot"),
                                              ("pct_alertas", "% cartera alertada", "#F5A524", "dash")]:
                    fig.add_scatter(x=sw_plot["umbral"], y=sw_plot[col], mode="lines", name=lab,
                                    line=dict(color=color, width=2.4, dash=dash),
                                    customdata=[fmt_pct(v) if col == "pct_alertas" else _n(v) for v in sw_plot[col]],
                                    hovertemplate="Umbral %{x:.2f}<br>" + lab + ": %{customdata}<extra></extra>")
                fig.add_hline(y=RECALL_ALTO_TARGET, line_dash="dash", line_color=RISK_COLORS["Alto"], line_width=1,
                              annotation_text="Meta 0,75", annotation_position="top right",
                              annotation_font_color=RISK_COLORS["Alto"])
                fig.add_vline(x=float(row["umbral"]), line_color=p["accent"], line_width=2)
                fig.add_scatter(x=[row["umbral"]], y=[row["recall"]], mode="markers", showlegend=False,
                                marker=dict(size=12, color=p["accent"], line=dict(color=p["text"], width=1.5)),
                                hovertemplate=f"Umbral elegido {_n(row['umbral'], 2)}<br>Recall {_n(row['recall'])}"
                                              "<extra></extra>")
                fig.update_layout(xaxis=dict(title="Umbral de P(Alto)", range=[0, 1]),
                                  yaxis=dict(title="Valor", range=[0, 1.02]))
                show_fig(fig, key=f"{P}_thr_curves", height=400)
        with c2:
            cmp_rows = [
                ("Regla argmax", argmax.get("recall"), argmax.get("precision"), argmax.get("pct_alertas")),
                (f"Elegido ({_n(row['umbral'], 2)})", row["recall"], row["precision"], row["pct_alertas"]),
            ]
            if best_f1:
                cmp_rows.append((f"Máx. F1 ({_n(best_f1['umbral'], 2)})", best_f1["recall"], best_f1["precision"],
                                 best_f1["pct_alertas"]))
            cdf = pd.DataFrame([(r, _n(a), _n(b), fmt_pct(c)) for r, a, b, c in cmp_rows],
                               columns=["Regla", "Recall", "Precisión", "% cartera"])
            st.dataframe(cdf, hide_index=True, width="stretch")
            st.markdown(insight(
                "Argmax subdetecta Alto",
                f"Con la clase más probable el recall Alto es <b>{_n(argmax.get('recall'))}</b>. Con P(Alto) ≥ "
                f"<b>{_n(obj.get('umbral'), 2)}</b> sube a <b>{_n(obj.get('recall'))}</b> y cumple DE-03, pero alerta "
                f"al <b>{fmt_pct(obj.get('pct_alertas'))}</b> de la cartera con precisión <b>{_n(obj.get('precision'), 2)}"
                "</b> (≈ 4 de cada 5 alertas son falsas).", tone="alto", icon="🎯"), unsafe_allow_html=True)

        if best_f1 and obj:
            st.markdown(
                "<div class='md-box'><div class='ttl'>Recomendación · política de alerta en dos niveles</div>"
                "<div class='md-policy'>"
                f"<div><div class='lv' style='color:{RISK_COLORS['Alto']}'>Nivel 1 · Gestión intensiva</div>"
                f"<div class='rule'>P(Alto) ≥ {_n(best_f1['umbral'], 2)}</div>"
                f"<div class='txt'>{fmt_pct(best_f1['pct_alertas'])} de la cartera · precisión "
                f"{_n(best_f1['precision'], 2)} · recall {_n(best_f1['recall'], 2)}. Llamada y acuerdo de pago.</div></div>"
                f"<div><div class='lv' style='color:{RISK_COLORS['Medio']}'>Nivel 2 · Preventivo de bajo costo</div>"
                f"<div class='rule'>{_n(obj['umbral'], 2)} ≤ P(Alto) &lt; {_n(best_f1['umbral'], 2)}</div>"
                f"<div class='txt'>Completa el recall a {_n(obj['recall'], 2)} (cumple DE-03) con recordatorios "
                f"automáticos (SMS / correo), sin saturar a los gestores.</div></div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

# ======================================================================================
# 4. Variables (MO-03)
# ======================================================================================
with tab4:
    imp = art.get("importancia") or {}
    items = pd.DataFrame(imp.get("items") or [])
    section("¿Qué variables usa más el modelo?", imp.get("fuente", "Importancia de variables del modelo."),
            kicker="MO-03 · Interpretabilidad")
    if items.empty:
        st.info("No hay importancias de variables en los artefactos.", icon="ℹ️")
    else:
        top = items.sort_values("importancia", ascending=False).head(12).iloc[::-1]
        is_id = top["variable"].isin(ID_LIKE_FEATURES)
        share_id = items.loc[items["variable"].isin(ID_LIKE_FEATURES), "importancia"].sum() / items["importancia"].sum()
        c1, c2 = st.columns([1.5, 1], gap="medium")
        with c1:
            with st.container(border=True):
                st.markdown("<div class='md-note'>Top 12 por importancia Gini. En <b style='color:#E5484D'>rojo</b> "
                            "las variables tipo identificador.</div>", unsafe_allow_html=True)
                labels = [_label(v) + (" ⚠" if i else "") for v, i in zip(top["variable"], is_id)]
                fig = go.Figure(go.Bar(
                    x=top["importancia"], y=labels, orientation="h",
                    marker_color=[RISK_COLORS["Alto"] if i else p["subtle"] for i in is_id],
                    text=[fmt_pct(v) for v in top["importancia"]], textposition="outside", cliponaxis=False,
                    customdata=np.stack([top["variable"], top["tipo"], [fmt_pct(v) for v in top["importancia"]]], -1),
                    hovertemplate="%{customdata[0]} (%{customdata[1]})<br>Importancia: %{customdata[2]}<extra></extra>"))
                fig.update_layout(xaxis=dict(title="Importancia (reducción media de impureza)", tickformat=".0%",
                                             range=[0, float(top["importancia"].max()) * 1.2]),
                                  yaxis_title=None)
                show_fig(fig, key=f"{P}_imp", height=440, legend=False)
        with c2:
            st.markdown(insight(
                "Alerta de gobierno: posible memorización",
                f"Variables tipo identificador (llave del crédito, fecha de nacimiento, fecha de aprobación como "
                f"categoría) suman el <b>{fmt_pct(share_id)}</b> de la importancia. El modelo puede estar "
                "memorizando registros en vez de aprender patrones.", tone="alto", icon="🚩"),
                unsafe_allow_html=True)
            st.markdown(insight(
                "Acción recomendada",
                "Reentrenar <b>sin</b> estas variables (o transformarlas: edad, mes de aprobación) y validar con "
                "una <b>partición temporal</b> (entrenar en semestres pasados, probar en el más reciente).",
                tone="accent", icon="🔁"), unsafe_allow_html=True)
            st.markdown("<div class='md-note' style='margin-top:10px'><b>Variables confirmadas por Boruta</b> "
                        f"({len(BORUTA_FEATURES)}), según el reporte:</div>", unsafe_allow_html=True)
            chips = "".join(
                f"<span class='md-chip{' id' if b.lower() in ID_LIKE_FEATURES else ''}'>{esc(b)}</span>"
                for b in BORUTA_FEATURES)
            st.markdown(f"<div class='md-chips'>{chips}</div>", unsafe_allow_html=True)

    with st.expander("📄 Ficha del modelo (hiperparámetros y limitaciones)"):
        f1c, f2c = st.columns([1, 1.3], gap="medium")
        with f1c:
            hp = PACKAGED_MODEL.get("hiperparametros", {})
            st.markdown(f"**{PACKAGED_MODEL.get('algoritmo', '')}** · versión {PACKAGED_MODEL.get('version', '')} "
                        f"· {imp.get('n_arboles', hp.get('n_estimators', '—'))} árboles")
            st.dataframe(pd.DataFrame({"Hiperparámetro": list(hp), "Valor": [str(v) for v in hp.values()]}),
                         hide_index=True, width="stretch")
        with f2c:
            st.markdown(
                f"- **Preprocesamiento:** {PACKAGED_MODEL.get('preprocesamiento', '—')}\n"
                f"- **Partición:** {PACKAGED_MODEL.get('particion', '—')} · {fmt_int(sel['n_test'])} créditos evaluados\n"
                f"- **Salida:** clase de riesgo (Alto / Medio / Bajo) y probabilidad por clase\n\n"
                "**Limitaciones**\n"
                "- La clase Medio es muy escasa (< 1 %): sus métricas son inestables.\n"
                "- Precisión de Alto baja: cada alerta debe gestionarse como señal preventiva, no como sentencia.\n"
                "- Variables tipo identificador con alta importancia: riesgo de memorización / fuga.\n"
                "- La partición aleatoria (no temporal) puede sobrestimar el desempeño futuro.\n"
                f"- El AUC por semestre cae bajo {_n(0.80, 2)} en casi todos los periodos (ME-01): monitorear y "
                "reentrenar.\n"
                "- Uso: apoyo a la gestión preventiva de cartera; no sustituye el criterio del gestor."
            )
            st.markdown(badge(f"Generado: {art.get('generado', '—')}", "neutral"), unsafe_allow_html=True)

footer()
