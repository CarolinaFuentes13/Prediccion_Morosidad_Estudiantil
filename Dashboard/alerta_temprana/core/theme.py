"""Sistema visual: paleta institucional (negro · amarillo Uniandes · blanco), CSS,
plantilla de Plotly y formateadores en español (COP, porcentajes, fechas).
"""
from __future__ import annotations

import math

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from core.config import RISK_COLORS

# ================= Tokens =================
YELLOW = "#FFD100"
YELLOW_SOFT = "#FFF3B0"
INK = "#0B0B0C"

PALETTES = {
    "light": {
        "bg": "#F6F6F3", "surface": "#FFFFFF", "surface_2": "#FAFAF7", "border": "#E7E5DF",
        "text": "#141414", "muted": "#6B6A64", "subtle": "#9A988F", "ink": INK,
        "accent": YELLOW, "accent_text": "#141414", "accent_soft": "#FFF6CC",
        "grid": "#ECEAE4", "shadow": "0 1px 2px rgba(16,16,16,.04), 0 8px 24px rgba(16,16,16,.06)",
        "hero_from": "#0B0B0C", "hero_to": "#26251F",
    },
    "dark": {
        "bg": "#0E1117", "surface": "#161A22", "surface_2": "#1B202A", "border": "#2A303C",
        "text": "#F2F2EE", "muted": "#A4A7AE", "subtle": "#7B7F88", "ink": "#F2F2EE",
        "accent": YELLOW, "accent_text": "#141414", "accent_soft": "#3A3310",
        "grid": "#262B35", "shadow": "0 1px 2px rgba(0,0,0,.3), 0 8px 24px rgba(0,0,0,.35)",
        "hero_from": "#15171C", "hero_to": "#2A2718",
    },
}
CATEGORICAL = ["#141414", "#FFD100", "#3E63DD", "#30A46C", "#E5484D", "#8E4EC6", "#F5A524", "#12A594",
               "#D6409F", "#6E6A5E"]
CATEGORICAL_DARK = ["#F2F2EE", "#FFD100", "#7C9CFF", "#46C487", "#FF6B70", "#B083F0", "#FFB547", "#2BC4B4",
                    "#F06BB8", "#B5B0A2"]
SEQ_RISK = ["#FFF9DB", "#FFE27A", "#FFC23D", "#F5873A", "#E5484D", "#9E1C22"]


def theme_mode() -> str:
    """Tema activo. El tablero está fijado en modo claro (``base = "light"`` en .streamlit/config.toml)."""
    return "light"


def pal() -> dict:
    return PALETTES[theme_mode()]


# ================= Plotly =================
def _build_template(mode: str) -> go.layout.Template:
    p = PALETTES[mode]
    colorway = CATEGORICAL_DARK if mode == "dark" else CATEGORICAL
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Inter, 'Segoe UI', Roboto, sans-serif", size=13, color=p["text"]),
        title=dict(font=dict(size=15, color=p["text"]), x=0, xanchor="left", y=0.98),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=colorway,
        margin=dict(l=8, r=8, t=40, b=8),
        hoverlabel=dict(bgcolor=p["surface"], bordercolor=p["border"],
                        font=dict(color=p["text"], family="Inter, sans-serif", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=p["muted"]), title_text=""),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=p["border"], tickcolor=p["border"],
                   tickfont=dict(color=p["muted"], size=11), title=dict(font=dict(color=p["muted"], size=12))),
        yaxis=dict(showgrid=True, gridcolor=p["grid"], zeroline=False, linecolor=p["border"],
                   tickfont=dict(color=p["muted"], size=11), title=dict(font=dict(color=p["muted"], size=12))),
        separators=",.",
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=p["muted"]))),
    )
    return t


pio.templates["sat_light"] = _build_template("light")
pio.templates["sat_dark"] = _build_template("dark")


def plotly_template() -> str:
    return "sat_dark" if theme_mode() == "dark" else "sat_light"


def style_fig(fig: go.Figure, height: int | None = None, legend: bool = True) -> go.Figure:
    """Aplica la plantilla institucional a una figura Plotly."""
    fig.update_layout(template=plotly_template(), showlegend=legend)
    if height:
        fig.update_layout(height=height)
    return fig


PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "locale": "es",
}


def show_fig(fig: go.Figure, key: str | None = None, height: int | None = None, legend: bool = True,
             on_select: str = "ignore", selection_mode=("points", "box", "lasso")):
    """st.plotly_chart con estilo y configuración homogéneos.

    Pase siempre un ``key`` único por página (Streamlit falla con dos gráficos idénticos sin key).
    Con ``on_select="rerun"`` devuelve la selección del usuario (eventos de clic/caja).
    """
    style_fig(fig, height=height, legend=legend)
    if key is None:
        import hashlib

        key = "fig_" + hashlib.md5(fig.to_json().encode()).hexdigest()[:12]
    return st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key, on_select=on_select,
                           selection_mode=selection_mode)


def risk_color_map() -> dict:
    return dict(RISK_COLORS)


# ================= Formateadores =================
def _es_number(x: float, decimals: int = 0) -> str:
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return _es_number(float(x), 0)


def fmt_num(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return _es_number(float(x), decimals)


def fmt_pct(x, decimals: int = 1) -> str:
    """0.1234 → '12,3 %'"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{_es_number(float(x) * 100, decimals)} %"


def fmt_cop(x, compact: bool = True) -> str:
    """Pesos colombianos. compact=True → '$ 1.234 M' / '$ 12,3 mil M'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    x = float(x)
    if not compact:
        return f"$ {_es_number(x, 0)}"
    a = abs(x)
    if a >= 1e12:
        return f"$ {_es_number(x / 1e12, 2)} B"
    if a >= 1e9:
        return f"$ {_es_number(x / 1e9, 1)} mil M"
    if a >= 1e6:
        return f"$ {_es_number(x / 1e6, 1)} M"
    if a >= 1e3:
        return f"$ {_es_number(x / 1e3, 0)} mil"
    return f"$ {_es_number(x, 0)}"


def fmt_delta_pp(x, decimals: int = 1) -> str:
    """Diferencia en puntos porcentuales: 0.012 → '+1,2 pp'"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    sign = "+" if x >= 0 else "−"
    return f"{sign}{_es_number(abs(x) * 100, decimals)} pp"


MESES_CORTOS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def fmt_date(d) -> str:
    try:
        return f"{d.day:02d} {MESES_CORTOS[d.month - 1]} {d.year}"
    except Exception:
        return "—"


def fmt_period(p: str) -> str:
    """'2024-03' → 'mar 2024'"""
    try:
        y, m = str(p).split("-")[:2]
        return f"{MESES_CORTOS[int(m) - 1]} {y}"
    except Exception:
        return str(p)


# ================= CSS =================
def inject_css():
    p = pal()
    mode = theme_mode()
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {{
      --sat-bg: {p['bg']}; --sat-surface: {p['surface']}; --sat-surface-2: {p['surface_2']};
      --sat-border: {p['border']}; --sat-text: {p['text']}; --sat-muted: {p['muted']};
      --sat-subtle: {p['subtle']}; --sat-accent: {p['accent']}; --sat-accent-soft: {p['accent_soft']};
      --sat-accent-text: {p['accent_text']}; --sat-shadow: {p['shadow']};
      --sat-alto: {RISK_COLORS['Alto']}; --sat-medio: {RISK_COLORS['Medio']}; --sat-bajo: {RISK_COLORS['Bajo']};
      --sat-radius: 16px;
    }}
    html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stText, button, input, textarea, select {{
      font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }}
    [data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] {{ background: var(--sat-bg); }}
    .block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1480px; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    button[kind="primary"], button[data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] button {{
      background: {'#FFD100' if mode == 'dark' else '#141414'} !important; color: {'#141414' if mode == 'dark' else '#FFFFFF'} !important;
      border: 1px solid {'#FFD100' if mode == 'dark' else '#141414'} !important; }}
    button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{ filter: brightness(1.08); box-shadow: 0 0 0 3px rgba(255,209,0,.35); }}
    [data-testid="stSidebar"] {{ border-right: 1px solid var(--sat-border); }}
    [data-testid="stSidebar"] [data-testid="stImage"] img {{ {'filter: invert(1) hue-rotate(180deg) brightness(1.2);' if mode == 'dark' else ''} }}
    h1, h2, h3 {{ letter-spacing: -0.02em; }}

    /* ---------- Hero ---------- */
    .sat-hero {{
      position: relative; overflow: hidden; border-radius: 22px; padding: 26px 30px 24px 30px;
      background: linear-gradient(120deg, {p['hero_from']} 0%, {p['hero_to']} 100%); color: #F7F7F2;
      box-shadow: var(--sat-shadow); margin-bottom: 18px;
    }}
    .sat-hero:after {{
      content: ""; position: absolute; right: -80px; top: -80px; width: 280px; height: 280px; border-radius: 50%;
      background: radial-gradient(circle, rgba(255,209,0,.35) 0%, rgba(255,209,0,0) 70%);
    }}
    .sat-hero .eyebrow {{ display:inline-flex; gap:8px; align-items:center; font-size: 12px; font-weight: 700;
      letter-spacing: .12em; text-transform: uppercase; color: {YELLOW}; }}
    .sat-hero h1 {{ font-family: 'Space Grotesk', 'Inter', sans-serif; font-size: 30px; line-height: 1.15;
      margin: 8px 0 6px 0; color: #FFFFFF; font-weight: 700; padding: 0; }}
    .sat-hero p {{ color: #CFCFC6; margin: 0; font-size: 15px; max-width: 900px; }}
    .sat-hero .meta {{ margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }}
    .sat-hero .meta span {{ font-size: 12px; padding: 4px 10px; border-radius: 999px;
      background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); color: #EDEDE6; }}
    .sat-hero .meta span.hl {{ background: {YELLOW}; color: #141414; border-color: {YELLOW}; font-weight: 700; }}

    /* ---------- Section ---------- */
    .sat-section {{ margin: 26px 0 10px 0; }}
    .sat-section .kicker {{ font-size: 11px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase;
      color: var(--sat-muted); display:flex; align-items:center; gap:8px; }}
    .sat-section .kicker:before {{ content: ""; width: 18px; height: 3px; border-radius: 3px; background: var(--sat-accent); }}
    .sat-section h3 {{ margin: 4px 0 2px 0; font-size: 21px; font-weight: 700; color: var(--sat-text); padding: 0; }}
    .sat-section p {{ margin: 0; color: var(--sat-muted); font-size: 14px; }}

    /* ---------- KPI cards ---------- */
    .sat-kpi {{ background: var(--sat-surface); border: 1px solid var(--sat-border); border-radius: var(--sat-radius);
      padding: 16px 18px 14px 18px; box-shadow: var(--sat-shadow); height: 100%; position: relative; overflow: hidden; }}
    .sat-kpi .lbl {{ font-size: 12.5px; font-weight: 600; color: var(--sat-muted); display:flex; gap:6px; align-items:center; }}
    .sat-kpi .val {{ font-family: 'Space Grotesk', 'Inter', sans-serif; font-size: 28px; font-weight: 700;
      color: var(--sat-text); line-height: 1.15; margin-top: 6px; letter-spacing: -0.02em; }}
    .sat-kpi .sub {{ font-size: 12.5px; color: var(--sat-muted); margin-top: 4px; }}
    .sat-kpi .delta {{ display:inline-block; font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 999px; margin-top: 6px; }}
    .sat-kpi .delta.up {{ background: rgba(229,72,77,.12); color: var(--sat-alto); }}
    .sat-kpi .delta.down {{ background: rgba(48,164,108,.14); color: var(--sat-bajo); }}
    .sat-kpi .delta.flat {{ background: var(--sat-surface-2); color: var(--sat-muted); }}
    .sat-kpi.tone-alto {{ border-top: 4px solid var(--sat-alto); }}
    .sat-kpi.tone-medio {{ border-top: 4px solid var(--sat-medio); }}
    .sat-kpi.tone-bajo {{ border-top: 4px solid var(--sat-bajo); }}
    .sat-kpi.tone-accent {{ border-top: 4px solid var(--sat-accent); }}
    .sat-kpi.tone-ink {{ border-top: 4px solid var(--sat-text); }}
    .sat-kpi .bar {{ height: 6px; border-radius: 6px; background: var(--sat-surface-2); margin-top: 10px; overflow: hidden; }}
    .sat-kpi .bar > i {{ display:block; height: 100%; border-radius: 6px; background: var(--sat-accent); }}

    /* ---------- Cards / insights ---------- */
    .sat-card {{ background: var(--sat-surface); border: 1px solid var(--sat-border); border-radius: var(--sat-radius);
      padding: 16px 18px; box-shadow: var(--sat-shadow); }}
    .sat-insight {{ border-radius: 14px; padding: 14px 16px; border: 1px solid var(--sat-border);
      background: var(--sat-surface); display:flex; gap: 12px; align-items:flex-start; box-shadow: var(--sat-shadow); height: 100%; }}
    .sat-insight .ico {{ font-size: 20px; line-height: 1; }}
    .sat-insight .ttl {{ font-weight: 700; font-size: 14px; color: var(--sat-text); margin-bottom: 2px; }}
    .sat-insight .txt {{ font-size: 13.5px; color: var(--sat-muted); line-height: 1.45; }}
    .sat-insight.tone-alto {{ border-left: 4px solid var(--sat-alto); }}
    .sat-insight.tone-medio {{ border-left: 4px solid var(--sat-medio); }}
    .sat-insight.tone-bajo {{ border-left: 4px solid var(--sat-bajo); }}
    .sat-insight.tone-accent {{ border-left: 4px solid var(--sat-accent); }}
    .sat-insight.tone-info {{ border-left: 4px solid #3E63DD; }}

    /* ---------- Badges / chips ---------- */
    .sat-badge {{ display:inline-flex; align-items:center; gap:6px; font-size: 12px; font-weight: 700;
      padding: 3px 10px; border-radius: 999px; border: 1px solid transparent; white-space: nowrap; }}
    .sat-badge.alto {{ background: rgba(229,72,77,.12); color: var(--sat-alto); border-color: rgba(229,72,77,.35); }}
    .sat-badge.medio {{ background: rgba(245,165,36,.14); color: #B7700A; border-color: rgba(245,165,36,.4); }}
    .sat-badge.bajo {{ background: rgba(48,164,108,.12); color: var(--sat-bajo); border-color: rgba(48,164,108,.35); }}
    .sat-badge.neutral {{ background: var(--sat-surface-2); color: var(--sat-muted); border-color: var(--sat-border); }}
    .sat-badge.accent {{ background: var(--sat-accent); color: #141414; }}
    .sat-chips {{ display:flex; flex-wrap:wrap; gap:6px; margin: 2px 0 8px 0; }}
    .sat-chip {{ font-size: 12px; padding: 3px 10px; border-radius: 999px; background: var(--sat-accent-soft);
      color: var(--sat-text); border: 1px solid var(--sat-border); }}
    .sat-chip b {{ font-weight: 700; }}

    /* ---------- Sidebar brand ---------- */
    .sat-brand {{ display:flex; flex-direction:column; gap:2px; margin: 2px 0 8px 0; }}
    .sat-brand .name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 17px; color: var(--sat-text); }}
    .sat-brand .sub {{ font-size: 12px; color: var(--sat-muted); }}
    .sat-user {{ display:flex; align-items:center; gap:10px; padding: 10px 12px; border-radius: 12px;
      background: var(--sat-surface-2); border: 1px solid var(--sat-border); margin-bottom: 6px; }}
    .sat-user .avatar {{ width: 32px; height: 32px; border-radius: 50%; background: {YELLOW}; color: #141414;
      display:flex; align-items:center; justify-content:center; font-weight: 800; font-size: 13px; }}
    .sat-user .who {{ font-size: 13px; font-weight: 700; color: var(--sat-text); line-height: 1.2; }}
    .sat-user .role {{ font-size: 11.5px; color: var(--sat-muted); }}

    /* ---------- Misc ---------- */
    .sat-footer {{ margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--sat-border); color: var(--sat-subtle);
      font-size: 12px; display:flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }}
    .sat-muted {{ color: var(--sat-muted); }}
    .sat-empty {{ text-align:center; padding: 40px 20px; border: 1px dashed var(--sat-border); border-radius: var(--sat-radius);
      color: var(--sat-muted); background: var(--sat-surface); }}
    .sat-empty .big {{ font-size: 34px; }}
    div[data-testid="stMetric"] {{ background: var(--sat-surface); border: 1px solid var(--sat-border);
      border-radius: 14px; padding: 12px 14px; box-shadow: var(--sat-shadow); }}
    div[data-testid="stExpander"] details {{ border-radius: 14px; border-color: var(--sat-border); background: var(--sat-surface); }}
    div[data-testid="stTabs"] button[role="tab"] p {{ font-weight: 600; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: var(--sat-radius); }}
    .stDownloadButton button, .stButton button {{ border-radius: 10px; font-weight: 600; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
