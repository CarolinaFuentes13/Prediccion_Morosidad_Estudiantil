"""Componentes de interfaz reutilizables (HTML + Streamlit) del SAT.

Todas las páginas deben construir su interfaz con estas piezas para mantener
una estética homogénea: ``page_header``, ``section``, ``kpi_card``/``kpi_row``,
``insight``, ``risk_badge``, ``filter_chips``, ``empty_state``, ``download_bar``.
"""
from __future__ import annotations

import html
from datetime import date

import pandas as pd
import streamlit as st

from core.config import APP_NAME, PROGRAMA_ACADEMICO, RISK_ICONS, UNIVERSIDAD
from core.data import get_active_meta, to_excel_bytes


def esc(x) -> str:
    return html.escape(str(x))


def page_header(title: str, subtitle: str = "", eyebrow: str = "", meta: list[str] | None = None,
                highlight: str | None = None) -> None:
    """Encabezado tipo *hero* (fondo oscuro con acento amarillo)."""
    ds = get_active_meta()
    chips = [f"Actualizado {date.today().strftime('%d/%m/%Y')}", f"Fuente: {ds['name']}"] + list(meta or [])
    chips_html = "".join(f"<span>{esc(c)}</span>" for c in chips)
    if highlight:
        chips_html = f"<span class='hl'>{esc(highlight)}</span>" + chips_html
    st.markdown(
        f"""
        <div class="sat-hero">
          <div class="eyebrow">◆ {esc(eyebrow or APP_NAME)}</div>
          <h1>{esc(title)}</h1>
          <p>{esc(subtitle)}</p>
          <div class="meta">{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "", kicker: str = "") -> None:
    st.markdown(
        f"""<div class="sat-section"><div class="kicker">{esc(kicker)}</div><h3>{esc(title)}</h3>
        <p>{esc(subtitle)}</p></div>""",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", tone: str = "ink", delta: str | None = None,
             delta_dir: str = "flat", bar: float | None = None, icon: str = "", help: str | None = None) -> str:
    """Devuelve el HTML de una tarjeta KPI. ``tone``: alto|medio|bajo|accent|ink.
    ``delta_dir``: up (malo, rojo) | down (bueno, verde) | flat."""
    delta_html = f"<div class='delta {delta_dir}'>{esc(delta)}</div>" if delta else ""
    bar_html = ""
    if bar is not None:
        w = max(0.0, min(1.0, float(bar))) * 100
        bar_html = f"<div class='bar'><i style='width:{w:.1f}%'></i></div>"
    title_attr = f" title='{esc(help)}'" if help else ""
    return (f"<div class='sat-kpi tone-{tone}'{title_attr}><div class='lbl'>{esc(icon)} {esc(label)}</div>"
            f"<div class='val'>{esc(value)}</div><div class='sub'>{esc(sub)}</div>{delta_html}{bar_html}</div>")


def kpi_row(cards: list[str], gap: str = "small") -> None:
    cols = st.columns(len(cards), gap=gap)
    for c, h in zip(cols, cards):
        with c:
            st.markdown(h, unsafe_allow_html=True)


def insight(title: str, text: str, tone: str = "accent", icon: str = "💡") -> str:
    """HTML de una tarjeta de hallazgo. ``tone``: alto|medio|bajo|accent|info."""
    return (f"<div class='sat-insight tone-{tone}'><div class='ico'>{esc(icon)}</div>"
            f"<div><div class='ttl'>{esc(title)}</div><div class='txt'>{text}</div></div></div>")


def insight_row(items: list[str]) -> None:
    cols = st.columns(len(items))
    for c, h in zip(cols, items):
        with c:
            st.markdown(h, unsafe_allow_html=True)


def risk_badge(level) -> str:
    lv = str(level)
    cls = {"Alto": "alto", "Medio": "medio", "Bajo": "bajo"}.get(lv, "neutral")
    return f"<span class='sat-badge {cls}'>{RISK_ICONS.get(lv, '⚪')} {esc(lv)}</span>"


def badge(text: str, kind: str = "neutral") -> str:
    return f"<span class='sat-badge {kind}'>{esc(text)}</span>"


def filter_chips(chips: list[tuple[str, str]]) -> None:
    """Muestra los filtros activos como chips (label, valor)."""
    if not chips:
        st.markdown("<div class='sat-chips'><span class='sat-chip'>Sin filtros: toda la cartera</span></div>",
                    unsafe_allow_html=True)
        return
    html_chips = "".join(f"<span class='sat-chip'><b>{esc(k)}:</b> {esc(v)}</span>" for k, v in chips)
    st.markdown(f"<div class='sat-chips'>{html_chips}</div>", unsafe_allow_html=True)


def empty_state(title: str = "Sin resultados", text: str = "Ajusta los filtros para ver información.",
                icon: str = "🔎") -> None:
    st.markdown(f"<div class='sat-empty'><div class='big'>{esc(icon)}</div><h4>{esc(title)}</h4>"
                f"<div>{esc(text)}</div></div>", unsafe_allow_html=True)


def card(html_body: str) -> None:
    st.markdown(f"<div class='sat-card'>{html_body}</div>", unsafe_allow_html=True)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    d = df.copy()
    for c in d.columns:
        if isinstance(d[c].dtype, pd.CategoricalDtype):
            d[c] = d[c].astype(str)
    return d.to_csv(index=False).encode("utf-8-sig")


def download_bar(df: pd.DataFrame, basename: str, key: str, extra_sheets: dict[str, pd.DataFrame] | None = None,
                 label: str = "Exportar") -> None:
    """Botones de descarga CSV + Excel (DB-02)."""
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.download_button(f"⬇️ {label} CSV", _csv_bytes(df), file_name=f"{basename}.csv", mime="text/csv",
                           key=f"{key}_csv", width="stretch")
    with c2:
        try:
            sheets = {"datos": df, **(extra_sheets or {})}
            st.download_button(f"⬇️ {label} Excel", to_excel_bytes(sheets), file_name=f"{basename}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"{key}_xlsx", width="stretch")
        except Exception as exc:  # openpyxl ausente
            st.caption(f"Excel no disponible: {exc}")


def footer() -> None:
    st.markdown(
        f"<div class='sat-footer'><span>{esc(APP_NAME)} · {esc(PROGRAMA_ACADEMICO)} · {esc(UNIVERSIDAD)}</span>"
        f"<span>Nombres anonimizados · Uso exclusivo para gestión preventiva de cartera</span></div>",
        unsafe_allow_html=True,
    )
