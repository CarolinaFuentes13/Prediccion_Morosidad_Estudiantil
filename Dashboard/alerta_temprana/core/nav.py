"""Registro único de páginas (clave → archivo, título, icono, grupo) y utilidades de navegación."""
from __future__ import annotations

import streamlit as st

PAGES = {
    "resumen": ("views/resumen.py", "Resumen ejecutivo", ":material/space_dashboard:", "Visión general"),
    "cola": ("views/cola_gestion.py", "Cola de gestión", ":material/format_list_numbered:", "Gestión de cartera"),
    "segmentos": ("views/segmentos.py", "Segmentos y territorio", ":material/donut_small:", "Análisis de riesgo"),
    "modelo": ("views/modelo.py", "Desempeño del modelo", ":material/model_training:", "Modelo"),
    "carga": ("views/carga_prediccion.py", "Cargar y predecir", ":material/upload_file:", "Operación"),
}
GROUP_ORDER = ["Visión general", "Gestión de cartera", "Análisis de riesgo", "Modelo", "Operación"]
# Páginas que usan los filtros globales de cartera en la barra lateral.
FILTER_PAGES = {"resumen", "cola", "segmentos"}

FICHA_KEY = "ficha_estudiante"  # id_estudiante que debe abrir la Ficha 360°


def goto(key: str) -> None:
    """Navega a otra página del tablero (si el rol tiene acceso)."""
    st.switch_page(PAGES[key][0])


def open_ficha(id_estudiante: str) -> None:
    st.session_state[FICHA_KEY] = str(id_estudiante)
    goto("ficha")


def can_access(key: str) -> bool:
    from core.config import PAGE_ACCESS

    u = st.session_state.get("sat_user") or {}
    return u.get("role") in PAGE_ACCESS.get(key, set())
