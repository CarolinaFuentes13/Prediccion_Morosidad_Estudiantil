"""SAT · Sistema de Alertas Tempranas de morosidad en créditos estudiantiles.

Ejecutar desde la raíz del repositorio:
    streamlit run Dashboard/alerta_temprana/app.py
"""
from __future__ import annotations

import streamlit as st

from core.auth import auth_mode, logout, require_login, user_badge
from core.config import APP_NAME, LOGO_PATH, PAGE_ACCESS, ASSETS_DIR
from core.data import get_active_df, get_active_meta, reset_active_df
from core.filters import render_sidebar_filters
from core.nav import FILTER_PAGES, GROUP_ORDER, PAGES
from core.theme import inject_css, theme_mode

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Sistema de alertas tempranas de morosidad · Universidad de los Andes"},
)
inject_css()

logo_dark = ASSETS_DIR / "logo_uni_dark.png"
st.logo(str(logo_dark if theme_mode() == "dark" and logo_dark.exists() else LOGO_PATH), size="large")

# Recordar la página pedida en la URL (enlace directo) para abrirla después del login:
# antes de autenticarse no se llama a st.navigation y Streamlit vuelve a la página por defecto.
NEXT_KEY = "_sat_next_page"
if "sat_user" not in st.session_state and NEXT_KEY not in st.session_state:
    try:
        from urllib.parse import urlparse

        _path = urlparse(st.context.url or "").path.rstrip("/").rsplit("/", 1)[-1]
    except Exception:
        _path = ""
    st.session_state[NEXT_KEY] = _path if _path in PAGES else ""

user = require_login()

# ---------- Navegación según rol ----------
pages_by_key = {
    k: st.Page(path, title=title, icon=icon, url_path=k, default=(k == "resumen"))
    for k, (path, title, icon, _) in PAGES.items()
    if user["role"] in PAGE_ACCESS[k]
}
groups: dict[str, list] = {}
for k, pg_ in pages_by_key.items():
    groups.setdefault(PAGES[k][3], []).append(pg_)
nav = st.navigation({g: groups[g] for g in GROUP_ORDER if g in groups})
current_key = next((k for k, p in pages_by_key.items() if p.url_path == nav.url_path), None)
_next = st.session_state.pop(NEXT_KEY, "")
if _next and _next in pages_by_key and _next != current_key:
    st.switch_page(pages_by_key[_next])

# ---------- Barra lateral ----------
with st.sidebar:
    user_badge(user)
    if st.button("Cerrar sesión", icon=":material/logout:", width="stretch", key="logout_btn"):
        logout()
        st.rerun()
    meta = get_active_meta()
    if meta["source"] != "base":
        st.info(f"📂 Datos activos: **{meta['name']}** · {meta.get('rows', 0):,} filas".replace(",", "."))
        st.button("Volver a la base oficial", icon=":material/restart_alt:", width="stretch",
                  on_click=reset_active_df, key="reset_ds_btn")
    if auth_mode() == "demo":
        st.caption("🧪 Modo demo · sin contraseñas configuradas")
    st.divider()

if current_key in FILTER_PAGES:
    render_sidebar_filters(get_active_df())

nav.run()
