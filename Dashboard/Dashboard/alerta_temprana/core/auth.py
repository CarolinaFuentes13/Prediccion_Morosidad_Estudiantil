"""Control de acceso con roles (pendiente de criticidad alta en el reporte).

Los usuarios se definen en ``.streamlit/secrets.toml`` (o en *Secrets* de
Streamlit Community Cloud) con contraseñas en SHA-256:

    [auth.users.cartera]
    name = "Equipo de Cartera"
    role = "cartera"            # admin | cartera | direccion | analitica
    password_sha256 = "…"       # python -c "import hashlib;print(hashlib.sha256(b'clave').hexdigest())"

Si no hay usuarios configurados, el tablero arranca en **modo demostración**:
se elige un rol sin contraseña y se muestra un aviso permanente.
"""
from __future__ import annotations

import hashlib
import hmac

import streamlit as st

from core.config import APP_NAME, APP_TAGLINE, LOGO_PATH, PROGRAMA_ACADEMICO, ROLES, UNIVERSIDAD

USER_KEY = "sat_user"


def _configured_users() -> dict:
    try:
        users = st.secrets.get("auth", {}).get("users", {})
        return {k: dict(v) for k, v in users.items()}
    except Exception:
        return {}


def auth_mode() -> str:
    return "secrets" if _configured_users() else "demo"


def current_user() -> dict | None:
    return st.session_state.get(USER_KEY)


def logout() -> None:
    st.session_state.pop(USER_KEY, None)


def _check(username: str, password: str) -> dict | None:
    users = _configured_users()
    u = users.get(username.strip().lower()) or users.get(username.strip())
    if not u:
        return None
    expected = str(u.get("password_sha256", "")).lower()
    got = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if expected and hmac.compare_digest(expected, got):
        role = u.get("role", "direccion")
        return {"username": username.strip(), "name": u.get("name", username), "role": role if role in ROLES else "direccion",
                "mode": "secrets"}
    return None


def login_screen() -> None:
    """Pantalla de inicio de sesión (se muestra en lugar del tablero)."""
    st.markdown(
        """<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none;}
        .block-container {max-width: 1100px; padding-top: 3rem;}</style>""",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="sat-hero" style="min-height: 430px; padding: 34px 34px;">
              <div class="eyebrow">◆ {APP_NAME}</div>
              <h1 style="font-size:34px;">Anticipar la mora<br/>antes de que ocurra.</h1>
              <p>{APP_TAGLINE}</p>
              <div class="meta" style="margin-top:22px;">
                <span class="hl">Random Forest · AUC 0,82</span><span>14.660 créditos</span><span>147 reglas de calidad</span>
              </div>
              <p style="margin-top:26px; font-size:13px;">{PROGRAMA_ACADEMICO} · {UNIVERSIDAD}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=190)
        st.markdown("### Iniciar sesión")
        if auth_mode() == "secrets":
            with st.form("login", border=True):
                user = st.text_input("Usuario", placeholder="p. ej. cartera")
                pwd = st.text_input("Contraseña", type="password")
                ok = st.form_submit_button("Entrar", type="primary", width="stretch")
            if ok:
                u = _check(user, pwd)
                if u:
                    st.session_state[USER_KEY] = u
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")
            st.caption("🔒 Acceso restringido. Solicita tus credenciales al administrador del tablero.")
        else:
            st.info(
                "**Modo demostración.** No hay usuarios configurados en `st.secrets`; elige un rol para explorar. "
                "Configura `[auth.users]` para exigir contraseña.",
                icon="🧪",
            )
            desc = {
                "admin": "Acceso completo, incluida la configuración y la carga de datos.",
                "cartera": "Resumen, cola de gestión, segmentos y carga de archivos.",
                "direccion": "Resumen ejecutivo, segmentos y desempeño del modelo.",
                "analitica": "Resumen, segmentos, desempeño del modelo y carga de archivos.",
            }
            role = st.radio("Rol", list(ROLES), format_func=lambda r: ROLES[r], captions=[desc[r] for r in ROLES],
                            index=0, key="demo_role")
            if st.button("Entrar en modo demo", type="primary", width="stretch"):
                st.session_state[USER_KEY] = {"username": f"demo-{role}", "name": f"Demo · {ROLES[role]}",
                                              "role": role, "mode": "demo"}
                st.rerun()


def require_login() -> dict:
    """Devuelve el usuario autenticado o dibuja el login y detiene la ejecución."""
    u = current_user()
    if u is None:
        login_screen()
        st.stop()
    return u


def user_badge(u: dict) -> None:
    initials = "".join(p[0] for p in u["name"].replace("·", " ").split()[:2]).upper()
    st.markdown(
        f"""<div class="sat-user"><div class="avatar">{initials}</div>
        <div><div class="who">{u['name']}</div><div class="role">{ROLES.get(u['role'], u['role'])}
        {' · demo' if u.get('mode') == 'demo' else ''}</div></div></div>""",
        unsafe_allow_html=True,
    )
