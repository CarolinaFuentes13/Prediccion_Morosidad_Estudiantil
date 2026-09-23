"""Pruebas de humo: cada página del tablero se ejecuta sin excepciones para cada rol.

Ejecutar desde la raíz del repo:  python -m pytest Dashboard/alerta_temprana/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from core.config import PAGE_ACCESS  # noqa: E402
from core.nav import PAGES  # noqa: E402

ROLE_FOR_PAGE = {k: sorted(v)[0] for k, v in PAGE_ACCESS.items()}


def _app(role: str) -> AppTest:
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=120)
    at.session_state["sat_user"] = {"username": f"test-{role}", "name": f"Test {role}", "role": role, "mode": "demo"}
    return at


@pytest.mark.parametrize("key", list(PAGES))
def test_page_runs(key):
    at = _app("admin")
    at.run()
    at.switch_page(PAGES[key][0])
    at.run()
    assert not at.exception, f"{key}: {[e.value for e in at.exception]}"


@pytest.mark.parametrize("role", ["cartera", "direccion", "analitica"])
def test_roles_default_page(role):
    at = _app(role)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_login_screen_without_user():
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Iniciar sesión" in m.value for m in at.markdown)
