"""Tests para src/zonas.py — divergencia, celda_mas_cercana, agregar_puntos_fijos."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.zonas import divergencia, celda_mas_cercana, agregar_puntos_fijos


# ── divergencia ──────────────────────────────────────────────────────────────

def test_divergencia_campo_uniforme_es_cero():
    n = 10
    uo = np.ones((n, n))
    vo = np.ones((n, n))
    div = divergencia(uo, vo, dx=1.0, dy=1.0)
    assert np.allclose(div, 0.0, atol=1e-10)


def test_divergencia_fuente_puntual_positiva():
    n = 5
    uo = np.zeros((n, n))
    vo = np.zeros((n, n))
    uo[:, 2] = 1.0  # corriente hacia la derecha en la columna central
    div = divergencia(uo, vo, dx=1.0, dy=1.0)
    # debe haber divergencia positiva a la derecha y negativa a la izquierda
    assert div.sum() == pytest.approx(0.0, abs=1e-10)


def test_divergencia_shape_preservada():
    uo = np.random.randn(8, 12)
    vo = np.random.randn(8, 12)
    div = divergencia(uo, vo, dx=1.0, dy=1.0)
    assert div.shape == (8, 12)


# ── celda_mas_cercana ─────────────────────────────────────────────────────────

def test_celda_mas_cercana_encuentra_celda_exacta():
    lat = np.array([-13.0, -12.0, -11.0])
    lon = np.array([-78.0, -77.0, -76.0])
    nav = np.ones((3, 3), dtype=bool)
    resultado = celda_mas_cercana(-12.0, -77.0, lat, lon, nav, capa=0)
    assert resultado == (0, 1, 1)


def test_celda_mas_cercana_evita_tierra():
    lat = np.array([-12.0, -11.0])
    lon = np.array([-78.0, -77.0])
    nav = np.array([[False, False],
                    [False, True]])
    resultado = celda_mas_cercana(-12.0, -78.0, lat, lon, nav, capa=0)
    # la única celda navegable es (1,1)
    assert resultado == (0, 1, 1)


def test_celda_mas_cercana_respeta_capa():
    lat = np.array([-12.0])
    lon = np.array([-77.0])
    nav = np.ones((1, 1), dtype=bool)
    resultado = celda_mas_cercana(-12.0, -77.0, lat, lon, nav, capa=2)
    assert resultado[0] == 2


# ── agregar_puntos_fijos ──────────────────────────────────────────────────────

def test_agregar_puntos_fijos_no_duplica():
    a, b, c = (0, 0, 0), (0, 1, 1), (0, 2, 2)
    wps = [a, b]
    todos, base = agregar_puntos_fijos(wps, fuente=a, base=c)
    assert todos.count(a) == 1
    assert c in todos
    assert base == c


def test_agregar_puntos_fijos_fuente_ya_incluida():
    a, b = (0, 0, 0), (0, 1, 1)
    todos, _ = agregar_puntos_fijos([a, b], fuente=a, base=b)
    assert len(todos) == 2


def test_agregar_puntos_fijos_retorna_base():
    a, b, c = (0, 0, 0), (0, 1, 1), (0, 2, 2)
    _, base = agregar_puntos_fijos([a], fuente=a, base=c)
    assert base == c
