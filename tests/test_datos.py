"""Tests para src/datos.py — marcar_navegables y funciones puras."""
from __future__ import annotations

import numpy as np
import pytest

from src.datos import marcar_navegables


def test_marcar_navegables_sin_nan():
    uo = np.ones((2, 3, 4))
    vo = np.ones((2, 3, 4))
    nav = marcar_navegables(uo, vo)
    assert nav.all()
    assert nav.dtype == bool


def test_marcar_navegables_nan_uo_marca_false():
    uo = np.ones((1, 2, 2))
    vo = np.ones((1, 2, 2))
    uo[0, 0, 0] = np.nan
    nav = marcar_navegables(uo, vo)
    assert not nav[0, 0, 0]
    assert nav[0, 0, 1]


def test_marcar_navegables_nan_vo_marca_false():
    uo = np.ones((1, 2, 2))
    vo = np.ones((1, 2, 2))
    vo[0, 1, 1] = np.nan
    nav = marcar_navegables(uo, vo)
    assert not nav[0, 1, 1]


def test_marcar_navegables_nan_en_ambos():
    uo = np.full((1, 3, 3), np.nan)
    vo = np.full((1, 3, 3), np.nan)
    nav = marcar_navegables(uo, vo)
    assert not nav.any()


def test_marcar_navegables_preserva_shape():
    shape = (3, 5, 7)
    uo = np.random.randn(*shape)
    vo = np.random.randn(*shape)
    nav = marcar_navegables(uo, vo)
    assert nav.shape == shape
