"""Tests para src/grafo.py — Grafo, GrafoBase, construir_grafo, costo_arista."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.grafo import Grafo, Nodo


# ── Grafo.agregar_nodo / agregar_arista ──────────────────────────────────────

def test_grafo_agrega_nodo():
    g = Grafo()
    n: Nodo = (0, 0, 0)
    g.agregar_nodo(n)
    assert g.num_nodos == 1
    assert g.num_aristas == 0


def test_grafo_agrega_arista_implica_ambos_nodos():
    g = Grafo()
    a, b = (0, 0, 0), (0, 0, 1)
    g.agregar_arista(a, b, 5.0)
    assert g.num_nodos == 2
    assert g.num_aristas == 1


def test_grafo_vecinos_vacios_nodo_sin_salidas():
    g = Grafo()
    a, b = (0, 0, 0), (0, 0, 1)
    g.agregar_arista(a, b, 2.0)
    assert g.vecinos(b) == []


# ── Grafo.peso ───────────────────────────────────────────────────────────────

def test_peso_devuelve_valor_arista_existente():
    g = Grafo()
    a, b = (0, 0, 0), (0, 0, 1)
    g.agregar_arista(a, b, 7.5)
    assert g.peso(a, b) == pytest.approx(7.5)


def test_peso_devuelve_none_arista_inexistente():
    g = Grafo()
    a, b = (0, 0, 0), (0, 0, 1)
    g.agregar_nodo(a)
    g.agregar_nodo(b)
    assert g.peso(a, b) is None


def test_peso_arista_negativa():
    g = Grafo()
    a, b = (0, 0, 0), (0, 0, 1)
    g.agregar_arista(a, b, -3.0)
    assert g.peso(a, b) == pytest.approx(-3.0)


def test_peso_no_confunde_direccion():
    g = Grafo()
    a, b = (0, 0, 0), (0, 0, 1)
    g.agregar_arista(a, b, 1.0)
    assert g.peso(b, a) is None


# ── Grafo.nodos / aristas ────────────────────────────────────────────────────

def test_nodos_incluye_extremos_arista():
    g = Grafo()
    a, b = (0, 0, 0), (0, 1, 0)
    g.agregar_arista(a, b, 1.0)
    assert set(g.nodos()) == {a, b}


def test_aristas_itera_correctamente():
    g = Grafo()
    a, b, c = (0, 0, 0), (0, 0, 1), (0, 1, 0)
    g.agregar_arista(a, b, 2.0)
    g.agregar_arista(b, c, 3.0)
    resultado = list(g.aristas())
    assert (a, b, 2.0) in resultado
    assert (b, c, 3.0) in resultado
    assert len(resultado) == 2
