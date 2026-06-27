"""Tests para src/metricas.py — EstadoBateria, estado_bateria, energia_total."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.grafo import Grafo
from src.metricas import EstadoBateria, estado_bateria, energia_total, costos_por_tramo


def _grafo_lineal() -> tuple[Grafo, list]:
    """Grafo a→b→c con pesos 3.0 y -1.0 (regeneración en el segundo tramo)."""
    g = Grafo()
    a, b, c = (0, 0, 0), (0, 0, 1), (0, 0, 2)
    g.agregar_arista(a, b, 3.0)
    g.agregar_arista(b, c, -1.0)
    return g, [a, b, c]


# ── EstadoBateria ─────────────────────────────────────────────────────────────

def test_estado_bateria_es_dataclass():
    eb = EstadoBateria(niveles=[100.0, 97.0], consumido=3.0,
                       regenerado=0.0, minimo=97.0, viable=True)
    assert eb.niveles == [100.0, 97.0]
    assert eb.consumido == 3.0
    assert eb.viable is True


# ── estado_bateria ────────────────────────────────────────────────────────────

def test_estado_bateria_propulsion_reduce_nivel():
    g, ruta = _grafo_lineal()
    # solo primer tramo (a→b, peso 3.0)
    resultado = estado_bateria([ruta[0], ruta[1]], g, e_max=100.0)
    assert isinstance(resultado, EstadoBateria)
    assert resultado.niveles == pytest.approx([100.0, 97.0])
    assert resultado.consumido == pytest.approx(3.0)
    assert resultado.regenerado == pytest.approx(0.0)
    assert resultado.viable is True


def test_estado_bateria_regeneracion_aumenta_nivel():
    g, ruta = _grafo_lineal()
    resultado = estado_bateria(ruta, g, e_max=100.0)
    # a→b consume 3, b→c regenera 1
    assert resultado.niveles == pytest.approx([100.0, 97.0, 98.0])
    assert resultado.regenerado == pytest.approx(1.0)


def test_estado_bateria_no_supera_emax():
    g = Grafo()
    a, b = (0, 0, 0), (0, 0, 1)
    g.agregar_arista(a, b, -50.0)  # regeneración masiva
    resultado = estado_bateria([a, b], g, e_max=10.0, bateria_inicial=10.0)
    assert resultado.niveles[-1] == pytest.approx(10.0)


def test_estado_bateria_arista_inexistente_lanza_error():
    g = Grafo()
    a, b, c = (0, 0, 0), (0, 0, 1), (0, 0, 2)
    g.agregar_arista(a, b, 1.0)
    with pytest.raises(ValueError, match="no existe"):
        estado_bateria([a, b, c], g, e_max=100.0)


# ── energia_total ─────────────────────────────────────────────────────────────

def test_energia_total_suma_pesos():
    g, ruta = _grafo_lineal()
    total = energia_total(ruta, g)
    assert total == pytest.approx(3.0 + (-1.0))


def test_energia_total_arista_inexistente_lanza_error():
    g = Grafo()
    a, b = (0, 0, 0), (0, 0, 1)
    g.agregar_nodo(a)
    g.agregar_nodo(b)
    with pytest.raises(ValueError, match="no existe"):
        energia_total([a, b], g)


# ── costos_por_tramo ──────────────────────────────────────────────────────────

def test_costos_por_tramo_extrae_diagonal():
    M = np.array([
        [0.0, 1.0, 5.0],
        [4.0, 0.0, 2.0],
        [3.0, 6.0, 0.0],
    ])
    # orden 0→1→2→0
    costos = costos_por_tramo([0, 1, 2, 0], M)
    assert costos == pytest.approx([1.0, 2.0, 3.0])
