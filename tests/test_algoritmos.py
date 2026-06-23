"""Pruebas del módulo de algoritmos (apoya RNF-06: determinismo).

Ejecutar con:  pytest
"""
from __future__ import annotations

import math

import numpy as np

from src.grafo import Grafo
from src.algoritmos import bellman_ford, hay_ciclo_negativo, atsp_fuerza_bruta


def test_bellman_ford_camino_simple() -> None:
    """En un grafo lineal A->B->C, la distancia a C es la suma de pesos."""
    g = Grafo()
    a, b, c = (0, 0, 0), (0, 0, 1), (0, 0, 2)
    g.agregar_arista(a, b, 3.0)
    g.agregar_arista(b, c, 5.0)

    dist, pred = bellman_ford(g, a)

    assert dist[a] == 0.0
    assert dist[b] == 3.0
    assert dist[c] == 8.0
    assert pred[b] == a
    assert pred[c] == b


def test_bellman_ford_prefiere_arista_negativa() -> None:
    """Cuando hay una arista de regeneración (peso negativo), debe usarse."""
    # Dos caminos de a hasta c:
    #   a -> c directo:   costo 10
    #   a -> b -> c:      costo 4 + (−2) = 2  (óptimo)
    g = Grafo()
    a, b, c = (0, 0, 0), (0, 0, 1), (0, 0, 2)
    g.agregar_arista(a, c, 10.0)
    g.agregar_arista(a, b, 4.0)
    g.agregar_arista(b, c, -2.0)

    dist, _ = bellman_ford(g, a)

    assert dist[c] == 2.0


def test_detecta_ciclo_negativo() -> None:
    """Un grafo con un ciclo de peso total negativo debe ser detectado."""
    # a -> b -> c -> a  con peso total 1 − 4 + 1 = −2.
    g = Grafo()
    a, b, c = (0, 0, 0), (0, 0, 1), (0, 0, 2)
    g.agregar_arista(a, b, 1.0)
    g.agregar_arista(b, c, -4.0)
    g.agregar_arista(c, a, 1.0)

    dist, _ = bellman_ford(g, a)

    assert hay_ciclo_negativo(g, dist) is True


def test_atsp_orden_optimo_caso_pequeno() -> None:
    """Para una matriz conocida, el ATSP devuelve el orden de menor costo."""
    # base=0; intermedios={1, 2}
    # Orden 0→1→2→0: M[0,1]+M[1,2]+M[2,0] = 1+1+1 = 3  (óptimo)
    # Orden 0→2→1→0: M[0,2]+M[2,1]+M[1,0] = 5+2+4 = 11
    M = np.array([
        [0.0, 1.0, 5.0],
        [4.0, 0.0, 1.0],
        [1.0, 2.0, 0.0],
    ])
    orden, costo = atsp_fuerza_bruta(M, base=0)

    assert orden == [0, 1, 2, 0]
    assert math.isclose(costo, 3.0)
