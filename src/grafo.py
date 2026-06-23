"""Construcción del grafo dirigido ponderado y función de costo (RF-03, RF-04).

El espacio marino se modela como un grafo dirigido y asimétrico donde cada
celda navegable es un nodo y el peso de cada arista es la energía neta de
desplazamiento del AUV, calculada a partir de la corriente local.
"""
from __future__ import annotations

# Permite ejecutar/depurar este módulo directamente sin romper los imports
# relativos. Cuando se importa como parte del paquete src, __package__ ya
# tiene el valor correcto y este bloque no hace nada.
if __package__ is None or __package__ == "":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    __package__ = "src"

import itertools
import math
from typing import Iterator

from .config import ParametrosModelo
from .datos import CampoCorrientes

# Un nodo se identifica por su terna de índices en la malla (prof, lat, lon).
Nodo = tuple[int, int, int]

# Metros por grado de latitud (aprox.). La longitud se corrige por cos(lat).
_GRADOS_A_METROS = 111_320.0


class Grafo:
    """Grafo dirigido y ponderado representado por lista de adyacencia.

    Los pesos pueden ser negativos (régimen de regeneración), por lo que el
    grafo está pensado para resolverse con Bellman-Ford y no con Dijkstra.
    """

    def __init__(self) -> None:
        # adyacencia[u] = lista de (v, peso) de las aristas dirigidas u -> v.
        self._adyacencia: dict[Nodo, list[tuple[Nodo, float]]] = {}

    def agregar_nodo(self, u: Nodo) -> None:
        """Registra un nodo sin aristas si aún no existe."""
        self._adyacencia.setdefault(u, [])

    def agregar_arista(self, u: Nodo, v: Nodo, peso: float) -> None:
        """Agrega la arista dirigida u -> v con el peso dado."""
        self._adyacencia.setdefault(u, []).append((v, peso))
        self._adyacencia.setdefault(v, [])

    def vecinos(self, u: Nodo) -> list[tuple[Nodo, float]]:
        """Devuelve las aristas salientes (v, peso) del nodo u."""
        return self._adyacencia.get(u, [])

    def nodos(self) -> Iterator[Nodo]:
        """Itera sobre todos los nodos del grafo."""
        return iter(self._adyacencia)

    def aristas(self) -> Iterator[tuple[Nodo, Nodo, float]]:
        """Itera sobre todas las aristas como ternas (u, v, peso)."""
        for u, lista in self._adyacencia.items():
            for v, peso in lista:
                yield u, v, peso

    @property
    def num_nodos(self) -> int:
        """Cantidad de nodos del grafo."""
        return len(self._adyacencia)

    @property
    def num_aristas(self) -> int:
        """Cantidad de aristas dirigidas del grafo."""
        return sum(len(lista) for lista in self._adyacencia.values())


def costo_arista(
    origen: Nodo,
    destino: Nodo,
    campo: CampoCorrientes,
    params: ParametrosModelo,
) -> float:
    """Calcula la energía neta de recorrer la arista origen -> destino (RF-04).

    Geometría: se convierte el desplazamiento de grados a metros (corrigiendo
    la longitud por cos(lat)), de donde salen la longitud L y el vector
    unitario de dirección e. Con la corriente local v_c = (uo, vo, 0) se define
    la velocidad relativa al agua v_r = s * e - v_c.

    Régimen (decidido por v_paralela = v_c . e):
      - Propulsión (v_paralela < s): peso positivo  k_p * |v_r|^3 * (L/s).
      - Regeneración (v_paralela >= s): peso negativo
        -k_r * eta * |v_r|^3 * (L/s), con la recuperación acotada por e_max.

    Args:
        origen: Nodo de partida (prof, lat, lon).
        destino: Nodo de llegada (prof, lat, lon).
        campo: Campo de corrientes.
        params: Parámetros del modelo.

    Returns:
        Energía neta de la arista; positiva si se consume, negativa si se
        recupera.
    """
    pa, ia, ja = origen
    pb, ib, jb = destino

    # Desplazamiento en metros (x: este, y: norte, z: profundidad).
    lat_media = math.radians((campo.lat[ia] + campo.lat[ib]) / 2.0)
    dx = (campo.lon[jb] - campo.lon[ja]) * _GRADOS_A_METROS * math.cos(lat_media)
    dy = (campo.lat[ib] - campo.lat[ia]) * _GRADOS_A_METROS
    dz = campo.prof[pb] - campo.prof[pa]

    longitud = math.sqrt(dx * dx + dy * dy + dz * dz)
    if longitud == 0.0:
        return 0.0
    ex, ey, ez = dx / longitud, dy / longitud, dz / longitud

    # Corriente local en el nodo de origen (sin componente vertical).
    cx = float(campo.uo[pa, ia, ja])
    cy = float(campo.vo[pa, ia, ja])

    # Velocidad relativa al agua: v_r = s * e - v_c.
    rx = params.s * ex - cx
    ry = params.s * ey - cy
    rz = params.s * ez
    vr3 = (rx * rx + ry * ry + rz * rz) ** 1.5

    # Proyección de la corriente sobre la dirección de avance.
    v_paralela = cx * ex + cy * ey
    tiempo = longitud / params.s

    if v_paralela < params.s:
        return params.k_p * vr3 * tiempo

    recuperada = params.k_r * params.eta * vr3 * tiempo
    return -min(recuperada, params.e_max)


def construir_grafo(campo: CampoCorrientes, params: ParametrosModelo) -> Grafo:
    """Construye el grafo dirigido completo a partir del campo (RF-03).

    Conecta cada celda navegable con sus hasta 26 vecinos en las tres
    dimensiones. Cada celda agrega sus aristas salientes; como cada par de
    celdas se recorre desde ambos extremos, el resultado contiene las dos
    aristas dirigidas (A -> B y B -> A) con pesos calculados de forma
    independiente. Las celdas de tierra se omiten y actúan como obstáculos.

    Args:
        campo: Campo de corrientes con la máscara de navegabilidad.
        params: Parámetros del modelo.

    Returns:
        El grafo dirigido y ponderado del dominio.
    """
    grafo = Grafo()
    n_prof, n_lat, n_lon = campo.uo.shape
    nav = campo.navegable

    offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]

    for p in range(n_prof):
        for i in range(n_lat):
            for j in range(n_lon):
                if not nav[p, i, j]:
                    continue
                origen = (p, i, j)
                grafo.agregar_nodo(origen)
                for dp, di, dj in offsets:
                    pp, ii, jj = p + dp, i + di, j + dj
                    if (
                        0 <= pp < n_prof
                        and 0 <= ii < n_lat
                        and 0 <= jj < n_lon
                        and nav[pp, ii, jj]
                    ):
                        destino = (pp, ii, jj)
                        peso = costo_arista(origen, destino, campo, params)
                        grafo.agregar_arista(origen, destino, peso)

    return grafo