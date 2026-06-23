"""Algoritmos de planificación: Bellman-Ford y ATSP (RF-04, RF-05).

Capa 1: rutas de mínima energía entre zonas con Bellman-Ford (admite pesos
negativos). Capa 2: orden óptimo de visita resuelto como un Problema del
Viajante Asimétrico (ATSP) por enumeración exacta (fuerza bruta).
"""
from __future__ import annotations

# TODO: borrar este bloque cuando termines de probar algoritmos.py
if __package__ is None or __package__ == "":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    __package__ = "src"
# TODO: fin bloque fix __package__

import itertools
import math

import numpy as np

from .grafo import Grafo, Nodo


def bellman_ford(
    grafo: Grafo,
    origen: Nodo,
) -> tuple[dict[Nodo, float], dict[Nodo, Nodo | None]]:
    """Calcula las distancias de mínima energía desde un origen (RF-04).

    Implementa relajación con terminación temprana: corta en cuanto una
    iteración completa no relaja ninguna arista. En una malla esto converge en
    decenas de pasadas (no en V-1), lo que cumple el presupuesto de tiempo
    del RNF-01.

    Args:
        grafo: Grafo dirigido y ponderado (puede tener pesos negativos).
        origen: Nodo desde el que se calculan las distancias.

    Returns:
        Tupla (dist, pred) donde dist[v] es la energía mínima origen -> v y
        pred[v] el predecesor de v en el árbol de caminos mínimos.
        Nodos inalcanzables quedan con dist = +inf.
    """
    dist: dict[Nodo, float] = {n: math.inf for n in grafo.nodos()}
    pred: dict[Nodo, Nodo | None] = {n: None for n in grafo.nodos()}
    dist[origen] = 0.0

    aristas = list(grafo.aristas())

    for _ in range(grafo.num_nodos - 1):
        relajado = False
        for u, v, w in aristas:
            if dist[u] < math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                relajado = True
        if not relajado:
            break   # convergencia anticipada

    return dist, pred


def hay_ciclo_negativo(grafo: Grafo, dist: dict[Nodo, float]) -> bool:
    """Detecta la existencia de algún ciclo negativo alcanzable (RNF-07).

    Una pasada adicional de relajación tras Bellman-Ford: si alguna arista aún
    puede relajarse, existe un ciclo negativo. Funciona como control de
    validación del modelo (no debería ocurrir si los parámetros están bien
    calibrados).

    Args:
        grafo: Grafo evaluado.
        dist: Distancias devueltas por bellman_ford.

    Returns:
        True si se detecta al menos un ciclo negativo.
    """
    for u, v, w in grafo.aristas():
        if dist[u] < math.inf and dist[u] + w < dist[v]:
            return True
    return False


def reconstruir_camino(
    pred: dict[Nodo, Nodo | None],
    origen: Nodo,
    destino: Nodo,
) -> list[Nodo]:
    """Reconstruye la secuencia de nodos del camino origen -> destino.

    Args:
        pred: Diccionario de predecesores de bellman_ford.
        origen: Nodo inicial.
        destino: Nodo final.

    Returns:
        Lista ordenada de nodos desde origen hasta destino.
        Si destino es inalcanzable devuelve lista vacía.
    """
    if pred.get(destino) is None and destino != origen:
        return []

    camino: list[Nodo] = []
    actual: Nodo | None = destino
    while actual is not None:
        camino.append(actual)
        if actual == origen:
            break
        actual = pred.get(actual)
    else:
        return []   # no se llegó al origen (grafo desconectado)

    camino.reverse()
    return camino


def matriz_costos(
    grafo: Grafo,
    waypoints: list[Nodo],
) -> tuple[np.ndarray, dict[tuple[int, int], list[Nodo]]]:
    """Construye la matriz de costos energéticos entre cada par de zonas (RF-04).

    Ejecuta Bellman-Ford desde cada waypoint y registra tanto el costo como el
    camino nodo a nodo entre cada par. Por la asimetría del grafo, M[i, j] no
    es igual a M[j, i].

    Args:
        grafo: Grafo del dominio.
        waypoints: Lista de zonas a conectar (incluida la base).

    Returns:
        Tupla (M, caminos) donde M[i, j] es la energía mínima del waypoint i al
        j, y caminos[(i, j)] la secuencia de nodos correspondiente.
    """
    k = len(waypoints)
    M = np.full((k, k), math.inf)
    caminos: dict[tuple[int, int], list[Nodo]] = {}

    for i, origen in enumerate(waypoints):
        dist, pred = bellman_ford(grafo, origen)
        for j, destino in enumerate(waypoints):
            if i == j:
                M[i, j] = 0.0
                caminos[(i, j)] = [origen]
                continue
            M[i, j] = dist[destino]
            caminos[(i, j)] = reconstruir_camino(pred, origen, destino)

    return M, caminos


def atsp_fuerza_bruta(matriz: np.ndarray, base: int = 0) -> tuple[list[int], float]:
    """Resuelve el orden óptimo de visita por enumeración exacta (RF-05).

    Prueba las (k-1)! permutaciones de los waypoints intermedios partiendo y
    regresando a la base, y devuelve la de menor energía total.

    Args:
        matriz: Matriz de costos asimétrica entre zonas (k x k).
        base: Índice de la zona base (partida y retorno obligatorio).

    Returns:
        Tupla (orden, costo_total) donde orden es la secuencia completa de
        índices empezando y terminando en base, y costo_total la energía neta
        de esa ruta.
    """
    k = matriz.shape[0]
    intermedios = [i for i in range(k) if i != base]

    mejor_orden: list[int] = []
    mejor_costo = math.inf

    for perm in itertools.permutations(intermedios):
        secuencia = [base] + list(perm) + [base]
        costo = sum(
            matriz[secuencia[i], secuencia[i + 1]]
            for i in range(len(secuencia) - 1)
        )
        if costo < mejor_costo:
            mejor_costo = costo
            mejor_orden = secuencia

    return mejor_orden, mejor_costo


def ensamblar_ruta(
    orden: list[int],
    caminos: dict[tuple[int, int], list[Nodo]],
) -> list[Nodo]:
    """Concatena los tramos en el orden óptimo para formar la ruta completa.

    Evita duplicar el nodo de unión entre tramos consecutivos.

    Args:
        orden: Secuencia óptima de índices de zona (de atsp_fuerza_bruta),
               empezando y terminando en la base.
        caminos: Caminos nodo a nodo entre pares de zonas (de matriz_costos).

    Returns:
        Secuencia completa de nodos de la ruta del AUV, base -> ... -> base.
    """
    ruta: list[Nodo] = []
    for i in range(len(orden) - 1):
        tramo = caminos[(orden[i], orden[i + 1])]
        if not tramo:
            continue
        # El primer nodo del tramo ya está en ruta (excepto al inicio).
        if ruta:
            ruta.extend(tramo[1:])
        else:
            ruta.extend(tramo)
    return ruta


# TODO: borrar este bloque cuando termines de probar algoritmos.py
if __name__ == "__main__":
    import pathlib
    from src.datos import cargar_corrientes, resumen
    from src.config import ParametrosModelo
    from src.grafo import construir_grafo
    from src.zonas import (
        divergencia, seleccionar_waypoints,
        seleccionar_centinelas, celda_mas_cercana,
        agregar_puntos_fijos,
    )

    nc = pathlib.Path(__file__).parent.parent / "data" / "lima3.nc"
    print("Cargando campo …")
    campo = cargar_corrientes(str(nc))
    params = ParametrosModelo()

    capa = 0
    lat_media = math.radians(float(campo.lat.mean()))
    dy = abs(float(campo.lat[1] - campo.lat[0])) * 111_320.0
    dx = abs(float(campo.lon[1] - campo.lon[0])) * 111_320.0 * math.cos(lat_media)

    uo  = campo.uo[capa]
    vo  = campo.vo[capa]
    nav = campo.navegable[capa]

    div  = divergencia(uo, vo, dx, dy)
    wps  = seleccionar_waypoints(div, nav, params.k_zonas, capa=capa, dist_min_celdas=3)
    cent = seleccionar_centinelas(uo, vo, nav, campo.lon, n=2, capa=capa)
    base = celda_mas_cercana(-12.05, -77.15, campo.lat, campo.lon, nav, capa=capa)
    todos, base_nodo = agregar_puntos_fijos(wps + cent, wps[0], base)

    print(f"Waypoints totales: {len(todos)}  →  {math.factorial(len(todos)-1)} órdenes ATSP")

    print("\nConstruyendo grafo …")
    grafo = construir_grafo(campo, params)
    print(f"Nodos: {grafo.num_nodos}  Aristas: {grafo.num_aristas}")

    print("\nComprobando ciclos negativos …")
    dist0, _ = bellman_ford(grafo, todos[0])
    if hay_ciclo_negativo(grafo, dist0):
        print("  ⚠ Se detectó un ciclo negativo — revisar parámetros del modelo.")
    else:
        print("  ✓ Sin ciclos negativos.")

    print("\nCalculando matriz de costos …")
    M, caminos = matriz_costos(grafo, todos)
    base_idx = todos.index(base_nodo)
    print("Matriz de costos [J] (filas=origen, cols=destino):")
    print(np.array2string(M, precision=1, suppress_small=True))

    print("\nResolviendo ATSP …")
    orden, costo_total = atsp_fuerza_bruta(M, base=base_idx)
    print(f"Orden óptimo de visita: {orden}")
    print(f"Costo total de la misión: {costo_total:.2f} J")

    ruta = ensamblar_ruta(orden, caminos)
    print(f"Nodos en la ruta completa: {len(ruta)}")

    print("\n✓ algoritmos.py OK")
# TODO: fin bloque de prueba
