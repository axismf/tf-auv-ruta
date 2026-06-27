"""Algoritmos de planificación: Bellman-Ford y ATSP (RF-04, RF-05)."""
from __future__ import annotations

from .dependencias import *
from .grafo import Grafo, Nodo


__all__ = [
    "ensamblar_ruta",
    "atsp_fuerza_bruta",
    "matriz_costos",
    "bellman_ford",
    "reconstruir_camino",
    "hay_ciclo_negativo",
]


def ensamblar_ruta(
    orden: list[int],
    caminos: dict[tuple[int, int], list[Nodo]],
) -> list[Nodo]:
    """Concatena los tramos en el orden óptimo para formar la ruta completa.

    Evita duplicar el nodo de unión entre tramos consecutivos.

    Args:
        orden:   Secuencia óptima de índices de zona (salida de atsp_fuerza_bruta).
        caminos: Caminos nodo a nodo entre pares de zonas (salida de matriz_costos).

    Returns:
        Secuencia completa de nodos base → ... → base.
    """
    ruta: list[Nodo] = []
    for i in range(len(orden) - 1):
        tramo = caminos[(orden[i], orden[i + 1])]
        if not tramo:
            continue
        ruta.extend(tramo[1:] if ruta else tramo)
    return ruta


def atsp_fuerza_bruta(
    matriz: np.ndarray,
    base: int = 0,
) -> tuple[list[int], float]:
    """Resuelve el ATSP por enumeración exacta de las (k-1)! permutaciones (RF-05).

    Args:
        matriz: Matriz de costos asimétrica entre zonas (k × k).
        base:   Índice de la zona base (partida y retorno obligatorio).

    Returns:
        (orden, costo_total) — orden incluye la base al inicio y al final.
    """
    k           = matriz.shape[0]
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


def matriz_costos(
    grafo: Grafo,
    waypoints: list[Nodo],
) -> tuple[np.ndarray, dict[tuple[int, int], list[Nodo]]]:
    """Construye la matriz de costos energéticos entre cada par de zonas (RF-04).

    Ejecuta Bellman-Ford desde cada waypoint. M[i, j] = energía mínima i → j.

    Args:
        grafo:     Grafo del dominio.
        waypoints: Lista de zonas a conectar (incluida la base).

    Returns:
        (M, caminos): M[i, j] = energía mínima; caminos[(i, j)] = secuencia de nodos.
    """
    k = len(waypoints)
    M = np.full((k, k), math.inf)
    caminos: dict[tuple[int, int], list[Nodo]] = {}

    for i, origen in enumerate(waypoints):
        dist, pred = bellman_ford(grafo, origen)
        for j, destino in enumerate(waypoints):
            if i == j:
                M[i, j]         = 0.0
                caminos[(i, j)] = [origen]
                continue
            M[i, j]         = dist[destino]
            caminos[(i, j)] = reconstruir_camino(pred, origen, destino)

    return M, caminos


def bellman_ford(
    grafo: Grafo,
    origen: Nodo,
) -> tuple[dict[Nodo, float], dict[Nodo, Nodo | None]]:
    """Distancias de mínima energía desde origen, con terminación anticipada (RF-04).

    Args:
        grafo:  Grafo dirigido (puede tener pesos negativos).
        origen: Nodo desde el que se calculan las distancias.

    Returns:
        (dist, pred): dist[v] = energía mínima origen → v;
                      pred[v] = predecesor de v en el árbol de caminos mínimos.
    """
    dist: dict[Nodo, float]       = {n: math.inf for n in grafo.nodos()}
    pred: dict[Nodo, Nodo | None] = {n: None     for n in grafo.nodos()}
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
            break

    return dist, pred


def reconstruir_camino(
    pred: dict[Nodo, Nodo | None],
    origen: Nodo,
    destino: Nodo,
) -> list[Nodo]:
    """Reconstruye la secuencia de nodos del camino origen → destino.

    Returns:
        Lista ordenada de nodos, o lista vacía si destino es inalcanzable.
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
        return []

    camino.reverse()
    return camino


def hay_ciclo_negativo(
    grafo: Grafo,
    dist: dict[Nodo, float],
) -> bool:
    """Detecta la existencia de algún ciclo negativo alcanzable.

    Una pasada adicional de relajación tras Bellman-Ford lo revela.
    """
    for u, v, w in grafo.aristas():
        if dist[u] < math.inf and dist[u] + w < dist[v]:
            return True
    return False


if __name__ == "__main__":
    # Ejecutar con: python -m src.algoritmos
    from src.datos import cargar_corrientes
    from src.config import ParametrosModelo
    from src.grafo import construir_grafo
    from src.zonas import (
        divergencia, seleccionar_waypoints, seleccionar_centinelas,
        celda_mas_cercana, agregar_puntos_fijos,
    )

    nc     = pathlib.Path(__file__).parent.parent / "data" / "lima3.nc"
    campo  = cargar_corrientes(str(nc))
    params = ParametrosModelo()
    capa   = 0

    lat_media = math.radians(float(campo.lat.mean()))
    dy = abs(float(campo.lat[1] - campo.lat[0])) * GRADOS_A_METROS
    dx = abs(float(campo.lon[1] - campo.lon[0])) * GRADOS_A_METROS * math.cos(lat_media)

    div  = divergencia(campo.uo[capa], campo.vo[capa], dx, dy)
    wps  = seleccionar_waypoints(div, campo.navegable[capa], params.k_zonas,
                                  capa=capa, dist_min_celdas=3)
    cent = seleccionar_centinelas(campo.uo[capa], campo.vo[capa],
                                   campo.navegable[capa], campo.lon, n=2, capa=capa)
    base = celda_mas_cercana(-12.05, -77.15, campo.lat, campo.lon,
                              campo.navegable[capa], capa=capa)
    todos, base_nodo = agregar_puntos_fijos(wps + cent, wps[0], base)
    base_idx         = todos.index(base_nodo)

    grafo        = construir_grafo(campo, params)
    dist0, _     = bellman_ford(grafo, todos[0])
    print(f"Ciclo negativo: {'SÍ ⚠' if hay_ciclo_negativo(grafo, dist0) else 'No ✓'}")

    M, caminos   = matriz_costos(grafo, todos)
    orden, costo = atsp_fuerza_bruta(M, base=base_idx)
    ruta         = ensamblar_ruta(orden, caminos)
    print(f"Orden: {orden}  Costo: {costo:.2f} J  Nodos en ruta: {len(ruta)}")
    print("\n✓ algoritmos.py OK")
