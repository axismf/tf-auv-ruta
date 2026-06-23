"""Cálculo y exportación de métricas de la misión (RF-08, RF-09)."""
from __future__ import annotations

if __package__ is None or __package__ == "":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    __package__ = "src"

import csv
import math

import numpy as np

from .datos import CampoCorrientes
from .grafo import Grafo, Nodo


def energia_total(ruta: list[Nodo], grafo: Grafo) -> float:
    """Suma la energía neta de todos los tramos de la ruta (RF-08).

    Recorre la ruta nodo a nodo y acumula el peso de cada arista. Si una
    arista no existe en el grafo (tramo desconectado), lanza ValueError.

    Args:
        ruta: Secuencia de nodos de la ruta completa.
        grafo: Grafo del que se leen los pesos de las aristas.

    Returns:
        Energía neta total de la misión [J].
    """
    total = 0.0
    for u, v in zip(ruta, ruta[1:]):
        pesos = {dest: w for dest, w in grafo.vecinos(u)}
        if v not in pesos:
            raise ValueError(f"Arista {u} -> {v} no existe en el grafo.")
        total += pesos[v]
    return total


def costos_por_tramo(orden: list[int], matriz: np.ndarray) -> list[float]:
    """Devuelve el costo energético de cada tramo entre zonas (RF-08).

    Args:
        orden: Secuencia óptima de índices de zona, empezando y terminando
               en la base (salida de atsp_fuerza_bruta).
        matriz: Matriz de costos asimétrica entre zonas.

    Returns:
        Lista de costos [J], uno por tramo del recorrido (len = len(orden)-1).
    """
    return [float(matriz[orden[i], orden[i + 1]]) for i in range(len(orden) - 1)]


def resumen_mision(
    orden: list[int],
    waypoints: list[Nodo],
    matriz: np.ndarray,
    campo: CampoCorrientes,
) -> str:
    """Genera un texto con los resultados numéricos de la misión (RF-08).

    Args:
        orden: Secuencia óptima de índices de zona.
        waypoints: Lista de zonas (para traducir índices a coordenadas).
        matriz: Matriz de costos entre zonas.
        campo: Campo de corrientes (para lat/lon reales).

    Returns:
        Texto formateado con el orden de visita, costo por tramo y total.
    """
    tramos = costos_por_tramo(orden, matriz)
    total = sum(tramos)
    lineas = ["── Resultado de la misión ──────────────────────"]
    for k, (i, j) in enumerate(zip(orden, orden[1:])):
        _, li, lj = waypoints[i]
        _, di, dj = waypoints[j]
        lineas.append(
            f"  Tramo {k+1}: zona {i} ({campo.lat[li]:.2f}°,{campo.lon[lj]:.2f}°)"
            f" → zona {j} ({campo.lat[di]:.2f}°,{campo.lon[dj]:.2f}°)"
            f"  {tramos[k]:>10.1f} J"
        )
    lineas.append(f"{'─'*50}")
    lineas.append(f"  Energía total de la misión: {total:>12.1f} J")
    return "\n".join(lineas)


def estado_bateria(
    ruta: list[Nodo],
    grafo: Grafo,
    e_max: float,
    bateria_inicial: float | None = None,
) -> dict:
    """Simula el estado de batería del AUV a lo largo de la ruta (RF-08).

    Recorre la ruta nodo a nodo acumulando el costo de cada arista. Los pesos
    positivos consumen batería; los negativos la recargan (regeneración). La
    carga está acotada en [0, e_max]: no puede superar la capacidad ni quedar
    en negativo (lo que indicaría que la misión no es viable).

    Args:
        ruta: Secuencia de nodos de la ruta completa.
        grafo: Grafo del que se leen los pesos de las aristas.
        e_max: Capacidad máxima de la batería [J].
        bateria_inicial: Carga al inicio [J]. Si es None se usa e_max (llena).

    Returns:
        Diccionario con:
        - "niveles": lista de carga [J] en cada nodo de la ruta.
        - "consumido": energía total gastada en propulsión [J].
        - "regenerado": energía total recuperada [J].
        - "minimo": nivel mínimo alcanzado [J].
        - "viable": True si la batería nunca llega a 0.
    """
    if bateria_inicial is None:
        bateria_inicial = e_max

    niveles = [bateria_inicial]
    consumido = 0.0
    regenerado = 0.0

    for u, v in zip(ruta, ruta[1:]):
        pesos = {dest: w for dest, w in grafo.vecinos(u)}
        if v not in pesos:
            raise ValueError(f"Arista {u} -> {v} no existe en el grafo.")
        w = pesos[v]
        nueva = niveles[-1] - w          # w > 0 → gasta; w < 0 → recarga
        nueva = min(nueva, e_max)        # no superar capacidad máxima
        if w > 0:
            consumido  += w
        else:
            regenerado += -w
        niveles.append(max(nueva, 0.0)) # no bajar de 0 (anotamos pero marcamos)

    return {
        "niveles":    niveles,
        "consumido":  consumido,
        "regenerado": regenerado,
        "minimo":     min(niveles),
        "viable":     min(niveles) > 0,
    }


def exportar_csv(
    ruta: list[Nodo],
    campo: CampoCorrientes,
    ruta_salida: str,
) -> None:
    """Exporta la ruta resultante a un archivo CSV (RF-09).

    Traduce cada nodo (prof_idx, lat_idx, lon_idx) a sus coordenadas reales
    antes de escribir, para que el CSV sea interpretable fuera del programa.

    Columnas: paso, prof_m, lat_deg, lon_deg

    Args:
        ruta: Secuencia de nodos de la ruta.
        campo: Campo de corrientes (para traducir índices a lat/lon/prof).
        ruta_salida: Ruta del archivo CSV de salida.
    """
    with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paso", "prof_m", "lat_deg", "lon_deg"])
        for paso, (p, i, j) in enumerate(ruta):
            writer.writerow([
                paso,
                round(float(campo.prof[p]), 4),
                round(float(campo.lat[i]), 6),
                round(float(campo.lon[j]), 6),
            ])


if __name__ == "__main__":
    import pathlib
    from src.datos import cargar_corrientes
    from src.config import ParametrosModelo
    from src.grafo import construir_grafo
    from src.zonas import (
        divergencia, seleccionar_waypoints,
        seleccionar_centinelas, celda_mas_cercana,
        agregar_puntos_fijos,
    )
    from src.algoritmos import matriz_costos, atsp_fuerza_bruta, ensamblar_ruta

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
    base_idx = todos.index(base_nodo)

    print("Construyendo grafo …")
    grafo = construir_grafo(campo, params)

    print("Calculando matriz de costos …")
    M, caminos = matriz_costos(grafo, todos)

    print("Resolviendo ATSP …")
    orden, _ = atsp_fuerza_bruta(M, base=base_idx)
    ruta = ensamblar_ruta(orden, caminos)

    # --- Métricas ---
    print()
    print(resumen_mision(orden, todos, M, campo))

    e_total = energia_total(ruta, grafo)
    print(f"\n  Verificación nodo a nodo:   {e_total:.1f} J")

    # --- CSV ---
    out_csv = pathlib.Path(__file__).parent.parent / "outputs" / "rutas" / "ruta.csv"
    out_csv.parent.mkdir(exist_ok=True)
    exportar_csv(ruta, campo, str(out_csv))
    print(f"\n  CSV exportado → {out_csv}")

    print("\n✓ metricas.py OK")
