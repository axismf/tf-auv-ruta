"""Núcleo del planificador de rutas de mínima energía para un AUV.

Flujo de dependencias (unidireccional):
    datos → grafo → zonas → algoritmos → metricas
    visualizacion consume el resto sin ser consumida por ningún otro módulo.
    Ningún módulo del núcleo depende de la interfaz (Streamlit).
"""
from __future__ import annotations

from .config import ParametrosModelo
from .datos import CampoCorrientes, cargar_corrientes, marcar_navegables, resumen
from .grafo import Grafo, Nodo, construir_grafo, costo_arista
from .zonas import (
    divergencia,
    seleccionar_waypoints,
    seleccionar_centinelas,
    celda_mas_cercana,
    agregar_puntos_fijos,
)
from .algoritmos import (
    ensamblar_ruta,
    atsp_fuerza_bruta,
    matriz_costos,
    bellman_ford,
    reconstruir_camino,
    hay_ciclo_negativo,
)
from .metricas import (
    EstadoBateria,
    resumen_mision,
    estado_bateria,
    exportar_csv,
    costos_por_tramo,
    energia_total,
)
from .visualizacion import (
    plot_campo,
    plot_divergencia,
    plot_zonas,
    plot_ruta,
    plot_3d,
    plot_bateria,
    plot_grafo_costos,
    plot_tours_atsp,
)

__all__ = [
    # config
    "ParametrosModelo",
    # datos
    "CampoCorrientes", "cargar_corrientes", "marcar_navegables", "resumen",
    # grafo
    "Grafo", "Nodo", "construir_grafo", "costo_arista",
    # zonas
    "divergencia", "seleccionar_waypoints", "seleccionar_centinelas",
    "celda_mas_cercana", "agregar_puntos_fijos",
    # algoritmos
    "ensamblar_ruta", "atsp_fuerza_bruta", "matriz_costos",
    "bellman_ford", "reconstruir_camino", "hay_ciclo_negativo",
    # metricas
    "EstadoBateria", "resumen_mision", "estado_bateria", "exportar_csv",
    "costos_por_tramo", "energia_total",
    # visualizacion
    "plot_campo", "plot_divergencia", "plot_zonas", "plot_ruta",
    "plot_3d", "plot_bateria", "plot_grafo_costos", "plot_tours_atsp",
]
