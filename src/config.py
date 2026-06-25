"""Parámetros de configuración del modelo energético del AUV.

Centraliza todas las constantes ajustables del modelo en un único objeto
inmutable, lo que refuerza el determinismo exigido por el RNF-05
(mismos parámetros -> mismos resultados).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParametrosModelo:
    """Parámetros físicos y de misión del planificador de rutas.

    Attributes:
        s: Velocidad de crucero del AUV respecto al agua [m/s].
        k_p: Coeficiente de costo en régimen de propulsión.
        k_r: Coeficiente de recuperación en régimen de regeneración.
        eta: Eficiencia de conversión de la regeneración, en (0, 1).
        e_max: Capacidad máxima de batería [J]; tope de energía recuperable.
        k_zonas: Número de zonas de convergencia a visitar (waypoints).
        resolucion_grados: Tamaño de celda de la malla [grados] (1/12).
    """

    s: float = 1.0
    k_p: float = 3.0
    k_r: float = 1.0
    eta: float = 0.3
    e_max: float = 2.0e6
    k_zonas: int = 6
    resolucion_grados: float = 1.0 / 12.0

    def __post_init__(self) -> None:
        """Valida los rangos de los parámetros (apoya el RNF-10)."""
        if not 0.0 < self.eta < 1.0:
            raise ValueError("eta debe estar en el intervalo (0, 1).")
        if self.s <= 0.0:
            raise ValueError("La velocidad de crucero s debe ser positiva.")
        if self.k_zonas < 2:
            raise ValueError("Se requieren al menos 2 zonas para una ruta.")
