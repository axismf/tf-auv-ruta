"""Identificación de las zonas prioritarias de muestreo (RF-03)."""
from __future__ import annotations

from .dependencias import *
from .grafo import Nodo


__all__ = [
    "divergencia",
    "seleccionar_waypoints",
    "seleccionar_centinelas",
    "celda_mas_cercana",
    "agregar_puntos_fijos",
]


def divergencia(
    uo: np.ndarray,
    vo: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Divergencia horizontal por diferencias finitas. Negativo = convergencia.

    div = ∂uo/∂x + ∂vo/∂y
    """
    return np.gradient(uo, dx, axis=1) + np.gradient(vo, dy, axis=0)


def seleccionar_waypoints(
    div: np.ndarray,
    navegable: np.ndarray,
    k: int,
    capa: int = 0,
    dist_min_celdas: int = 3,
) -> list[Nodo]:
    """Selecciona las k celdas de mayor convergencia como waypoints.

    Aplica separación mínima para no elegir celdas contiguas de la misma zona.

    Args:
        div:             Campo de divergencia (n_lat, n_lon).
        navegable:       Máscara de celdas navegables (n_lat, n_lon).
        k:               Número de zonas a seleccionar.
        capa:            Índice de profundidad al que pertenecen las celdas.
        dist_min_celdas: Separación mínima en celdas entre dos waypoints.
    """
    n_lat, n_lon    = div.shape
    div_filtrada    = np.where(navegable, div, np.inf)
    idx_flat_sorted = np.argsort(div_filtrada.ravel())

    seleccionados: list[Nodo] = []
    for flat in idx_flat_sorted:
        if len(seleccionados) == k:
            break
        if div_filtrada.ravel()[flat] >= 0:
            break
        i, j = divmod(int(flat), n_lon)
        demasiado_cerca = any(
            math.hypot(i - pi, j - pj) < dist_min_celdas
            for _, pi, pj in seleccionados
        )
        if not demasiado_cerca:
            seleccionados.append((capa, i, j))

    return seleccionados


def seleccionar_centinelas(
    uo: np.ndarray,
    vo: np.ndarray,
    navegable: np.ndarray,
    lon: np.ndarray,
    n: int = 2,
    capa: int = 0,
    fraccion_offshore: float = 0.4,
) -> list[Nodo]:
    """Selecciona n centinelas offshore para detección temprana.

    Busca celdas en la franja occidental donde la corriente entrante
    (uo > 0, hacia la costa) es más rápida.

    Args:
        uo:                Componente zonal en la capa (n_lat, n_lon).
        vo:                Componente meridional (n_lat, n_lon).
        navegable:         Máscara de celdas navegables (n_lat, n_lon).
        lon:               Vector de longitudes de la malla.
        n:                 Número de centinelas a seleccionar.
        capa:              Índice de profundidad.
        fraccion_offshore: Fracción del dominio occidental considerada offshore.
    """
    n_lat, n_lon = uo.shape

    n_cols_offshore  = max(1, int(n_lon * fraccion_offshore))
    mascara_offshore = np.zeros((n_lat, n_lon), dtype=bool)
    mascara_offshore[:, :n_cols_offshore] = True

    candidata       = mascara_offshore & navegable & (uo > 0)
    uo_candidatos   = np.where(candidata, uo, -np.inf)
    idx_flat_sorted = np.argsort(-uo_candidatos.ravel())

    dist_min = max(2, n_lat // (n + 1))
    seleccionados: list[Nodo] = []
    for flat in idx_flat_sorted:
        if len(seleccionados) == n:
            break
        if uo_candidatos.ravel()[flat] <= 0:
            break
        i, j = divmod(int(flat), n_lon)
        demasiado_cerca = any(
            math.hypot(i - pi, j - pj) < dist_min
            for _, pi, pj in seleccionados
        )
        if not demasiado_cerca:
            seleccionados.append((capa, i, j))

    return seleccionados


def celda_mas_cercana(
    lat_objetivo: float,
    lon_objetivo: float,
    lat: np.ndarray,
    lon: np.ndarray,
    navegable: np.ndarray,
    capa: int = 0,
) -> Nodo:
    """Celda navegable más próxima a las coordenadas dadas.

    Útil para fijar la base de la misión en un punto geográfico real.
    """
    dist          = (lat[:, None] - lat_objetivo) ** 2 + (lon[None, :] - lon_objetivo) ** 2
    dist_filtrada = np.where(navegable, dist, np.inf)
    flat          = int(np.argmin(dist_filtrada))
    i, j          = divmod(flat, len(lon))
    return (capa, i, j)


def agregar_puntos_fijos(
    waypoints: list[Nodo],
    fuente: Nodo,
    base: Nodo,
) -> tuple[list[Nodo], Nodo]:
    """Incorpora fuente de contaminación y base de partida sin duplicar.

    Returns:
        (lista completa de waypoints incluyendo fuente y base, celda base)
    """
    todos: list[Nodo] = list(waypoints)
    if fuente not in todos:
        todos.append(fuente)
    if base not in todos:
        todos.append(base)
    return todos, base


if __name__ == "__main__":
    # Ejecutar con: python -m src.zonas
    from src.datos import cargar_corrientes
    from src.config import ParametrosModelo

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
    todos, _ = agregar_puntos_fijos(wps + cent, wps[0], base)

    print(f"Waypoints convergencia: {len(wps)}")
    print(f"Centinelas offshore:    {len(cent)}")
    print(f"Total puntos misión:    {len(todos)}")
    print(f"Órdenes ATSP:           {math.factorial(len(todos) - 1)}")
    print("\n✓ zonas.py OK")
