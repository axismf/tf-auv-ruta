"""Identificación de las zonas prioritarias de muestreo (RF-03).

Las zonas de interés se derivan del propio campo de corrientes mediante la
divergencia horizontal: donde el flujo converge, los contaminantes se
acumulan, de modo que esas celdas son los waypoints de la misión. Se
complementan con centinelas offshore para detección temprana de derrames.
"""
from __future__ import annotations

# TODO: borrar este bloque cuando termines de probar zonas.py
if __package__ is None or __package__ == "":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    __package__ = "src"
# TODO: fin bloque fix __package__

import math
import numpy as np

# Una celda se identifica por su terna de índices en la malla (prof, lat, lon).
Celda = tuple[int, int, int]

# Metros por grado de latitud.
_GRADOS_A_METROS = 111_320.0


def divergencia(uo: np.ndarray, vo: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Calcula la divergencia horizontal del flujo por diferencias finitas (RF-03).

    div(x, y) = d(uo)/dx + d(vo)/dy

    Usa diferencias centradas en el interior y diferencias de un lado en los
    bordes para conservar la forma de la malla.

    Args:
        uo: Componente zonal en una capa (n_lat, n_lon).
        vo: Componente meridional en una capa (n_lat, n_lon).
        dx: Separación entre celdas en x [m].
        dy: Separación entre celdas en y [m].

    Returns:
        Campo de divergencia (n_lat, n_lon). Valores negativos = convergencia.
    """
    duo_dx = np.gradient(uo, dx, axis=1)   # derivada en dirección lon (x)
    dvo_dy = np.gradient(vo, dy, axis=0)   # derivada en dirección lat (y)
    return duo_dx + dvo_dy


def seleccionar_waypoints(
    div: np.ndarray,
    navegable: np.ndarray,
    k: int,
    capa: int = 0,
    dist_min_celdas: int = 3,
) -> list[Celda]:
    """Selecciona las k celdas de mayor convergencia como waypoints (RF-03).

    Aplica un criterio de separación mínima entre candidatos para evitar
    seleccionar celdas contiguas que representen la misma zona de acumulación.

    Args:
        div: Campo de divergencia de la capa de interés (n_lat, n_lon).
        navegable: Máscara de celdas navegables de esa capa (n_lat, n_lon).
        k: Número de zonas a seleccionar.
        capa: Índice de profundidad al que pertenecen las celdas.
        dist_min_celdas: Distancia mínima en celdas entre dos waypoints
            seleccionados. Evita elegir celdas contiguas de la misma zona.

    Returns:
        Lista de hasta k celdas (prof, lat, lon) con divergencia más negativa
        y separadas entre sí al menos dist_min_celdas.
    """
    n_lat, n_lon = div.shape

    # Candidatos ordenados de mayor a menor convergencia (más negativo primero).
    div_filtrada = np.where(navegable, div, np.inf)
    idx_flat_sorted = np.argsort(div_filtrada.ravel())

    seleccionados: list[Celda] = []
    for flat in idx_flat_sorted:
        if len(seleccionados) == k:
            break
        if div_filtrada.ravel()[flat] >= 0:
            break                           # ya no hay convergencia real
        i, j = divmod(int(flat), n_lon)
        # Rechazar si está demasiado cerca de un waypoint ya elegido.
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
) -> list[Celda]:
    """Selecciona n centinelas offshore para detección temprana (RF-03).

    Busca celdas en la franja más occidental del dominio (zona abierta al
    océano, lejos de la costa) donde la corriente entrante es más rápida,
    es decir, la componente zonal uo es más positiva (flujo hacia el este,
    hacia la costa). Esto identifica los puntos donde una mancha de
    contaminante offshore llegaría antes.

    Los centinelas se distribuyen espacialmente usando el mismo criterio de
    distancia mínima que seleccionar_waypoints.

    Args:
        uo: Componente zonal en la capa (n_lat, n_lon).
        vo: Componente meridional en la capa (n_lat, n_lon).
        navegable: Máscara de celdas navegables de esa capa (n_lat, n_lon).
        lon: Vector de longitudes de la malla.
        n: Número de centinelas a seleccionar.
        capa: Índice de profundidad al que pertenecen las celdas.
        fraccion_offshore: Fracción del dominio (desde el oeste) considerada
            zona offshore. 0.4 = 40 % más occidental de la malla.

    Returns:
        Lista de n celdas (prof, lat, lon) centinelas offshore.
    """
    n_lat, n_lon = uo.shape

    # Columnas de la franja offshore (las más al oeste, lon más pequeño).
    n_cols_offshore = max(1, int(n_lon * fraccion_offshore))
    mascara_offshore = np.zeros((n_lat, n_lon), dtype=bool)
    mascara_offshore[:, :n_cols_offshore] = True

    # Candidatos: offshore, navegable, con corriente entrante (uo > 0).
    candidata = mascara_offshore & navegable & (uo > 0)

    # Ordenar por rapidez de la componente entrante (mayor uo primero).
    uo_candidatos = np.where(candidata, uo, -np.inf)
    idx_flat_sorted = np.argsort(-uo_candidatos.ravel())

    dist_min = max(2, n_lat // (n + 1))    # separación mínima adaptativa
    seleccionados: list[Celda] = []
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
) -> Celda:
    """Devuelve la celda navegable más cercana a unas coordenadas dadas.

    Útil para fijar la base de la misión en un punto geográfico real
    (por ejemplo, el puerto del Callao).

    Args:
        lat_objetivo: Latitud deseada en grados.
        lon_objetivo: Longitud deseada en grados.
        lat: Vector de latitudes de la malla.
        lon: Vector de longitudes de la malla.
        navegable: Máscara de celdas navegables de la capa (n_lat, n_lon).
        capa: Índice de profundidad.

    Returns:
        Celda (prof, lat_idx, lon_idx) navegable más próxima al objetivo.
    """
    dist = (lat[:, None] - lat_objetivo) ** 2 + (lon[None, :] - lon_objetivo) ** 2
    dist_filtrada = np.where(navegable, dist, np.inf)
    flat = int(np.argmin(dist_filtrada))
    i, j = divmod(flat, len(lon))
    return (capa, i, j)


def agregar_puntos_fijos(
    waypoints: list[Celda],
    fuente: Celda,
    base: Celda,
) -> tuple[list[Celda], Celda]:
    """Incorpora la fuente de contaminación y la base de partida (RF-03).

    Si fuente o base ya están en la lista, no se duplican.

    Args:
        waypoints: Zonas obtenidas por convergencia y centinelas.
        fuente: Celda de la fuente de contaminación conocida.
        base: Celda de partida y retorno de la misión.

    Returns:
        Tupla (lista completa de waypoints incluyendo fuente y base, celda base).
    """
    todos: list[Celda] = list(waypoints)
    if fuente not in todos:
        todos.append(fuente)
    if base not in todos:
        todos.append(base)
    return todos, base


# TODO: borrar este bloque cuando termines de probar zonas.py
if __name__ == "__main__":
    import pathlib
    from src.datos import cargar_corrientes, resumen
    from src.config import ParametrosModelo

    nc = pathlib.Path(__file__).parent.parent / "data" / "lima3.nc"
    print(f"Cargando {nc} …")
    campo = cargar_corrientes(str(nc))
    print(resumen(campo))

    params = ParametrosModelo()
    capa = 0

    lat_media = math.radians(float(campo.lat.mean()))
    dy = abs(float(campo.lat[1] - campo.lat[0])) * _GRADOS_A_METROS
    dx = abs(float(campo.lon[1] - campo.lon[0])) * _GRADOS_A_METROS * math.cos(lat_media)

    uo = campo.uo[capa]
    vo = campo.vo[capa]
    nav = campo.navegable[capa]

    div = divergencia(uo, vo, dx, dy)

    # Waypoints de convergencia deduplicados
    wps = seleccionar_waypoints(div, nav, params.k_zonas, capa=capa, dist_min_celdas=3)
    print(f"\nWaypoints de convergencia ({len(wps)}):")
    for wp in wps:
        p, i, j = wp
        print(f"  {wp}  lat={campo.lat[i]:.3f}°  lon={campo.lon[j]:.3f}°  div={div[i,j]:.6f}")

    # Centinelas offshore
    centinelas = seleccionar_centinelas(uo, vo, nav, campo.lon, n=2, capa=capa)
    print(f"\nCentinelas offshore ({len(centinelas)}):")
    for c in centinelas:
        p, i, j = c
        print(f"  {c}  lat={campo.lat[i]:.3f}°  lon={campo.lon[j]:.3f}°  uo={uo[i,j]:.3f} m/s")

    # Base en el Callao
    base = celda_mas_cercana(-12.05, -77.15, campo.lat, campo.lon, nav, capa=capa)
    p, i, j = base
    print(f"\nBase (Callao): {base}  lat={campo.lat[i]:.3f}°  lon={campo.lon[j]:.3f}°")

    # Lista completa
    todos, base_ret = agregar_puntos_fijos(wps + centinelas, wps[0], base)
    print(f"\nTotal waypoints de misión: {len(todos)}")
    print(f"Órdenes ATSP a evaluar:   {math.factorial(len(todos) - 1)}")

    print("\n✓ zonas.py OK")
# TODO: fin bloque de prueba
