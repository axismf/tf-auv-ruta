"""Cálculo y exportación de métricas de la misión (RF-08, RF-09)."""
from __future__ import annotations

from .dependencias import *
from .datos import CampoCorrientes
from .grafo import Grafo, Nodo


__all__ = [
    "EstadoBateria",
    "resumen_mision",
    "estado_bateria",
    "exportar_csv",
    "costos_por_tramo",
    "energia_total",
]


@dataclass
class EstadoBateria:
    """Estado energético del AUV a lo largo de la ruta."""

    niveles:    list[float]
    consumido:  float
    regenerado: float
    minimo:     float
    viable:     bool


def resumen_mision(
    orden: list[int],
    waypoints: list[Nodo],
    matriz: np.ndarray,
    campo: CampoCorrientes,
) -> str:
    """Texto formateado con orden de visita, costo por tramo y total."""
    tramos = costos_por_tramo(orden, matriz)
    total  = sum(tramos)
    lineas = ["── Resultado de la misión ──────────────────────"]
    for k, (i, j) in enumerate(zip(orden, orden[1:])):
        _, li, lj = waypoints[i]
        _, di, dj = waypoints[j]
        lineas.append(
            f"  Tramo {k+1}: zona {i} ({campo.lat[li]:.2f}°, {campo.lon[lj]:.2f}°)"
            f" → zona {j} ({campo.lat[di]:.2f}°, {campo.lon[dj]:.2f}°)"
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
) -> EstadoBateria:
    """Simula el estado de batería del AUV a lo largo de la ruta.

    Pesos positivos consumen; negativos recargan. Carga acotada en [0, e_max].

    Args:
        ruta:            Secuencia de nodos de la ruta completa.
        grafo:           Grafo del que se leen los pesos.
        e_max:           Capacidad máxima de la batería [J].
        bateria_inicial: Carga al inicio [J]. Si None usa e_max (llena).

    Returns:
        EstadoBateria con niveles, consumido, regenerado, minimo y viable.
    """
    if bateria_inicial is None:
        bateria_inicial = e_max

    niveles    = [bateria_inicial]
    consumido  = 0.0
    regenerado = 0.0

    for u, v in zip(ruta, ruta[1:]):
        w = grafo.peso(u, v)
        if w is None:
            raise ValueError(f"Arista {u} → {v} no existe en el grafo.")
        nueva = min(niveles[-1] - w, e_max)
        if w > 0:
            consumido  += w
        else:
            regenerado += -w
        niveles.append(max(nueva, 0.0))

    return EstadoBateria(
        niveles=niveles,
        consumido=consumido,
        regenerado=regenerado,
        minimo=min(niveles),
        viable=min(niveles) > 0,
    )


def exportar_csv(
    ruta: list[Nodo],
    campo: CampoCorrientes,
    ruta_salida: str,
) -> None:
    """Exporta la ruta a un CSV con columnas: paso, prof_m, lat_deg, lon_deg."""
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


def costos_por_tramo(
    orden: list[int],
    matriz: np.ndarray,
) -> list[float]:
    """Costo energético de cada tramo entre zonas consecutivas."""
    return [float(matriz[orden[i], orden[i + 1]]) for i in range(len(orden) - 1)]


def energia_total(
    ruta: list[Nodo],
    grafo: Grafo,
) -> float:
    """Suma la energía neta de todos los tramos (verificación nodo a nodo).

    Raises:
        ValueError: Si alguna arista de la ruta no existe en el grafo.
    """
    total = 0.0
    for u, v in zip(ruta, ruta[1:]):
        w = grafo.peso(u, v)
        if w is None:
            raise ValueError(f"Arista {u} → {v} no existe en el grafo.")
        total += w
    return total


if __name__ == "__main__":
    # Ejecutar con: python -m src.metricas
    from src.datos import cargar_corrientes
    from src.config import ParametrosModelo
    from src.grafo import construir_grafo
    from src.zonas import (
        divergencia, seleccionar_waypoints, seleccionar_centinelas,
        celda_mas_cercana, agregar_puntos_fijos,
    )
    from src.algoritmos import matriz_costos, atsp_fuerza_bruta, ensamblar_ruta

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

    grafo      = construir_grafo(campo, params)
    M, caminos = matriz_costos(grafo, todos)
    orden, _   = atsp_fuerza_bruta(M, base=base_idx)
    ruta       = ensamblar_ruta(orden, caminos)

    print(resumen_mision(orden, todos, M, campo))
    print(f"\nVerificación nodo a nodo: {energia_total(ruta, grafo):.1f} J")

    out = pathlib.Path(__file__).parent.parent / "outputs" / "rutas" / "ruta.csv"
    out.parent.mkdir(exist_ok=True)
    exportar_csv(ruta, campo, str(out))
    print(f"CSV exportado → {out}")
    print("\n✓ metricas.py OK")
