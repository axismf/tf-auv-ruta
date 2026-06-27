"""Construcción del grafo dirigido ponderado y función de costo (RF-03, RF-04)."""
from __future__ import annotations

from .dependencias import *
from .config import ParametrosModelo
from .datos import CampoCorrientes


__all__ = ["GrafoBase", "Grafo", "Nodo", "construir_grafo", "costo_arista"]

Nodo = tuple[int, int, int]  # (prof_idx, lat_idx, lon_idx)


class GrafoBase(ABC):
    """Contrato del grafo: interfaz que toda implementación debe cumplir."""

    @abstractmethod
    def agregar_nodo(self, u: Nodo) -> None: ...

    @abstractmethod
    def agregar_arista(self, u: Nodo, v: Nodo, peso: float) -> None: ...

    @abstractmethod
    def vecinos(self, u: Nodo) -> list[tuple[Nodo, float]]: ...

    @abstractmethod
    def nodos(self) -> Iterator[Nodo]: ...

    @abstractmethod
    def aristas(self) -> Iterator[tuple[Nodo, Nodo, float]]: ...

    @abstractmethod
    def peso(self, u: Nodo, v: Nodo) -> float | None: ...

    @property
    @abstractmethod
    def num_nodos(self) -> int: ...

    @property
    @abstractmethod
    def num_aristas(self) -> int: ...


class Grafo(GrafoBase):
    """Grafo dirigido y ponderado por lista de adyacencia.

    Pesos pueden ser negativos (regeneración) → usar Bellman-Ford, no Dijkstra.
    """

    def __init__(self) -> None:
        self._adyacencia: dict[Nodo, list[tuple[Nodo, float]]] = {}

    def agregar_nodo(self, u: Nodo) -> None:
        """Registra el nodo u sin aristas si aún no existe."""
        self._adyacencia.setdefault(u, [])

    def agregar_arista(self, u: Nodo, v: Nodo, peso: float) -> None:
        """Agrega la arista dirigida u → v con el peso dado."""
        self._adyacencia.setdefault(u, []).append((v, peso))
        self._adyacencia.setdefault(v, [])

    def vecinos(self, u: Nodo) -> list[tuple[Nodo, float]]:
        """Aristas salientes (v, peso) del nodo u."""
        return self._adyacencia.get(u, [])

    def nodos(self) -> Iterator[Nodo]:
        return iter(self._adyacencia)

    def aristas(self) -> Iterator[tuple[Nodo, Nodo, float]]:
        for u, lista in self._adyacencia.items():
            for v, p in lista:
                yield u, v, p

    def peso(self, u: Nodo, v: Nodo) -> float | None:
        """Peso de la arista u → v, o None si la arista no existe."""
        for dest, w in self._adyacencia.get(u, []):
            if dest == v:
                return w
        return None

    @property
    def num_nodos(self) -> int:
        return len(self._adyacencia)

    @property
    def num_aristas(self) -> int:
        return sum(len(lista) for lista in self._adyacencia.values())


def construir_grafo(campo: CampoCorrientes, params: ParametrosModelo) -> Grafo:
    """Construye el grafo dirigido completo del dominio marino.

    Conecta cada celda navegable con sus hasta 26 vecinos en 3D.
    Las celdas de tierra actúan como obstáculos y se omiten.
    """
    grafo   = Grafo()
    n_prof, n_lat, n_lon = campo.uo.shape
    nav     = campo.navegable
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
                        grafo.agregar_arista(
                            origen, destino,
                            costo_arista(origen, destino, campo, params),
                        )
    return grafo


def costo_arista(
    origen: Nodo,
    destino: Nodo,
    campo: CampoCorrientes,
    params: ParametrosModelo,
) -> float:
    """Energía neta [J] de recorrer la arista origen → destino.

    Positivo = propulsión (consume batería), negativo = regeneración (recarga).
    """
    pa, ia, ja = origen
    pb, ib, jb = destino

    lat_media = math.radians((campo.lat[ia] + campo.lat[ib]) / 2.0)
    dx = (campo.lon[jb] - campo.lon[ja]) * GRADOS_A_METROS * math.cos(lat_media)
    dy = (campo.lat[ib] - campo.lat[ia]) * GRADOS_A_METROS
    dz =  campo.prof[pb] - campo.prof[pa]

    longitud = math.sqrt(dx*dx + dy*dy + dz*dz)
    if longitud == 0.0:
        return 0.0

    ex, ey, ez = dx / longitud, dy / longitud, dz / longitud
    cx = float(campo.uo[pa, ia, ja])
    cy = float(campo.vo[pa, ia, ja])

    # v_r = s·ê − v_c  (velocidad relativa al agua)
    rx, ry, rz = params.s * ex - cx, params.s * ey - cy, params.s * ez
    vr3        = (rx*rx + ry*ry + rz*rz) ** 1.5
    v_paralela = cx * ex + cy * ey
    tiempo     = longitud / params.s

    if v_paralela < params.s:
        return params.k_p * vr3 * tiempo

    return -min(params.k_r * params.eta * vr3 * tiempo, params.e_max)


if __name__ == "__main__":
    # Ejecutar con: python -m src.grafo
    from src.datos import cargar_corrientes

    nc    = pathlib.Path(__file__).parent.parent / "data" / "lima3.nc"
    campo = cargar_corrientes(str(nc))
    grafo = construir_grafo(campo, ParametrosModelo())
    print(f"Nodos: {grafo.num_nodos}  Aristas: {grafo.num_aristas}")
    print("\n✓ grafo.py OK")
