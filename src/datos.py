"""Carga y preprocesamiento del dataset de corrientes (RF-01, RF-02).

Lee el producto NetCDF de Copernicus Marine (CMEMS) y entrega el campo de
corrientes ya estructurado, con las celdas de tierra marcadas como no
navegables. La carga es tolerante a las variantes de nombres de variables y
coordenadas que usan los distintos productos CMEMS.
"""
from __future__ import annotations

from dataclasses import dataclass

if __package__ is None or __package__ == "":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    __package__ = "src"

import numpy as np
import xarray as xr

# Nombres alternativos con los que cada magnitud puede aparecer en un NetCDF
# de CMEMS. Se prueba en orden hasta encontrar el primero presente.
_ALIAS = {
    "lat": ("latitude", "lat", "nav_lat", "y"),
    "lon": ("longitude", "lon", "nav_lon", "x"),
    "prof": ("depth", "deptht", "lev", "z"),
    "time": ("time", "time_counter", "t"),
    "uo": ("uo", "u", "utotal", "eastward_sea_water_velocity"),
    "vo": ("vo", "v", "vtotal", "northward_sea_water_velocity"),
}


@dataclass(frozen=True)
class CampoCorrientes:
    """Campo de corrientes marinas sobre una malla regular.

    Convención de ejes para los arreglos uo/vo: (profundidad, latitud, longitud),
    con la latitud en orden creciente (sur -> norte).

    Attributes:
        lat: Latitudes de la malla [grados], orden creciente. Shape (n_lat,).
        lon: Longitudes de la malla [grados]. Shape (n_lon,).
        prof: Niveles de profundidad [m]. Shape (n_prof,).
        uo: Componente zonal de la corriente [m/s]. Shape (n_prof, n_lat, n_lon).
        vo: Componente meridional de la corriente [m/s]. Misma shape que uo.
        navegable: Máscara booleana de celdas de agua. Misma shape que uo.
    """

    lat: np.ndarray
    lon: np.ndarray
    prof: np.ndarray
    uo: np.ndarray
    vo: np.ndarray
    navegable: np.ndarray


def _resolver_nombre(ds: xr.Dataset, clave: str) -> str:
    """Encuentra en el dataset el nombre real de una magnitud por sus alias.

    Args:
        ds: Dataset abierto.
        clave: Clave lógica ("lat", "lon", "prof", "time", "uo", "vo").

    Returns:
        El nombre de la variable/coordenada tal como aparece en el dataset.

    Raises:
        KeyError: Si ninguno de los alias conocidos está presente.
    """
    disponibles = set(ds.variables) | set(ds.coords)
    for alias in _ALIAS[clave]:
        if alias in disponibles:
            return alias
    raise KeyError(
        f"No se encontró ninguna variable para '{clave}'. "
        f"Alias buscados: {_ALIAS[clave]}. Presentes: {sorted(disponibles)}."
    )


def marcar_navegables(uo: np.ndarray, vo: np.ndarray) -> np.ndarray:
    """Marca como no navegables las celdas de tierra (RF-02).

    Una celda es tierra si la corriente no está definida (NaN) en uo o vo.

    Args:
        uo: Componente zonal de la corriente.
        vo: Componente meridional de la corriente.

    Returns:
        Máscara booleana: True donde la celda es agua navegable.
    """
    return ~(np.isnan(uo) | np.isnan(vo))


def cargar_corrientes(ruta_nc: str, paso_tiempo: int | str = "media") -> CampoCorrientes:
    """Carga el NetCDF y devuelve el campo de corrientes (RF-01).

    Args:
        ruta_nc: Ruta al archivo .nc de Copernicus Marine.
        paso_tiempo: Índice del paso temporal a usar, o "media" para promediar
            sobre toda la dimensión de tiempo y obtener un campo estático. El
            promedio propaga los NaN (skipna=False), de modo que la tierra
            permanece como tierra en todos los pasos.

    Returns:
        Un CampoCorrientes con la máscara de navegabilidad ya calculada y la
        latitud en orden creciente.
    """
    with xr.open_dataset(ruta_nc) as ds:
        n_lat = _resolver_nombre(ds, "lat")
        n_lon = _resolver_nombre(ds, "lon")
        n_prof = _resolver_nombre(ds, "prof")
        n_uo = _resolver_nombre(ds, "uo")
        n_vo = _resolver_nombre(ds, "vo")

        uo = ds[n_uo]
        vo = ds[n_vo]

        # Reducir la dimensión temporal a un campo estático, si existe.
        try:
            n_time = _resolver_nombre(ds, "time")
        except KeyError:
            n_time = None
        if n_time is not None and n_time in uo.dims:
            if paso_tiempo == "media":
                uo = uo.mean(dim=n_time, skipna=False)
                vo = vo.mean(dim=n_time, skipna=False)
            else:
                uo = uo.isel({n_time: int(paso_tiempo)})
                vo = vo.isel({n_time: int(paso_tiempo)})

        # Asegurar el orden de ejes (profundidad, latitud, longitud).
        uo = uo.transpose(n_prof, n_lat, n_lon)
        vo = vo.transpose(n_prof, n_lat, n_lon)

        lat = np.asarray(ds[n_lat].values, dtype=float)
        lon = np.asarray(ds[n_lon].values, dtype=float)
        prof = np.asarray(ds[n_prof].values, dtype=float)
        uo_arr = np.ma.filled(uo.values, np.nan).astype(float)
        vo_arr = np.ma.filled(vo.values, np.nan).astype(float)

    # Normalizar la latitud a orden creciente (sur -> norte).
    if lat.size > 1 and lat[0] > lat[-1]:
        lat = lat[::-1]
        uo_arr = uo_arr[:, ::-1, :]
        vo_arr = vo_arr[:, ::-1, :]

    navegable = marcar_navegables(uo_arr, vo_arr)
    return CampoCorrientes(lat, lon, prof, uo_arr, vo_arr, navegable)


def resumen(campo: CampoCorrientes) -> str:
    """Devuelve un resumen legible del campo para verificación rápida.

    Args:
        campo: Campo de corrientes cargado.

    Returns:
        Texto con dimensiones, porcentaje de tierra y estadísticos de rapidez.
    """
    n_prof, n_lat, n_lon = campo.uo.shape
    total = campo.navegable.size
    agua = int(campo.navegable.sum())
    tierra = total - agua
    mag = np.sqrt(campo.uo ** 2 + campo.vo ** 2)
    mag_validas = mag[campo.navegable]
    return (
        f"Malla: {n_prof} prof x {n_lat} lat x {n_lon} lon = {total} celdas\n"
        f"Agua navegable: {agua}  |  Tierra (NaN): {tierra} "
        f"({100 * tierra / total:.1f} %)\n"
        f"Latitud:  {campo.lat.min():.2f} .. {campo.lat.max():.2f}\n"
        f"Longitud: {campo.lon.min():.2f} .. {campo.lon.max():.2f}\n"
        f"Profundidades [m]: {np.round(campo.prof, 2).tolist()}\n"
        f"Rapidez [m/s]: media {mag_validas.mean():.3f}  máx {mag_validas.max():.3f}"
    )


if __name__ == "__main__":
    import pathlib
    nc = pathlib.Path(__file__).parent.parent / "data" / "lima3.nc"
    print(f"Cargando {nc} …")
    campo = cargar_corrientes(str(nc))
    print(resumen(campo))
    print("\n✓ datos.py OK")