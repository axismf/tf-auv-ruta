"""Carga y preprocesamiento del dataset de corrientes marinas (RF-01, RF-02)."""
from __future__ import annotations

from .dependencias import *


__all__ = ["CampoCorrientes", "cargar_corrientes", "marcar_navegables", "resumen"]

# Nombres alternativos de cada magnitud en productos CMEMS; se prueban en orden.
_ALIAS: dict[str, tuple[str, ...]] = {
    "lat":  ("latitude",  "lat",  "nav_lat", "y"),
    "lon":  ("longitude", "lon",  "nav_lon", "x"),
    "prof": ("depth",     "deptht", "lev",   "z"),
    "time": ("time",      "time_counter",     "t"),
    "uo":   ("uo", "u", "utotal", "eastward_sea_water_velocity"),
    "vo":   ("vo", "v", "vtotal", "northward_sea_water_velocity"),
}


@dataclass(frozen=True)
class CampoCorrientes:
    """Campo de corrientes marinas sobre una malla regular.

    Ejes de uo/vo: (profundidad, latitud, longitud). Latitud en orden creciente.

    Attributes:
        lat:       Latitudes [°], creciente. Shape (n_lat,).
        lon:       Longitudes [°]. Shape (n_lon,).
        prof:      Profundidades [m]. Shape (n_prof,).
        uo:        Componente zonal [m/s]. Shape (n_prof, n_lat, n_lon).
        vo:        Componente meridional [m/s]. Igual shape que uo.
        navegable: Máscara booleana de celdas de agua. Igual shape que uo.
    """

    lat: np.ndarray
    lon: np.ndarray
    prof: np.ndarray
    uo: np.ndarray
    vo: np.ndarray
    navegable: np.ndarray


def cargar_corrientes(
    ruta_nc: str,
    paso_tiempo: int | Literal["media"] = "media",
) -> CampoCorrientes:
    """Carga el NetCDF de CMEMS y devuelve el campo de corrientes listo para usar.

    Args:
        ruta_nc:     Ruta al archivo .nc de Copernicus Marine.
        paso_tiempo: Índice temporal, o "media" para promediar sobre el tiempo.
    """
    with xr.open_dataset(ruta_nc) as ds:
        n_lat  = _resolver_nombre(ds, "lat")
        n_lon  = _resolver_nombre(ds, "lon")
        n_prof = _resolver_nombre(ds, "prof")
        n_uo   = _resolver_nombre(ds, "uo")
        n_vo   = _resolver_nombre(ds, "vo")

        uo = ds[n_uo]
        vo = ds[n_vo]

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

        uo = uo.transpose(n_prof, n_lat, n_lon)
        vo = vo.transpose(n_prof, n_lat, n_lon)

        lat  = np.asarray(ds[n_lat].values,  dtype=float)
        lon  = np.asarray(ds[n_lon].values,  dtype=float)
        prof = np.asarray(ds[n_prof].values, dtype=float)
        uo_arr = np.ma.filled(uo.values, np.nan).astype(float)
        vo_arr = np.ma.filled(vo.values, np.nan).astype(float)

    if lat.size > 1 and lat[0] > lat[-1]:
        lat    = lat[::-1]
        uo_arr = uo_arr[:, ::-1, :]
        vo_arr = vo_arr[:, ::-1, :]

    return CampoCorrientes(lat, lon, prof, uo_arr, vo_arr, marcar_navegables(uo_arr, vo_arr))


def marcar_navegables(uo: np.ndarray, vo: np.ndarray) -> np.ndarray:
    """Devuelve máscara booleana True donde la celda tiene dato de corriente."""
    return ~(np.isnan(uo) | np.isnan(vo))


def resumen(campo: CampoCorrientes) -> str:
    """Texto de verificación rápida del campo cargado."""
    n_prof, n_lat, n_lon = campo.uo.shape
    total = campo.navegable.size
    agua  = int(campo.navegable.sum())
    mag_v = np.sqrt(campo.uo ** 2 + campo.vo ** 2)[campo.navegable]
    return (
        f"Malla: {n_prof}p × {n_lat}lat × {n_lon}lon = {total} celdas\n"
        f"Agua: {agua}  |  Tierra: {total - agua} ({100*(total-agua)/total:.1f} %)\n"
        f"Lat: {campo.lat.min():.2f} .. {campo.lat.max():.2f}\n"
        f"Lon: {campo.lon.min():.2f} .. {campo.lon.max():.2f}\n"
        f"Prof [m]: {np.round(campo.prof, 2).tolist()}\n"
        f"Rapidez [m/s]: media {mag_v.mean():.3f}  máx {mag_v.max():.3f}"
    )


def _resolver_nombre(ds: xr.Dataset, clave: str) -> str:
    """Encuentra en el dataset el nombre real de una magnitud por sus alias.

    Raises:
        KeyError: Si ningún alias conocido está presente en el dataset.
    """
    disponibles = set(ds.variables) | set(ds.coords)
    for alias in _ALIAS[clave]:
        if alias in disponibles:
            return alias
    raise KeyError(
        f"Variable '{clave}' no encontrada. "
        f"Alias buscados: {_ALIAS[clave]}. Presentes: {sorted(disponibles)}."
    )


if __name__ == "__main__":
    # Ejecutar con: python -m src.datos
    nc = pathlib.Path(__file__).parent.parent / "data" / "lima3.nc"
    print(f"Cargando {nc} …")
    campo = cargar_corrientes(str(nc))
    print(resumen(campo))
    print("\n✓ datos.py OK")
