"""Script de diagnóstico: visualiza el campo de corrientes, las zonas de
convergencia, los centinelas offshore y la base sobre el dominio de Lima.

TODO: borrar este archivo cuando visualizacion.py esté implementado.

Ejecutar desde la raíz del proyecto:
    python demo_grafo.py
"""
import math
import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.config import ParametrosModelo
from src.datos import cargar_corrientes
from src.zonas import (
    divergencia,
    seleccionar_waypoints,
    seleccionar_centinelas,
    celda_mas_cercana,
)

# --- Carga ---
nc = pathlib.Path("data/lima3.nc")
campo = cargar_corrientes(str(nc))
params = ParametrosModelo()

capa = 0
uo = campo.uo[capa]
vo = campo.vo[capa]
nav = campo.navegable[capa]

# --- Divergencia, waypoints, centinelas y base ---
lat_media = math.radians(float(campo.lat.mean()))
dy = abs(float(campo.lat[1] - campo.lat[0])) * 111_320.0
dx = abs(float(campo.lon[1] - campo.lon[0])) * 111_320.0 * math.cos(lat_media)

div = divergencia(uo, vo, dx, dy)
wps = seleccionar_waypoints(div, nav, params.k_zonas, capa=capa, dist_min_celdas=3)
centinelas = seleccionar_centinelas(uo, vo, nav, campo.lon, n=2, capa=capa)
base = celda_mas_cercana(-12.05, -77.15, campo.lat, campo.lon, nav, capa=capa)

# --- Figura ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

LON, LAT = np.meshgrid(campo.lon, campo.lat)


def _pintar_puntos(ax):
    """Pinta waypoints, centinelas y base sobre el eje dado."""
    for idx, (p, i, j) in enumerate(wps):
        ax.plot(campo.lon[j], campo.lat[i], "r*", markersize=13, zorder=5)
        ax.annotate(f"C{idx+1}", (campo.lon[j], campo.lat[i]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="red", fontweight="bold")
    for idx, (p, i, j) in enumerate(centinelas):
        ax.plot(campo.lon[j], campo.lat[i], "b^", markersize=10, zorder=5)
        ax.annotate(f"S{idx+1}", (campo.lon[j], campo.lat[i]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="blue", fontweight="bold")
    p, i, j = base
    ax.plot(campo.lon[j], campo.lat[i], "gs", markersize=11, zorder=5)
    ax.annotate("Base", (campo.lon[j], campo.lat[i]),
                textcoords="offset points", xytext=(5, 4),
                fontsize=8, color="green", fontweight="bold")


legend = [
    mpatches.Patch(color="red",   label="Convergencia (Cx)"),
    mpatches.Patch(color="blue",  label="Centinela offshore (Sx)"),
    mpatches.Patch(color="green", label="Base (Callao)"),
]

# ── Panel izquierdo: campo de corrientes ──────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#d4b483")

mag = np.sqrt(uo**2 + vo**2)
mag_masked = np.where(nav, mag, np.nan)

q = ax.quiver(
    LON, LAT,
    np.where(nav, uo, np.nan),
    np.where(nav, vo, np.nan),
    mag_masked,
    cmap="viridis", clim=(0, np.nanmax(mag_masked)),
    scale=5, width=0.003,
)
fig.colorbar(q, ax=ax, label="Rapidez [m/s]")
_pintar_puntos(ax)
ax.legend(handles=legend, loc="lower right", fontsize=8)
ax.set_title(f"Campo de corrientes — capa {capa} (prof. {campo.prof[capa]:.2f} m)")
ax.set_xlabel("Longitud [°]")
ax.set_ylabel("Latitud [°]")

# ── Panel derecho: divergencia ────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor("#d4b483")

div_masked = np.where(nav, div, np.nan)
vmax = np.nanpercentile(np.abs(div_masked), 95)

im = ax.pcolormesh(LON, LAT, div_masked,
                   cmap="RdBu", vmin=-vmax, vmax=vmax, shading="auto")
fig.colorbar(im, ax=ax, label="Divergencia [1/s]")

conv = np.where(nav & (div < 0), div, np.nan)
ax.contourf(LON, LAT, conv, levels=5, cmap="Blues_r", alpha=0.35)

_pintar_puntos(ax)
ax.legend(handles=legend, loc="lower right", fontsize=8)
ax.set_title(f"Divergencia — capa {capa}  (rojo=diverge, azul=converge)")
ax.set_xlabel("Longitud [°]")
ax.set_ylabel("Latitud [°]")

plt.tight_layout()
out = pathlib.Path("TF/demo_grafo.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Imagen guardada en {out}")
plt.show()
