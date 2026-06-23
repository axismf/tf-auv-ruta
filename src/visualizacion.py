"""Visualizaciones del dominio para depuración y para el informe (RF-07).

Estos plots de matplotlib se usan durante el desarrollo para verificar cada
etapa (¿los waypoints caen donde deben?, ¿la ruta evita la tierra?) y, ya
pulidos, como figuras del informe. La capa Streamlit (app.py) reutiliza estas
mismas funciones, de modo que la visualización no depende de la UI.
"""
from __future__ import annotations

if __package__ is None or __package__ == "":
    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    __package__ = "src"

import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

from .datos import CampoCorrientes

# Una celda/nodo se identifica por su terna (prof, lat, lon).
Celda = tuple[int, int, int]

_TIERRA = "#d4b483"
_COLORES_PROF = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]   # una por capa


def _meshgrid(campo: CampoCorrientes):
    return np.meshgrid(campo.lon, campo.lat)


def plot_campo(
    campo: CampoCorrientes,
    capa: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dibuja el campo de corrientes como quiver sobre el dominio (RF-07).

    Args:
        campo: Campo de corrientes.
        capa: Índice de profundidad a graficar.
        ax: Ejes opcionales donde dibujar; si es None se crean nuevos.

    Returns:
        Los ejes con el campo dibujado.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    LON, LAT = _meshgrid(campo)
    uo  = campo.uo[capa]
    vo  = campo.vo[capa]
    nav = campo.navegable[capa]
    mag = np.where(nav, np.sqrt(uo**2 + vo**2), np.nan)

    ax.set_facecolor(_TIERRA)
    q = ax.quiver(
        LON, LAT,
        np.where(nav, uo, np.nan),
        np.where(nav, vo, np.nan),
        mag,
        cmap="viridis", clim=(0, np.nanmax(mag)),
        scale=5, width=0.003,
    )
    plt.colorbar(q, ax=ax, label="Rapidez [m/s]")
    ax.set_title(f"Campo de corrientes — capa {capa} (prof. {campo.prof[capa]:.2f} m)")
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    return ax


def plot_zonas(
    campo: CampoCorrientes,
    div: np.ndarray,
    waypoints: list[Celda],
    centinelas: list[Celda] | None = None,
    base: Celda | None = None,
    capa: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dibuja la divergencia y resalta las zonas seleccionadas (RF-07).

    Args:
        campo: Campo de corrientes.
        div: Campo de divergencia de la capa (n_lat, n_lon).
        waypoints: Celdas de convergencia seleccionadas.
        centinelas: Celdas centinela offshore (opcional).
        base: Celda base de la misión (opcional).
        capa: Índice de profundidad a graficar.
        ax: Ejes opcionales donde dibujar.

    Returns:
        Los ejes con las zonas dibujadas.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    LON, LAT = _meshgrid(campo)
    nav = campo.navegable[capa]
    div_masked = np.where(nav, div, np.nan)
    vmax = np.nanpercentile(np.abs(div_masked), 95)

    ax.set_facecolor(_TIERRA)
    im = ax.pcolormesh(LON, LAT, div_masked,
                       cmap="RdBu", vmin=-vmax, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax, label="Divergencia [1/s]")

    conv = np.where(nav & (div < 0), div, np.nan)
    ax.contourf(LON, LAT, conv, levels=5, cmap="Blues_r", alpha=0.3)

    for k, (p, i, j) in enumerate(waypoints):
        ax.plot(campo.lon[j], campo.lat[i], "r*", markersize=13, zorder=5)
        ax.annotate(f"C{k+1}", (campo.lon[j], campo.lat[i]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="red", fontweight="bold")

    if centinelas:
        for k, (p, i, j) in enumerate(centinelas):
            ax.plot(campo.lon[j], campo.lat[i], "b^", markersize=10, zorder=5)
            ax.annotate(f"S{k+1}", (campo.lon[j], campo.lat[i]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color="blue", fontweight="bold")

    if base is not None:
        _, i, j = base
        ax.plot(campo.lon[j], campo.lat[i], "gs", markersize=11, zorder=5)
        ax.annotate("Base", (campo.lon[j], campo.lat[i]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="green", fontweight="bold")

    leyenda = [
        mpatches.Patch(color="red",   label="Convergencia"),
        mpatches.Patch(color="blue",  label="Centinela offshore"),
        mpatches.Patch(color="green", label="Base (Callao)"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=8)
    ax.set_title(f"Divergencia y zonas de misión — capa {capa}")
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    return ax


def plot_ruta(
    campo: CampoCorrientes,
    ruta: list[Celda],
    waypoints: list[Celda] | None = None,
    centinelas: list[Celda] | None = None,
    base: Celda | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dibuja la ruta completa del AUV coloreada por profundidad (RF-07).

    Los segmentos se colorean según la capa de profundidad del nodo de origen,
    permitiendo ver cuándo el AUV sube o baja durante la misión.

    Args:
        campo: Campo de corrientes.
        ruta: Secuencia completa de nodos (prof, lat, lon).
        waypoints: Zonas de convergencia, para resaltarlas (opcional).
        centinelas: Zonas centinela offshore (opcional).
        base: Celda base de la misión (opcional).
        ax: Ejes opcionales donde dibujar.

    Returns:
        Los ejes con la ruta dibujada.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    # Fondo: campo de corrientes de la capa superficial (referencia).
    LON, LAT = _meshgrid(campo)
    nav = campo.navegable[0]
    uo  = campo.uo[0]
    vo  = campo.vo[0]
    mag = np.where(nav, np.sqrt(uo**2 + vo**2), np.nan)
    ax.set_facecolor(_TIERRA)
    ax.quiver(LON, LAT,
              np.where(nav, uo, np.nan),
              np.where(nav, vo, np.nan),
              mag, cmap="Greys", alpha=0.4,
              scale=5, width=0.003, zorder=1)

    # Ruta coloreada por profundidad, segmento a segmento.
    n_capas = len(campo.prof)
    cmap_prof = plt.get_cmap("plasma", n_capas)
    for u, v in zip(ruta, ruta[1:]):
        p, i, j   = u
        _, ib, jb = v
        color = cmap_prof(p)
        ax.plot(
            [campo.lon[j], campo.lon[jb]],
            [campo.lat[i], campo.lat[ib]],
            color=color, linewidth=1.8, zorder=3,
        )

    # Marcador de inicio (flecha en el primer segmento).
    if len(ruta) > 1:
        p0, i0, j0 = ruta[0]
        p1, i1, j1 = ruta[1]
        ax.annotate("", xy=(campo.lon[j1], campo.lat[i1]),
                    xytext=(campo.lon[j0], campo.lat[i0]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                    zorder=4)

    # Waypoints, centinelas y base.
    if waypoints:
        for k, (p, i, j) in enumerate(waypoints):
            ax.plot(campo.lon[j], campo.lat[i], "r*", markersize=13, zorder=5)
            ax.annotate(f"C{k+1}", (campo.lon[j], campo.lat[i]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color="red", fontweight="bold")
    if centinelas:
        for k, (p, i, j) in enumerate(centinelas):
            ax.plot(campo.lon[j], campo.lat[i], "b^", markersize=10, zorder=5)
            ax.annotate(f"S{k+1}", (campo.lon[j], campo.lat[i]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color="blue", fontweight="bold")
    if base is not None:
        _, i, j = base
        ax.plot(campo.lon[j], campo.lat[i], "gs", markersize=11, zorder=5)
        ax.annotate("Base", (campo.lon[j], campo.lat[i]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color="green", fontweight="bold")

    # Leyenda de profundidades.
    leyenda_prof = [
        mpatches.Patch(color=cmap_prof(p),
                       label=f"Prof. {campo.prof[p]:.2f} m")
        for p in range(n_capas)
    ]
    ax.legend(handles=leyenda_prof, loc="lower right", fontsize=8,
              title="Profundidad")
    ax.set_title("Ruta óptima del AUV (color = profundidad)")
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    return ax


def plot_3d(
    campo: CampoCorrientes,
    ruta: list[Celda],
    waypoints: list[Celda] | None = None,
    centinelas: list[Celda] | None = None,
    base: Celda | None = None,
) -> plt.Figure:
    """Visualización 3D de la ruta con planos de profundidad apilados (RF-07).

    Replica el estilo de Kularatne et al. (ICRA 2018, Fig. 6b): cada capa de
    profundidad es un plano horizontal coloreado por rapidez de corriente,
    apilados con separación visual uniforme. La ruta del AUV es una línea roja
    que se mueve en lat-lon y salta entre planos al cambiar de capa.

    El eje Z usa índices de capa (0, 1, 2, …) en lugar de metros reales para
    que los planos queden bien separados visualmente (las profundidades reales
    van de 0.49 a 3.82 m, un rango demasiado pequeño frente al dominio
    horizontal).

    Args:
        campo: Campo de corrientes.
        ruta: Secuencia completa de nodos (prof, lat, lon).
        waypoints: Zonas de convergencia (opcional).
        centinelas: Zonas centinela offshore (opcional).
        base: Celda base de la misión (opcional).

    Returns:
        La figura con la visualización 3D.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    LON, LAT = _meshgrid(campo)
    n_capas = len(campo.prof)
    cmap_mag = plt.get_cmap("Blues")

    mag_global_max = max(
        float(np.nanmax(np.sqrt(campo.uo[c]**2 + campo.vo[c]**2)))
        for c in range(n_capas)
    )

    # --- Planos apilados (z = índice de capa, no metros reales) ---
    for capa in range(n_capas):
        z_idx = float(capa)
        nav = campo.navegable[capa]
        mag = np.sqrt(campo.uo[capa]**2 + campo.vo[capa]**2)

        rgba = cmap_mag(mag / mag_global_max)
        rgba[~nav] = mcolors.to_rgba(_TIERRA)

        ax.plot_surface(
            LON, LAT, np.full_like(LON, z_idx),
            facecolors=rgba, alpha=0.72, shade=False,
            linewidth=0, antialiased=False,
        )

        # Etiqueta con la profundidad real al borde izquierdo del plano.
        ax.text(
            campo.lon[0], campo.lat[-1], z_idx,
            f"{campo.prof[capa]:.2f} m",
            fontsize=8, color="dimgray", ha="right", va="bottom",
        )

    # --- Ruta del AUV: x=lon, y=lat, z=índice de capa ---
    lons   = [campo.lon[j] for _, _, j in ruta]
    lats   = [campo.lat[i] for _, i, _ in ruta]
    z_ruta = [float(p)     for p, _, _ in ruta]

    ax.plot(lons, lats, z_ruta, color="red", linewidth=2.0, zorder=10)

    # Inicio y fin de la ruta.
    ax.scatter(lons[0],  lats[0],  z_ruta[0],
               color="lime", s=70, zorder=11, depthshade=False, label="Inicio")
    ax.scatter(lons[-1], lats[-1], z_ruta[-1],
               color="lime", marker="x", s=90, linewidths=2,
               zorder=11, depthshade=False, label="Fin")

    # --- Waypoints, centinelas y base sobre sus planos ---
    def _marcar(celdas, color, marker, prefix):
        if not celdas:
            return
        for k, (p, i, j) in enumerate(celdas):
            ax.scatter(campo.lon[j], campo.lat[i], float(p),
                       color=color, marker=marker, s=100,
                       zorder=12, depthshade=False)
            ax.text(campo.lon[j], campo.lat[i], float(p),
                    f" {prefix}{k+1}", fontsize=7, color=color)

    _marcar(waypoints,  "red",   "*", "C")
    _marcar(centinelas, "cyan",  "^", "S")
    if base is not None:
        p, i, j = base
        ax.scatter(campo.lon[j], campo.lat[i], float(p),
                   color="lime", marker="s", s=100, zorder=12, depthshade=False)
        ax.text(campo.lon[j], campo.lat[i], float(p),
                " Base", fontsize=7, color="lime")

    # --- Ejes ---
    ax.set_xlabel("Longitud [°]", labelpad=8)
    ax.set_ylabel("Latitud [°]",  labelpad=8)
    ax.set_zlabel("Capa de profundidad", labelpad=8)
    ax.set_zticks(list(range(n_capas)))
    ax.set_zticklabels([f"{campo.prof[c]:.2f} m" for c in range(n_capas)])
    ax.invert_zaxis()   # capa 0 (superficie) arriba

    ax.set_title("Ruta del AUV por capas de profundidad")

    leyenda = [
        plt.Line2D([0], [0], color="red",  linewidth=2,  label="Ruta AUV"),
        mpatches.Patch(color="red",   label="Convergencia (Cx)"),
        mpatches.Patch(color="cyan",  label="Centinela (Sx)"),
        mpatches.Patch(color="lime",  label="Base / inicio / fin"),
    ]
    ax.legend(handles=leyenda, loc="upper left", fontsize=8)

    return fig


def plot_bateria(
    campo: CampoCorrientes,
    ruta: list[Celda],
    niveles: list[float],
    e_max: float,
    waypoints: list[Celda] | None = None,
    orden: list[int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grafica el estado de batería del AUV a lo largo de la ruta (RF-07).

    Eje X: distancia acumulada recorrida [km].
    Eje Y: nivel de batería [J], con línea de capacidad máxima y zona roja
    de peligro (< 20 % de e_max). Los tramos de regeneración se sombrean
    en verde y los de consumo en naranja.

    Args:
        campo: Campo de corrientes (para calcular distancias reales).
        ruta: Secuencia completa de nodos.
        niveles: Estado de batería en cada nodo (de estado_bateria).
        e_max: Capacidad máxima de la batería [J].
        waypoints: Zonas de la misión para marcarlas sobre el eje X.
        orden: Orden de visita (para identificar cada waypoint en la ruta).
        ax: Ejes opcionales donde dibujar.

    Returns:
        Los ejes con el gráfico de batería.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    lat_rad = math.radians(float(campo.lat.mean()))
    dist_acum = [0.0]
    for (_, i0, j0), (_, i1, j1) in zip(ruta, ruta[1:]):
        dy = (campo.lat[i1] - campo.lat[i0]) * 111_320.0
        dx = (campo.lon[j1] - campo.lon[j0]) * 111_320.0 * math.cos(lat_rad)
        dist_acum.append(dist_acum[-1] + math.hypot(dx, dy))
    dist_km = [d / 1000 for d in dist_acum]

    # Sombrear tramos de regeneración (nivel sube) y consumo (nivel baja).
    for k in range(len(niveles) - 1):
        if niveles[k + 1] > niveles[k]:        # recarga
            ax.axvspan(dist_km[k], dist_km[k + 1], alpha=0.25, color="green")
        elif niveles[k + 1] < niveles[k]:      # consumo
            ax.axvspan(dist_km[k], dist_km[k + 1], alpha=0.12, color="orange")

    # Zona de peligro (< 20 % de e_max).
    umbral = 0.20 * e_max
    ax.axhspan(0, umbral, alpha=0.15, color="red", label="Zona crítica (<20%)")

    # Línea de capacidad máxima.
    ax.axhline(e_max, color="gray", linewidth=1.0, linestyle="--",
               label=f"Capacidad máx. ({e_max:.0f} J)")

    # Nivel de batería.
    ax.plot(dist_km, niveles, color="#1f77b4", linewidth=2.0, label="Batería [J]")
    ax.fill_between(dist_km, niveles, alpha=0.15, color="#1f77b4")

    # Marcar waypoints sobre el eje X.
    if waypoints and orden:
        wp_set = {wp: idx for idx, wp in enumerate(waypoints)}
        visitados = set()
        for paso, nodo in enumerate(ruta):
            if nodo in wp_set and nodo not in visitados:
                visitados.add(nodo)
                ax.axvline(dist_km[paso], color="gray",
                           linewidth=0.8, linestyle=":", alpha=0.7)
                ax.annotate(
                    f"Z{wp_set[nodo]}",
                    (dist_km[paso], niveles[paso]),
                    textcoords="offset points", xytext=(3, 6),
                    fontsize=7, color="dimgray",
                )

    ax.set_xlabel("Distancia recorrida [km]")
    ax.set_ylabel("Nivel de batería [J]")
    ax.set_title("Estado de batería del AUV a lo largo de la misión")
    ax.set_ylim(bottom=0, top=e_max * 1.08)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Leyenda de sombreado.
    from matplotlib.patches import Patch
    extra = [
        Patch(color="green",  alpha=0.4, label="Recarga (regeneración)"),
        Patch(color="orange", alpha=0.3, label="Consumo (propulsión)"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + extra, fontsize=8)

    return ax


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
    print("Cargando y procesando pipeline completo …")
    campo  = cargar_corrientes(str(nc))
    params = ParametrosModelo()

    capa = 0
    lat_media = math.radians(float(campo.lat.mean()))
    dy = abs(float(campo.lat[1] - campo.lat[0])) * 111_320.0
    dx = abs(float(campo.lon[1] - campo.lon[0])) * 111_320.0 * math.cos(lat_media)

    uo  = campo.uo[capa]
    vo  = campo.vo[capa]
    nav = campo.navegable[capa]

    div   = divergencia(uo, vo, dx, dy)
    wps   = seleccionar_waypoints(div, nav, params.k_zonas, capa=capa, dist_min_celdas=3)
    cent  = seleccionar_centinelas(uo, vo, nav, campo.lon, n=2, capa=capa)
    base  = celda_mas_cercana(-12.05, -77.15, campo.lat, campo.lon, nav, capa=capa)
    todos, base_nodo = agregar_puntos_fijos(wps + cent, wps[0], base)
    base_idx = todos.index(base_nodo)

    grafo = construir_grafo(campo, params)
    M, caminos = matriz_costos(grafo, todos)
    orden, costo = atsp_fuerza_bruta(M, base=base_idx)
    ruta  = ensamblar_ruta(orden, caminos)
    print(f"Pipeline OK — ruta de {len(ruta)} nodos, costo {costo:.1f} J")

    # --- Figura principal: zonas + ruta ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plot_zonas(campo, div, wps, centinelas=cent, base=base, capa=capa, ax=axes[0])
    plot_ruta(campo, ruta, waypoints=wps, centinelas=cent, base=base, ax=axes[1])

    plt.tight_layout()
    out1 = pathlib.Path(__file__).parent.parent / "outputs" / "figuras" / "demo_ruta.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Guardado → {out1}")

    # --- Figura 3D ---
    plot_3d(campo, ruta, waypoints=wps, centinelas=cent, base=base)
    out2 = pathlib.Path(__file__).parent.parent / "outputs" / "figuras" / "demo_3d.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Guardado → {out2}")

    # --- Figura de batería ---
    from src.metricas import estado_bateria
    bat = estado_bateria(ruta, grafo, params.e_max)
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    plot_bateria(campo, ruta, bat["niveles"], params.e_max,
                 waypoints=todos, orden=orden, ax=ax4)
    plt.tight_layout()
    out3 = pathlib.Path(__file__).parent.parent / "outputs" / "figuras" / "demo_bateria.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Guardado → {out3}")

    plt.show()
    print("\n✓ visualizacion.py OK")
