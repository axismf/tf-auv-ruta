"""Visualizaciones del dominio: campo, divergencia, zonas, ruta y batería (RF-07)."""
from __future__ import annotations

from .dependencias import *
from .datos import CampoCorrientes
from .grafo import Nodo


__all__ = [
    "plot_campo",
    "plot_divergencia",
    "plot_zonas",
    "plot_ruta",
    "plot_3d",
    "plot_bateria",
    "plot_grafo_costos",
    "plot_tours_atsp",
]

# Una celda/nodo se identifica por su terna (prof, lat, lon).
_TIERRA           = "#d4b483"
_COLORES_PROF     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_COLORES_POR_TIPO = {"Base": "#2ca02c", "C": "#d62728", "S": "#1f77b4"}
_MARCAS_POR_TIPO  = {"Base": "s",       "C": "*",        "S": "^"}
_TAMS_POR_TIPO    = {"Base": 200,        "C": 250,         "S": 180}


def _meshgrid(campo: CampoCorrientes) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(campo.lon, campo.lat)


def plot_campo(
    campo: CampoCorrientes,
    capa: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dibuja el campo de corrientes como quiver sobre el dominio (RF-07).

    Args:
        campo: Campo de corrientes.
        capa:  Índice de profundidad a graficar.
        ax:    Ejes opcionales donde dibujar; si es None se crean nuevos.
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


def plot_divergencia(
    campo: CampoCorrientes,
    div: np.ndarray,
    capa: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Mapa de divergencia horizontal sin waypoints — para la Fase 2 del journey.

    Args:
        campo: Campo de corrientes.
        div:   Campo de divergencia (n_lat, n_lon).
        capa:  Índice de profundidad graficado (solo para el título).
        ax:    Ejes opcionales donde dibujar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    LON, LAT = _meshgrid(campo)
    nav        = campo.navegable[capa]
    div_masked = np.where(nav, div, np.nan)

    valores = div_masked[~np.isnan(div_masked)]
    vmax    = float(np.percentile(np.abs(valores), 95)) if len(valores) else 1e-5

    ax.set_facecolor(_TIERRA)
    im = ax.pcolormesh(
        LON, LAT, div_masked,
        cmap="RdBu", vmin=-vmax, vmax=vmax, shading="auto",
    )
    plt.colorbar(im, ax=ax, label="Divergencia [1/s]")

    conv = np.where(nav & (div < 0), div, np.nan)
    if not np.all(np.isnan(conv)):
        ax.contourf(LON, LAT, conv, levels=5, cmap="Blues_r", alpha=0.25)

    ax.set_title(f"Divergencia horizontal — capa {capa}  ({campo.prof[capa]:.1f} m)")
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    return ax


def plot_zonas(
    campo: CampoCorrientes,
    div: np.ndarray,
    waypoints: list[Nodo],
    centinelas: list[Nodo] | None = None,
    base: Nodo | None = None,
    capa: int = 0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dibuja la divergencia y resalta las zonas seleccionadas (RF-07).

    Args:
        campo:      Campo de corrientes.
        div:        Campo de divergencia de la capa (n_lat, n_lon).
        waypoints:  Celdas de convergencia seleccionadas.
        centinelas: Celdas centinela offshore (opcional).
        base:       Celda base de la misión (opcional).
        capa:       Índice de profundidad a graficar.
        ax:         Ejes opcionales donde dibujar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    LON, LAT = _meshgrid(campo)
    nav        = campo.navegable[capa]
    div_masked = np.where(nav, div, np.nan)
    vmax       = np.nanpercentile(np.abs(div_masked), 95)

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
    ruta: list[Nodo],
    waypoints: list[Nodo] | None = None,
    centinelas: list[Nodo] | None = None,
    base: Nodo | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Dibuja la ruta completa del AUV coloreada por profundidad (RF-07).

    Args:
        campo:      Campo de corrientes.
        ruta:       Secuencia completa de nodos (prof, lat, lon).
        waypoints:  Zonas de convergencia para resaltarlas (opcional).
        centinelas: Zonas centinela offshore (opcional).
        base:       Celda base de la misión (opcional).
        ax:         Ejes opcionales donde dibujar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

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

    n_capas   = len(campo.prof)
    cmap_prof = plt.get_cmap("plasma", n_capas)
    for u, v in zip(ruta, ruta[1:]):
        p, i, j   = u
        _, ib, jb = v
        ax.plot(
            [campo.lon[j], campo.lon[jb]],
            [campo.lat[i], campo.lat[ib]],
            color=cmap_prof(p), linewidth=1.8, zorder=3,
        )

    if len(ruta) > 1:
        p0, i0, j0 = ruta[0]
        p1, i1, j1 = ruta[1]
        ax.annotate("", xy=(campo.lon[j1], campo.lat[i1]),
                    xytext=(campo.lon[j0], campo.lat[i0]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                    zorder=4)

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

    leyenda_prof = [
        mpatches.Patch(color=cmap_prof(p), label=f"Prof. {campo.prof[p]:.2f} m")
        for p in range(n_capas)
    ]
    ax.legend(handles=leyenda_prof, loc="lower right", fontsize=8, title="Profundidad")
    ax.set_title("Ruta óptima del AUV (color = profundidad)")
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    return ax


def plot_3d(
    campo: CampoCorrientes,
    ruta: list[Nodo],
    waypoints: list[Nodo] | None = None,
    centinelas: list[Nodo] | None = None,
    base: Nodo | None = None,
) -> go.Figure:
    """Visualización 3D interactiva de la ruta con planos de profundidad apilados.

    Args:
        campo:      Campo de corrientes.
        ruta:       Secuencia completa de nodos (prof, lat, lon).
        waypoints:  Zonas de convergencia (opcional).
        centinelas: Zonas centinela offshore (opcional).
        base:       Celda base de la misión (opcional).

    Returns:
        Figura Plotly interactiva.
    """
    LON, LAT = _meshgrid(campo)
    n_capas  = len(campo.prof)

    mag_global_max = max(
        float(np.nanmax(np.sqrt(campo.uo[c]**2 + campo.vo[c]**2)))
        for c in range(n_capas)
    )

    traces: list = []

    for capa in range(n_capas):
        mag = np.sqrt(campo.uo[capa]**2 + campo.vo[capa]**2).astype(float)
        mag[~campo.navegable[capa]] = np.nan

        traces.append(go.Surface(
            x=LON, y=LAT,
            z=np.full_like(LON, float(capa)),
            surfacecolor=mag,
            colorscale="Blues",
            cmin=0, cmax=mag_global_max,
            showscale=(capa == 0),
            colorbar=dict(title="Velocidad<br>(m/s)", thickness=14, x=1.02)
                if capa == 0 else {},
            opacity=0.78,
            name=f"{campo.prof[capa]:.2f} m",
            hovertemplate=(
                f"<b>Capa {campo.prof[capa]:.2f} m</b><br>"
                "Lon: %{x:.4f}°<br>Lat: %{y:.4f}°<br>"
                "Vel: %{surfacecolor:.3f} m/s<extra></extra>"
            ),
        ))

    lons   = [campo.lon[j] for _, _, j in ruta]
    lats   = [campo.lat[i] for _, i, _ in ruta]
    z_ruta = [float(p)     for p, _, _ in ruta]

    traces.append(go.Scatter3d(
        x=lons, y=lats, z=z_ruta,
        mode="lines",
        line=dict(color="red", width=5),
        name="Ruta AUV",
        hovertemplate="Lon: %{x:.4f}°<br>Lat: %{y:.4f}°<extra>Ruta AUV</extra>",
    ))
    traces.append(go.Scatter3d(
        x=[lons[0], lons[-1]], y=[lats[0], lats[-1]],
        z=[z_ruta[0], z_ruta[-1]],
        mode="markers+text",
        marker=dict(color="lime", size=7, symbol="circle"),
        text=["Inicio", "Fin"], textposition="top center",
        name="Inicio / Fin",
        hovertemplate="%{text}<extra></extra>",
    ))

    if waypoints:
        traces.append(go.Scatter3d(
            x=[campo.lon[j] for _, _, j in waypoints],
            y=[campo.lat[i] for _, i, _ in waypoints],
            z=[float(p)     for p, _, _ in waypoints],
            mode="markers+text",
            marker=dict(color="red", size=8, symbol="diamond"),
            text=[f"C{k+1}" for k in range(len(waypoints))],
            textposition="top center",
            name="Convergencia",
        ))

    if centinelas:
        traces.append(go.Scatter3d(
            x=[campo.lon[j] for _, _, j in centinelas],
            y=[campo.lat[i] for _, i, _ in centinelas],
            z=[float(p)     for p, _, _ in centinelas],
            mode="markers+text",
            marker=dict(color="cyan", size=8, symbol="diamond"),
            text=[f"S{k+1}" for k in range(len(centinelas))],
            textposition="top center",
            name="Centinela",
        ))

    if base is not None:
        p, i, j = base
        traces.append(go.Scatter3d(
            x=[campo.lon[j]], y=[campo.lat[i]], z=[float(p)],
            mode="markers+text",
            marker=dict(color="lime", size=10, symbol="square"),
            text=["Base"], textposition="top center",
            name="Base",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="Longitud [°]",
            yaxis_title="Latitud [°]",
            zaxis=dict(
                title="Profundidad",
                autorange="reversed",
                tickvals=list(range(n_capas)),
                ticktext=[f"{campo.prof[c]:.2f} m" for c in range(n_capas)],
            ),
            aspectmode="manual",
            aspectratio=dict(x=2.0, y=2.0, z=0.5),
        ),
        title="Ruta del AUV — capas de profundidad",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=40),
        height=620,
    )
    return fig


def plot_bateria(
    campo: CampoCorrientes,
    ruta: list[Nodo],
    niveles: list[float],
    e_max: float,
    waypoints: list[Nodo] | None = None,
    orden: list[int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grafica el estado de batería del AUV a lo largo de la ruta (RF-07).

    Args:
        campo:     Campo de corrientes (para calcular distancias reales).
        ruta:      Secuencia completa de nodos.
        niveles:   Estado de batería en cada nodo (de estado_bateria).
        e_max:     Capacidad máxima de la batería [J].
        waypoints: Zonas de la misión para marcarlas sobre el eje X.
        orden:     Orden de visita (para identificar cada waypoint en la ruta).
        ax:        Ejes opcionales donde dibujar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    lat_rad = math.radians(float(campo.lat.mean()))
    dist_acum = [0.0]
    for (_, i0, j0), (_, i1, j1) in zip(ruta, ruta[1:]):
        dy = (campo.lat[i1] - campo.lat[i0]) * GRADOS_A_METROS
        dx = (campo.lon[j1] - campo.lon[j0]) * GRADOS_A_METROS * math.cos(lat_rad)
        dist_acum.append(dist_acum[-1] + math.hypot(dx, dy))
    dist_km = [d / 1000 for d in dist_acum]

    for k in range(len(niveles) - 1):
        if niveles[k + 1] > niveles[k]:
            ax.axvspan(dist_km[k], dist_km[k + 1], alpha=0.25, color="green")
        elif niveles[k + 1] < niveles[k]:
            ax.axvspan(dist_km[k], dist_km[k + 1], alpha=0.12, color="orange")

    umbral = 0.20 * e_max
    ax.axhspan(0, umbral, alpha=0.15, color="red", label="Zona crítica (<20%)")
    ax.axhline(e_max, color="gray", linewidth=1.0, linestyle="--",
               label=f"Capacidad máx. ({e_max:.0f} J)")
    ax.plot(dist_km, niveles, color="#1f77b4", linewidth=2.0, label="Batería [J]")
    ax.fill_between(dist_km, niveles, alpha=0.15, color="#1f77b4")

    if waypoints and orden:
        wp_set    = {wp: idx for idx, wp in enumerate(waypoints)}
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
    ax.grid(True, alpha=0.3)

    base_legend = ax.get_legend_handles_labels()[0]
    extra = [
        mpatches.Patch(color="green",  alpha=0.4, label="Recarga (regeneración)"),
        mpatches.Patch(color="orange", alpha=0.3, label="Consumo (propulsión)"),
    ]
    ax.legend(handles=base_legend + extra, fontsize=8)
    return ax


def plot_grafo_costos(
    todos: list[Nodo],
    campo: CampoCorrientes,
    M: np.ndarray,
    etiquetas: dict,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grafo dirigido de N zonas con costos BF como pesos (RF-07).

    Args:
        todos:      Lista de celdas de misión (waypoints + centinelas + base).
        campo:      Campo de corrientes (para posición geográfica de los nodos).
        M:          Matriz de costos energéticos entre zonas.
        etiquetas:  Dict celda → etiqueta textual (p.e. "C1", "S2", "Base").
        ax:         Ejes opcionales donde dibujar.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    lons_nodos = [campo.lon[n[2]] for n in todos]
    lats_nodos = [campo.lat[n[1]] for n in todos]
    pos        = {n: (campo.lon[n[2]], campo.lat[n[1]]) for n in todos}

    dl = max((max(lons_nodos) - min(lons_nodos)) * 0.45, 0.05)
    db = max((max(lats_nodos) - min(lats_nodos)) * 0.45, 0.05)
    ax.set_xlim(min(lons_nodos) - dl, max(lons_nodos) + dl)
    ax.set_ylim(min(lats_nodos) - db, max(lats_nodos) + db)
    ax.set_facecolor(_TIERRA)

    mask = np.isfinite(M).copy()
    np.fill_diagonal(mask, False)
    costos_validos = M[mask]
    if len(costos_validos) == 0:
        vmin, vmax_c = 0.0, 1.0
    else:
        vmin   = float(costos_validos.min())
        vmax_c = float(costos_validos.max())
        if vmin == vmax_c:
            vmax_c = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax_c)
    cmap = plt.get_cmap("RdYlGn_r")

    for i, u in enumerate(todos):
        for j, v in enumerate(todos):
            if i == j:
                continue
            costo = float(M[i, j])
            if not math.isfinite(costo):
                continue
            xi, yi = pos[u]
            xj, yj = pos[v]
            ax.annotate(
                "",
                xy=(xj, yj), xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=cmap(norm(costo)),
                    lw=1.3,
                    connectionstyle="arc3,rad=0.2",
                    mutation_scale=11,
                ),
                zorder=2,
            )

    for nodo in todos:
        etiq  = etiquetas.get(nodo, "?")
        tipo  = "Base" if etiq == "Base" else etiq[0]
        color = _COLORES_POR_TIPO.get(tipo, "gray")
        x, y  = pos[nodo]
        ax.scatter(x, y, c=color,
                   marker=_MARCAS_POR_TIPO.get(tipo, "o"),
                   s=_TAMS_POR_TIPO.get(tipo, 180),
                   zorder=4, edgecolors="white", linewidths=0.8)
        ax.annotate(
            etiq, (x, y),
            textcoords="offset points", xytext=(7, 5),
            fontsize=9, fontweight="bold", color="white", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.8),
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Costo energético [J]", shrink=0.7, pad=0.02)

    ax.set_title("Grafo de costos entre zonas de misión\n(resultado de Bellman-Ford multi-fuente)")
    ax.set_xlabel("Longitud [°]")
    ax.set_ylabel("Latitud [°]")
    return ax


def plot_tours_atsp(
    M: np.ndarray,
    etiquetas: list[str],
    orden_optimo: list[int],
) -> plt.Figure:
    """Búsqueda exhaustiva ATSP: distribución de todos los tours + ciclo óptimo (RF-07).

    Args:
        M:             Matriz de costos entre zonas.
        etiquetas:     Nombre de cada zona en orden de índice.
        orden_optimo:  Tour ganador (incluye la base al inicio y al final).

    Returns:
        Figura matplotlib con dos paneles: barras de todos los tours + ciclo óptimo.
    """
    base        = orden_optimo[0]
    intermedios = [i for i in range(len(etiquetas)) if i != base]

    tours_costos: list[float] = []
    tours_seqs:   list[list[int]] = []
    for perm in itertools.permutations(intermedios):
        seq = [base] + list(perm) + [base]
        c   = sum(float(M[seq[k], seq[k + 1]]) for k in range(len(seq) - 1))
        tours_costos.append(c)
        tours_seqs.append(seq)

    orden_sort  = sorted(range(len(tours_costos)), key=lambda k: tours_costos[k])
    costos_sort = [tours_costos[k] for k in orden_sort]
    idx_opt     = next(
        k for k, orig_k in enumerate(orden_sort)
        if tours_seqs[orig_k] == orden_optimo
    )

    n_tours = len(costos_sort)
    fig_h   = max(4.0, min(n_tours * 0.06 + 2.5, 8.0))
    fig, (ax_bar, ax_ciclo) = plt.subplots(1, 2, figsize=(13, fig_h))

    colores          = ["#5899da"] * n_tours
    colores[idx_opt] = "#2ca02c"
    ax_bar.bar(range(n_tours), costos_sort, color=colores, width=1.0, zorder=2)

    costo_opt = costos_sort[idx_opt]
    rango     = max(costos_sort) - min(costos_sort) if len(costos_sort) > 1 else 1.0
    offset_x  = max(1, n_tours * 0.08)
    ax_bar.annotate(
        f"Óptimo: {costo_opt:.1f} J",
        xy=(idx_opt, costo_opt),
        xytext=(min(idx_opt + offset_x, n_tours - 1), costo_opt + rango * 0.1),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
        fontsize=8, color="#2ca02c", fontweight="bold",
    )
    ax_bar.set_xticks([])
    ax_bar.set_xlabel(f"{n_tours} tours evaluados (ordenados de menor a mayor costo)")
    ax_bar.set_ylabel("Costo total del tour [J]")
    ax_bar.set_title("Fuerza bruta ATSP\ntodos los órdenes de visita posibles")
    ax_bar.grid(axis="y", alpha=0.3, zorder=1)

    visitados = orden_optimo[:-1]
    n         = len(visitados)
    angulos   = {
        idx: math.pi / 2 - 2 * math.pi * k / n
        for k, idx in enumerate(visitados)
    }
    pos_c = {idx: (math.cos(a), math.sin(a)) for idx, a in angulos.items()}

    for k in range(len(orden_optimo) - 1):
        u, v   = orden_optimo[k], orden_optimo[k + 1]
        xu, yu = pos_c[u]
        xv, yv = pos_c[v]
        ax_ciclo.annotate(
            "",
            xy=(xv, yv), xytext=(xu, yu),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#555555",
                lw=1.8,
                connectionstyle="arc3,rad=0.05",
                mutation_scale=14,
                shrinkA=12, shrinkB=12,
            ),
            zorder=2,
        )
        mx, my = (xu + xv) / 2 * 1.18, (yu + yv) / 2 * 1.18
        etiq_cost = (f"{M[u, v]:.0f} J" if math.isfinite(float(M[u, v])) else "∞")
        ax_ciclo.text(
            mx, my, etiq_cost,
            fontsize=7, ha="center", va="center", color="#444444",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        )

    for idx in visitados:
        etiq  = etiquetas[idx]
        tipo  = "Base" if etiq == "Base" else etiq[0]
        color = _COLORES_POR_TIPO.get(tipo, "gray")
        mk    = "s" if tipo == "Base" else ("*" if tipo == "C" else "^")
        x, y  = pos_c[idx]
        ax_ciclo.scatter(x, y, c=color, s=300, marker=mk, zorder=4,
                         edgecolors="white", linewidths=1.0)
        ax_ciclo.text(x, y - 0.22, etiq,
                      ha="center", va="top",
                      fontsize=9, fontweight="bold", color=color)

    ax_ciclo.set_title(f"Tour óptimo — {costo_opt:.1f} J")
    ax_ciclo.set_xlim(-1.65, 1.65)
    ax_ciclo.set_ylim(-1.65, 1.65)
    ax_ciclo.set_aspect("equal")
    ax_ciclo.axis("off")

    leyenda = [
        mpatches.Patch(color="#2ca02c", label="Base"),
        mpatches.Patch(color="#d62728", label="Convergencia (C)"),
        mpatches.Patch(color="#1f77b4", label="Centinela (S)"),
    ]
    ax_ciclo.legend(handles=leyenda, loc="lower center", fontsize=8,
                    bbox_to_anchor=(0.5, -0.08), ncol=3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Ejecutar con: python -m src.visualizacion
    from src.datos import cargar_corrientes
    from src.config import ParametrosModelo
    from src.grafo import construir_grafo
    from src.zonas import (
        divergencia, seleccionar_waypoints,
        seleccionar_centinelas, celda_mas_cercana,
        agregar_puntos_fijos,
    )
    from src.algoritmos import matriz_costos, atsp_fuerza_bruta, ensamblar_ruta
    from src.metricas import estado_bateria

    nc     = pathlib.Path(__file__).parent.parent / "data" / "lima3.nc"
    campo  = cargar_corrientes(str(nc))
    params = ParametrosModelo()
    capa   = 0

    lat_media = math.radians(float(campo.lat.mean()))
    dy = abs(float(campo.lat[1] - campo.lat[0])) * GRADOS_A_METROS
    dx = abs(float(campo.lon[1] - campo.lon[0])) * GRADOS_A_METROS * math.cos(lat_media)

    uo  = campo.uo[capa]
    vo  = campo.vo[capa]
    nav = campo.navegable[capa]

    div          = divergencia(uo, vo, dx, dy)
    wps          = seleccionar_waypoints(div, nav, params.k_zonas, capa=capa)
    cent         = seleccionar_centinelas(uo, vo, nav, campo.lon, n=2, capa=capa)
    base         = celda_mas_cercana(-12.05, -77.15, campo.lat, campo.lon, nav, capa=capa)
    todos, base_nodo = agregar_puntos_fijos(wps + cent, wps[0], base)
    base_idx     = todos.index(base_nodo)

    grafo        = construir_grafo(campo, params)
    M, caminos   = matriz_costos(grafo, todos)
    orden, costo = atsp_fuerza_bruta(M, base=base_idx)
    ruta         = ensamblar_ruta(orden, caminos)
    bat          = estado_bateria(ruta, grafo, params.e_max)

    out_dir = pathlib.Path(__file__).parent.parent / "outputs" / "figuras"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_zonas(campo, div, wps, centinelas=cent, base=base, capa=capa, ax=axes[0])
    plot_ruta(campo, ruta, waypoints=wps, centinelas=cent, base=base, ax=axes[1])
    plt.tight_layout()
    plt.savefig(out_dir / "demo_ruta.png", dpi=150, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    plot_bateria(campo, ruta, bat.niveles, params.e_max,
                 waypoints=todos, orden=orden, ax=ax2)
    plt.tight_layout()
    plt.savefig(out_dir / "demo_bateria.png", dpi=150, bbox_inches="tight")

    print(f"Figuras guardadas en {out_dir}")
    print("\n✓ visualizacion.py OK")
