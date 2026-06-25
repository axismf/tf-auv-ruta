"""Interfaz web del planificador de rutas (RF-06, RF-07, RF-08, RF-09).

Capa de presentación construida con Streamlit. Solo orquesta los módulos del
núcleo (no contiene lógica de algoritmos), de modo que la UI pueda cambiarse
sin tocar el core.

Ejecutar con:
    streamlit run app.py
"""
from __future__ import annotations

import base64
import math
import io
import pathlib
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr

from src.config import ParametrosModelo
from src.datos import cargar_corrientes, _ALIAS
from src.grafo import construir_grafo
from src.zonas import (
    divergencia,
    seleccionar_waypoints,
    seleccionar_centinelas,
    celda_mas_cercana,
    agregar_puntos_fijos,
)
from src.algoritmos import (
    matriz_costos,
    atsp_fuerza_bruta,
    ensamblar_ruta,
    bellman_ford,
    hay_ciclo_negativo,
)
from src.metricas import resumen_mision, estado_bateria, exportar_csv
from src.visualizacion import (
    plot_campo, plot_divergencia, plot_zonas, plot_ruta, plot_3d, plot_bateria,
    plot_grafo_costos, plot_tours_atsp,
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_LAT_CALLAO = -12.05
_LON_CALLAO = -77.15
_DATA_DIR    = pathlib.Path(__file__).parent / "data"

_FASES_NOMBRES = [
    "Datos",
    "Corrientes",
    "Zonas",
    "Grafo",
    "ATSP",
    "Misión",
]

_FASES_DESCRIPCIONES = {
    6: ("Resultados de la misión + Exportar", "Completa Fase 5 primero"),
}

_SVG_AUV = (
    '<svg viewBox="0 0 220 80" xmlns="http://www.w3.org/2000/svg">'
    # cuerpo principal
    '<ellipse cx="108" cy="40" rx="88" ry="20" fill="#2E8B9E"/>'
    # nariz
    '<path d="M196,40 Q220,28 217,40 Q220,52 196,40" fill="#1a6b7e"/>'
    # cola
    '<ellipse cx="22" cy="40" rx="12" ry="20" fill="#1a6b7e"/>'
    # aletas de cola
    '<path d="M26,34 L6,18 L18,34" fill="#1a6b7e"/>'
    '<path d="M26,46 L6,62 L18,46" fill="#1a6b7e"/>'
    # aleta dorsal
    '<path d="M80,21 L90,5 L100,21" fill="#1a6b7e"/>'
    # ojo/visor
    '<circle cx="155" cy="34" r="7" fill="#7ec8e3" stroke="#1a6b7e" stroke-width="1.5"/>'
    # propulsor
    '<ellipse cx="10" cy="40" rx="3" ry="12" fill="none" stroke="#d0d0d0" stroke-width="1.8"/>'
    '<line x1="10" y1="28" x2="10" y2="52" stroke="#d0d0d0" stroke-width="1.8"/>'
    '<line x1="2" y1="40" x2="18" y2="40" stroke="#d0d0d0" stroke-width="1.5"/>'
    # sensor
    '<rect x="95" y="16" width="3" height="8" fill="#b0b0b0" rx="1"/>'
    '</svg>'
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults: dict = {
        "fase_actual":   1,
        "campo":         None,
        "nc_path":       None,
        "capa_preview":  0,
        # Fase 2
        "div":           None,
        "dx":            None,
        "dy":            None,
        # Fase 3 — parámetros
        "k_zonas_f3":           6,
        "k_cent_f3":            2,
        "dist_min_f3":          3,
        "bases_personalizadas": [],   # [{nombre, lat, lon}, ...]
        "base_key":             "callao",
        # Fase 3 — resultados
        "wps":       None,
        "cent":      None,
        "base_celda": None,
        "todos":     None,
        "base_nodo": None,
        "base_idx":  None,
        # Fase 4 — drones
        "drones": [
            {
                "nombre": "AUV Estándar",
                "s":      0.5,
                "eta":    0.30,
                "k_p":    1.0,
                "k_r":    1.0,
                "e_max":  1_000_000,
                "pct_ini": 100,
            }
        ],
        "drone_key":      0,
        "abrir_drone_idx": None,   # None=cerrado, -1=nuevo, >=0=editar
        # Fase 4 — resultados
        "grafo":         None,
        "params_f4":     None,
        "bat_ini_j":     None,
        "hay_ciclo_f4":  None,
        # Fase 5 — resultados
        "ruta":       None,
        "orden_f5":   None,
        "costo_f5":   None,
        "M_f5":       None,
        "caminos_f5": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fig_a_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _mostrar_figura(fig: plt.Figure, modo: str = "medio") -> None:
    ratios = {
        "compacto":   (1, 2, 1),
        "medio":      (1, 3, 1),
        "panoramico": (1, 8, 1),
    }.get(modo, (1, 3, 1))
    _izq, centro, _der = st.columns(ratios)
    with centro:
        st.pyplot(fig)


@st.cache_data(show_spinner=False)
def _calcular_div(nc_path: str) -> tuple:
    """Calcula divergencia superficial; cacheado por ruta de archivo."""
    campo   = cargar_corrientes(nc_path)
    capa    = 0
    lat_med = math.radians(float(campo.lat.mean()))
    dy      = abs(float(campo.lat[1] - campo.lat[0])) * 111_320.0
    dx      = abs(float(campo.lon[1] - campo.lon[0])) * 111_320.0 * math.cos(lat_med)
    div     = divergencia(campo.uo[capa], campo.vo[capa], dx, dy)
    return div, dx, dy


@st.cache_data(show_spinner=False)
def _cargar_campo(ruta: str):
    return cargar_corrientes(ruta)


@st.cache_data(show_spinner=False)
def _leer_metadata_nc(ruta: str) -> dict | None:
    """Lee solo coordenadas del .nc para mostrar en la tarjeta."""
    try:
        with xr.open_dataset(ruta) as ds:
            disponibles = set(ds.variables) | set(ds.coords)

            def _alias(clave: str) -> str | None:
                for a in _ALIAS[clave]:
                    if a in disponibles:
                        return a
                return None

            n_lat  = _alias("lat")
            n_lon  = _alias("lon")
            n_prof = _alias("prof")
            n_uo   = _alias("uo")
            n_vo   = _alias("vo")
            n_time = _alias("time")

            if not all([n_lat, n_lon, n_prof, n_uo, n_vo]):
                return None

            lat  = ds[n_lat].values.astype(float)
            lon  = ds[n_lon].values.astype(float)
            prof = ds[n_prof].values.astype(float)

            paso_lat_km = abs(float(lat[1] - lat[0])) * 111.32 if len(lat) > 1 else 0.0
            paso_lon_km = (
                abs(float(lon[1] - lon[0])) * 111.32
                * math.cos(math.radians(float(lat.mean())))
                if len(lon) > 1 else 0.0
            )
            paso_km = (paso_lat_km + paso_lon_km) / 2

            fecha_str = ""
            if n_time and n_time in ds:
                import pandas as pd
                try:
                    fecha_str = str(pd.Timestamp(ds[n_time].values[0]).date())
                except Exception:
                    fecha_str = str(ds[n_time].values[0])[:10]

            return {
                "lat_min": float(lat.min()),
                "lat_max": float(lat.max()),
                "lon_min": float(lon.min()),
                "lon_max": float(lon.max()),
                "n_lat":   len(lat),
                "n_lon":   len(lon),
                "n_prof":  len(prof),
                "paso_km": paso_km,
                "fecha":   fecha_str,
                "vars":    [v for v in [n_uo, n_vo] if v],
            }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
_ESTILOS = """
<style>
.stButton > button,
.stDownloadButton > button {
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.2px;
    padding: 0.55rem 1.1rem;
    transition: filter 0.15s ease-in-out, transform 0.15s ease-in-out;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    filter: brightness(0.96);
    transform: translateY(-1px);
}

.auv-hero {
    border-left: 4px solid #2E8B9E;
    padding: 0.2rem 0 0.2rem 1rem;
    margin-bottom: 0.6rem;
}
.auv-hero h1 {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
}
.auv-hero .sub {
    font-size: 0.95rem;
    margin-top: 0.35rem;
    max-width: 72ch;
    opacity: 0.72;
}

.capa-label {
    text-align: center;
    padding: 0.35rem 0.5rem;
    background: rgba(128,128,128,0.08);
    border-radius: 6px;
    font-size: 0.9rem;
}
</style>
"""


# ---------------------------------------------------------------------------
# Barra de progreso de fases
# ---------------------------------------------------------------------------
def _render_progreso() -> None:
    fase = st.session_state.fase_actual
    cols = st.columns(len(_FASES_NOMBRES))
    for i, (col, nombre) in enumerate(zip(cols, _FASES_NOMBRES), start=1):
        with col:
            if i < fase:
                st.markdown(f"**✅ {i}·{nombre}**")
            elif i == fase:
                st.markdown(f"**🔵 {i}·{nombre}**")
            else:
                st.markdown(f"🔒 {i}·{nombre}")
    st.divider()


# ---------------------------------------------------------------------------
# FASE 1 — Fuente de datos
# ---------------------------------------------------------------------------
def _render_fase1() -> None:
    fase = st.session_state.fase_actual

    c_titulo, c_badge = st.columns([7, 1])
    with c_titulo:
        st.markdown("### FASE 1 · Fuente de datos — campo de corrientes marinas")
    with c_badge:
        if fase > 1:
            st.markdown("**✅ Completada**")
        else:
            st.markdown("**🔵 Activa**")

    # Si ya está completada, mostrar solo resumen colapsado
    if fase > 1:
        nc_nombre = pathlib.Path(st.session_state.nc_path).name
        with st.expander(f"Dataset cargado: `{nc_nombre}` — ver detalles", expanded=False):
            _render_contenido_fase1()
        st.divider()
        return

    _render_contenido_fase1()
    st.divider()


def _render_contenido_fase1() -> None:
    # --- Expander de contexto (colapsado por defecto) ---
    with st.expander(
        "ⓘ  ¿Qué es Copernicus Marine Service y el formato NetCDF?",
        expanded=False,
    ):
        st.markdown("""
**Copernicus Marine Service (CMEMS)** es el servicio oceanográfico de la Unión Europea.
Provee campos de corrientes marinas en tiempo real y pronóstico, en grillas tridimensionales globales.

El archivo **NetCDF (.nc)** que utiliza este planificador contiene:

| Variable | Significado | Unidad |
|---|---|---|
| `uo` | Velocidad zonal — componente Este ↔ Oeste | m/s |
| `vo` | Velocidad meridional — componente Norte ↕ Sur | m/s |
| `lat` / `lon` | Coordenadas del centro de cada celda | grados |
| `depth` | Niveles de profundidad disponibles (0 m → 5 500 m) | m |

Producto compatible: `GLOBAL_ANALYSISFORECAST_PHY_001_024`
        """)

    st.markdown("**Seleccioná un dataset de corrientes para la misión:**")

    # --- Descubrir .nc en data/ ---
    archivos_nc = sorted(_DATA_DIR.glob("*.nc"))
    n_datasets  = len(archivos_nc)

    # Hasta 3 tarjetas de datasets + 1 de upload
    n_cols = min(n_datasets, 3) + 1
    cols   = st.columns(n_cols)

    for idx, nc_path in enumerate(archivos_nc[:3]):
        meta       = _leer_metadata_nc(str(nc_path))
        seleccion  = st.session_state.nc_path == str(nc_path)
        nombre_leg = nc_path.stem.replace("_", " ").title()

        with cols[idx]:
            with st.container(border=True):
                st.markdown(f"**📍 {nombre_leg}**")

                if meta:
                    st.markdown(
                        f"Lat `{meta['lat_min']:.1f}°` → `{meta['lat_max']:.1f}°`  \n"
                        f"Lon `{meta['lon_min']:.1f}°` → `{meta['lon_max']:.1f}°`  \n"
                        f"Grilla **{meta['n_lat']} × {meta['n_lon']}** celdas  \n"
                        f"Paso ≈ **{meta['paso_km']:.1f} km**/celda  \n"
                        f"Capas **{meta['n_prof']}** profundidades  \n"
                        f"Variables `{'  ·  '.join(meta['vars'])}`"
                        + (f"  \nFecha **{meta['fecha']}**" if meta["fecha"] else "")
                    )
                else:
                    st.caption("No se pudo leer la metadata.")

                btn_label = "✅  Seleccionado" if seleccion else "Seleccionar"
                btn_type  = "primary" if seleccion else "secondary"
                if st.button(
                    btn_label,
                    key=f"sel_{nc_path.stem}",
                    type=btn_type,
                    use_container_width=True,
                ):
                    st.session_state.nc_path      = str(nc_path)
                    st.session_state.campo        = None
                    st.session_state.capa_preview = 0
                    st.rerun()

    # Tarjeta de upload
    with cols[-1]:
        with st.container(border=True):
            st.markdown("**📂 Subir mi propio .nc**")
            st.caption(
                "Producto CMEMS compatible:  \n"
                "`GLOBAL_ANALYSISFORECAST_PHY_001_024`"
            )
            archivo_up = st.file_uploader(
                "Seleccionar",
                type=["nc"],
                label_visibility="collapsed",
                key="uploader_fase1",
            )
            if archivo_up is not None:
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                    tmp.write(archivo_up.read())
                    ruta_tmp = tmp.name
                st.session_state.nc_path      = ruta_tmp
                st.session_state.campo        = None
                st.session_state.capa_preview = 0
                st.rerun()

    # --- Resumen + preview si hay dataset seleccionado ---
    if st.session_state.nc_path:
        _render_dataset_cargado()


def _render_dataset_cargado() -> None:
    nc_path = st.session_state.nc_path

    with st.spinner("Cargando campo de corrientes…"):
        try:
            if st.session_state.campo is None:
                st.session_state.campo = _cargar_campo(nc_path)
            campo = st.session_state.campo
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return

    st.markdown("---")
    st.markdown(f"**✅  Dataset cargado:** `{pathlib.Path(nc_path).name}`")

    # --- Métricas de resumen ---
    n_prof, n_lat, n_lon = campo.uo.shape
    total   = campo.navegable.size
    agua    = int(campo.navegable.sum())
    lat_med = math.radians(float(campo.lat.mean()))
    paso_lat_km = abs(float(campo.lat[1] - campo.lat[0])) * 111.32 if n_lat > 1 else 0.0
    paso_lon_km = (
        abs(float(campo.lon[1] - campo.lon[0])) * 111.32 * math.cos(lat_med)
        if n_lon > 1 else 0.0
    )
    paso_km = (paso_lat_km + paso_lon_km) / 2

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Grilla",             f"{n_lat} × {n_lon}")
    col2.metric("Paso aprox.",        f"≈ {paso_km:.1f} km")
    col3.metric("Capas",              str(n_prof))
    col4.metric("Celdas navegables",  f"{agua}/{total}  ({100*agua/total:.0f} %)")

    c_cob, c_vel = st.columns(2)
    with c_cob:
        st.markdown(
            f"**Cobertura** —  "
            f"Lat `{campo.lat.min():.2f}°` → `{campo.lat.max():.2f}°`  ·  "
            f"Lon `{campo.lon.min():.2f}°` → `{campo.lon.max():.2f}°`"
        )
    with c_vel:
        mag_sup = np.sqrt(campo.uo[0] ** 2 + campo.vo[0] ** 2)
        mag_nav = mag_sup[campo.navegable[0]]
        if len(mag_nav):
            st.markdown(
                f"**Corriente superficial** —  "
                f"media `{float(np.nanmean(mag_nav)):.3f} m/s`  ·  "
                f"máx `{float(np.nanmax(mag_nav)):.3f} m/s`"
            )

    # --- Vista previa con navbar de capas ---
    st.markdown("**Vista previa del campo de corrientes**")

    capa_actual = st.session_state.capa_preview
    n_capas     = len(campo.prof)

    # Navbar: ◀  |  etiqueta central  |  ▶  |  dropdown
    nav1, nav2, nav3, nav4 = st.columns([1, 4, 1, 3])

    with nav1:
        if st.button("◀", key="capa_prev", disabled=(capa_actual == 0)):
            st.session_state.capa_preview = capa_actual - 1
            st.rerun()

    with nav2:
        prof_m     = campo.prof[capa_actual]
        nombre_cap = "superficie" if capa_actual == 0 else f"{prof_m:.1f} m"
        st.markdown(
            f"<div class='capa-label'>"
            f"<b>Capa {capa_actual}</b> · {nombre_cap}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with nav3:
        if st.button("▶", key="capa_next", disabled=(capa_actual == n_capas - 1)):
            st.session_state.capa_preview = capa_actual + 1
            st.rerun()

    with nav4:
        opciones    = [f"Capa {i}  ·  {campo.prof[i]:.1f} m" for i in range(n_capas)]
        nueva_capa  = st.selectbox(
            "Saltar a capa",
            options=opciones,
            index=capa_actual,
            key="sel_capa_drop",
            label_visibility="collapsed",
        )
        idx_nueva = opciones.index(nueva_capa)
        if idx_nueva != capa_actual:
            st.session_state.capa_preview = idx_nueva
            st.rerun()

    # Figura del campo
    with st.spinner(f"Renderizando capa {capa_actual}…"):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_campo(campo, capa=capa_actual, ax=ax)
        _mostrar_figura(fig, "medio")
        plt.close(fig)

    # --- Botón continuar ---
    st.markdown("")
    if st.session_state.fase_actual == 1:
        if st.button(
            "▶  Continuar → Fase 2: Análisis del campo de corrientes",
            type="primary",
            key="btn_continuar_f1",
        ):
            st.session_state.fase_actual = 2
            st.rerun()


# ---------------------------------------------------------------------------
# FASE 2 — Análisis del campo de corrientes
# ---------------------------------------------------------------------------
def _render_fase2() -> None:
    fase = st.session_state.fase_actual

    c_titulo, c_badge = st.columns([7, 1])
    with c_titulo:
        st.markdown("### FASE 2 · Análisis del campo de corrientes")
    with c_badge:
        if fase > 2:
            st.markdown("**✅ Completada**")
        elif fase == 2:
            st.markdown("**🔵 Activa**")
        else:
            st.caption("Completa Fase 1 primero")

    if fase < 2:
        st.divider()
        return

    if fase > 2:
        with st.expander("Ver análisis del campo de corrientes", expanded=False):
            _render_contenido_fase2()
        st.divider()
        return

    _render_contenido_fase2()
    st.divider()


def _render_contenido_fase2() -> None:
    campo   = st.session_state.campo
    nc_path = st.session_state.nc_path
    capa    = 0

    # Calcular dx/dy para el bloque "¿Qué entra?"
    lat_med = math.radians(float(campo.lat.mean()))
    dy_km   = abs(float(campo.lat[1] - campo.lat[0])) * 111.32
    dx_km   = abs(float(campo.lon[1] - campo.lon[0])) * 111.32 * math.cos(lat_med)

    # ── ¿Qué entra? ────────────────────────────────────────────────────────
    st.markdown("**¿Qué entra en esta fase?**")
    st.info(
        f"▸ Campo de corrientes de Fase 1 — `{pathlib.Path(nc_path).name}`  \n"
        f"▸ Componentes superficiales: `uo` (zonal Este↔Oeste) y `vo` (meridional Norte↕Sur) — capa 0  \n"
        f"▸ Resolución espacial: dx ≈ **{dx_km:.1f} km/celda**  ·  dy ≈ **{dy_km:.1f} km/celda**"
    )

    # ── ¿Qué se calcula? ───────────────────────────────────────────────────
    st.markdown("**¿Qué se calcula?**")
    with st.container(border=True):
        st.markdown(
            "La **divergencia horizontal** del campo de corrientes indica si el flujo "
            "se acumula o se dispersa en cada celda:\n\n"
            "- `div < 0`  →  **convergencia**: el flujo se acumula — "
            "los contaminantes se concentran aquí → zonas prioritarias para el AUV  \n"
            "- `div > 0`  →  **divergencia**: el flujo se dispersa  \n"
            "- `div ≈ 0`  →  flujo neutro  \n\n"
            "No hay parámetros ajustables: el resultado depende únicamente "
            "del campo cargado en Fase 1."
        )
        with st.expander("Ver fórmula matemática", expanded=False):
            st.latex(
                r"\nabla \cdot \mathbf{u} "
                r"= \frac{\partial u_o}{\partial x} + \frac{\partial v_o}{\partial y}"
            )
            st.caption("Calculada con diferencias finitas centradas (`numpy.gradient`).")

    # ── Calcular (automático) y mostrar resultado ───────────────────────────
    with st.spinner("Calculando divergencia…"):
        div, dx, dy = _calcular_div(nc_path)
        st.session_state.div = div
        st.session_state.dx  = dx
        st.session_state.dy  = dy

    nav  = campo.navegable[capa]
    conv = int((nav & (div < 0)).sum())
    agua = int(nav.sum())

    st.markdown("**Resultado**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Divergencia máx.",    f"{float(div[nav].max()):.2e} /s")
    col2.metric("Convergencia mín.",   f"{float(div[nav].min()):.2e} /s")
    col3.metric("Celdas convergentes", f"{conv} / {agua}  ({100*conv/agua:.0f} %)")

    with st.spinner("Renderizando mapa de divergencia…"):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_divergencia(campo, div, capa=capa, ax=ax)
        _mostrar_figura(fig, "medio")
        plt.close(fig)

    st.caption(
        "Azul oscuro = convergencia fuerte (flujo se acumula)  ·  "
        "Rojo oscuro = divergencia fuerte (flujo se dispersa)  ·  "
        "Tono marrón = tierra"
    )

    # ── Continuar (solo cuando la fase está activa) ────────────────────────
    if st.session_state.fase_actual == 2:
        st.markdown("")
        if st.button(
            "▶  Continuar → Fase 3: Selección de zonas de misión",
            type="primary",
            key="btn_continuar_f2",
        ):
            st.session_state.fase_actual = 3
            st.rerun()


# ---------------------------------------------------------------------------
# FASE 3 — Selección de zonas de misión
# ---------------------------------------------------------------------------
def _render_fase3() -> None:
    fase = st.session_state.fase_actual

    c_titulo, c_badge = st.columns([7, 1])
    with c_titulo:
        st.markdown("### FASE 3 · Selección de zonas de misión")
    with c_badge:
        if fase > 3:
            st.markdown("**✅ Completada**")
        elif fase == 3:
            st.markdown("**🔵 Activa**")
        else:
            st.caption("Completa Fase 2 primero")

    if fase < 3:
        st.divider()
        return

    if fase > 3:
        with st.expander("Ver zonas de misión seleccionadas", expanded=False):
            _render_contenido_fase3()
        st.divider()
        return

    _render_contenido_fase3()
    st.divider()


def _render_selector_base(campo) -> None:
    """Tarjetas para seleccionar la base de partida y retorno del AUV."""
    bases_custom = st.session_state.bases_personalizadas
    base_key     = st.session_state.base_key
    capa         = 0
    nav          = campo.navegable[capa]

    n_tarjetas = 1 + len(bases_custom) + 1          # callao + custom + "+"
    cols       = st.columns(min(n_tarjetas, 4))

    # ── Tarjeta Callao ──────────────────────────────────────────────────
    with cols[0]:
        bc = celda_mas_cercana(_LAT_CALLAO, _LON_CALLAO, campo.lat, campo.lon, nav, capa=capa)
        _, bi, bj = bc
        selec = (base_key == "callao")
        with st.container(border=True):
            st.markdown("**⚓ Puerto del Callao**")
            st.markdown(
                f"Lat: `{_LAT_CALLAO:.3f}°`  \n"
                f"Lon: `{_LON_CALLAO:.3f}°`  \n"
                f"Celda navegable: `({campo.lat[bi]:.3f}°, {campo.lon[bj]:.3f}°)`  \n"
                f"Estado: ✅ Navegable"
            )
            lbl  = "✅  Seleccionada" if selec else "Seleccionar"
            tipo = "primary" if selec else "secondary"
            if st.button(lbl, key="base_callao", type=tipo, use_container_width=True):
                st.session_state.base_key = "callao"
                st.rerun()

    # ── Tarjetas bases personalizadas ───────────────────────────────────
    for idx, info in enumerate(bases_custom):
        col_idx = 1 + idx
        if col_idx >= min(n_tarjetas, 4) - 1:
            break
        with cols[col_idx]:
            bc = celda_mas_cercana(info["lat"], info["lon"],
                                   campo.lat, campo.lon, nav, capa=capa)
            _, bi, bj = bc

            en_dominio = (
                campo.lat.min() <= info["lat"] <= campo.lat.max()
                and campo.lon.min() <= info["lon"] <= campo.lon.max()
            )
            d_km = math.hypot(
                (campo.lat[bi] - info["lat"]) * 111.32,
                (campo.lon[bj] - info["lon"]) * 111.32
                * math.cos(math.radians(float(campo.lat.mean()))),
            )
            estado = "✅ Navegable" if en_dominio else "⚠️ Fuera del dominio"

            selec = (base_key == idx)
            with st.container(border=True):
                st.markdown(f"**📍 {info['nombre']}**")
                st.markdown(
                    f"Lat: `{info['lat']:.3f}°`  \n"
                    f"Lon: `{info['lon']:.3f}°`  \n"
                    f"Celda navegable: `({campo.lat[bi]:.3f}°, {campo.lon[bj]:.3f}°)`  \n"
                    + (f"Ajuste: ≈ {d_km:.1f} km  \n" if d_km > 0.5 else "")
                    + f"Estado: {estado}"
                )
                lbl  = "✅  Seleccionada" if selec else "Seleccionar"
                tipo = "primary" if selec else "secondary"
                if st.button(lbl, key=f"base_custom_{idx}",
                             type=tipo, use_container_width=True):
                    st.session_state.base_key = idx
                    st.rerun()

    # ── Tarjeta "+" ─────────────────────────────────────────────────────
    with cols[-1]:
        with st.container(border=True):
            st.markdown("**➕ Base personalizada**")
            nombre_inp = st.text_input(
                "Nombre de la base",
                placeholder="Ej: Puerto Salaverry",
                key="inp_base_nombre",
            )
            lat_inp = st.number_input(
                "Latitud [°]", value=-12.0, step=0.01, format="%.3f",
                key="inp_base_lat",
            )
            lon_inp = st.number_input(
                "Longitud [°]", value=-77.0, step=0.01, format="%.3f",
                key="inp_base_lon",
            )
            if st.button("Agregar base", key="btn_agregar_base",
                         use_container_width=True):
                nombre_final = nombre_inp.strip() or f"Base {len(bases_custom) + 1}"
                st.session_state.bases_personalizadas.append({
                    "nombre": nombre_final,
                    "lat":    lat_inp,
                    "lon":    lon_inp,
                })
                st.session_state.base_key = len(st.session_state.bases_personalizadas) - 1
                st.rerun()


def _render_resultado_fase3(campo, div) -> None:
    wps     = st.session_state.wps
    cent    = st.session_state.cent
    base    = st.session_state.base_celda
    n_inter = len(st.session_state.todos) - 1
    ordenes = math.factorial(n_inter)

    st.markdown("**Resultado**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zonas convergencia",  len(wps))
    col2.metric("Centinelas offshore", len(cent))
    col3.metric("Total puntos",        len(st.session_state.todos))
    col4.metric("Órdenes ATSP",        f"{ordenes:,}")

    with st.spinner("Renderizando mapa de zonas…"):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_zonas(campo, div, wps, centinelas=cent, base=base, capa=0, ax=ax)
        _mostrar_figura(fig, "medio")
        plt.close(fig)

    st.markdown("**Detalle de waypoints**")
    tabla = []
    for k, (p, i, j) in enumerate(wps):
        tabla.append({
            "ID":   f"C{k+1}",
            "Tipo": "Convergencia",
            "Lat [°]": f"{campo.lat[i]:.3f}",
            "Lon [°]": f"{campo.lon[j]:.3f}",
            "Divergencia [1/s]": f"{div[i, j]:.2e}",
        })
    for k, (p, i, j) in enumerate(cent):
        tabla.append({
            "ID":   f"S{k+1}",
            "Tipo": "Centinela",
            "Lat [°]": f"{campo.lat[i]:.3f}",
            "Lon [°]": f"{campo.lon[j]:.3f}",
            "Divergencia [1/s]": "—",
        })
    _, bi, bj = base
    bk = st.session_state.base_key
    bnombre = (
        st.session_state.bases_personalizadas[bk]["nombre"]
        if bk != "callao" else "Callao"
    )
    tabla.append({
        "ID":   "B",
        "Tipo": f"Base ({bnombre})",
        "Lat [°]": f"{campo.lat[bi]:.3f}",
        "Lon [°]": f"{campo.lon[bj]:.3f}",
        "Divergencia [1/s]": "—",
    })
    st.table(tabla)


def _render_contenido_fase3() -> None:
    campo   = st.session_state.campo
    div     = st.session_state.div
    capa    = 0
    nav     = campo.navegable[capa]

    # ── ¿Qué entra? ──────────────────────────────────────────────────────
    st.markdown("**¿Qué entra en esta fase?**")
    st.info(
        "▸ Campo de divergencia calculado en Fase 2  \n"
        "▸ Máscara de celdas navegables (sin tierra)  \n"
        "▸ Componentes `uo`, `vo` superficiales para selección de centinelas"
    )

    # ── Configurar zonas ─────────────────────────────────────────────────
    st.markdown("**Configurar zonas de misión**")
    with st.container(border=True):
        k_zonas = st.slider(
            "Zonas de convergencia a visitar  \n"
            "*Celdas con mayor acumulación de contaminantes*",
            2, 8, st.session_state.k_zonas_f3, key="sl_k_zonas",
        )
        k_cent = st.slider(
            "Centinelas offshore  \n"
            "*Puntos de detección temprana en la franja oceánica abierta*",
            1, 4, st.session_state.k_cent_f3, key="sl_k_cent",
        )
        dist_min = st.slider(
            "Separación mínima entre zonas [celdas]  \n"
            "*Evita seleccionar celdas contiguas de la misma zona*",
            1, 6, st.session_state.dist_min_f3, key="sl_dist_min",
        )
        st.session_state.k_zonas_f3  = k_zonas
        st.session_state.k_cent_f3   = k_cent
        st.session_state.dist_min_f3 = dist_min

    # ── Base de la misión ─────────────────────────────────────────────────
    st.markdown("**Base de la misión**")
    st.caption("Punto de partida y retorno del AUV")
    _render_selector_base(campo)

    # ── Botón calcular ────────────────────────────────────────────────────
    if st.session_state.fase_actual == 3:
        st.markdown("")
        if st.button("Seleccionar zonas", type="primary", key="btn_calcular_f3"):
            base_key = st.session_state.base_key
            b_lat    = _LAT_CALLAO if base_key == "callao" else st.session_state.bases_personalizadas[base_key]["lat"]
            b_lon    = _LON_CALLAO if base_key == "callao" else st.session_state.bases_personalizadas[base_key]["lon"]

            with st.spinner("Seleccionando zonas…"):
                uo   = campo.uo[capa]
                vo   = campo.vo[capa]
                wps  = seleccionar_waypoints(div, nav, k_zonas, capa=capa, dist_min_celdas=dist_min)
                cent = seleccionar_centinelas(uo, vo, nav, campo.lon, n=k_cent, capa=capa)
                base = celda_mas_cercana(b_lat, b_lon, campo.lat, campo.lon, nav, capa=capa)
                todos, base_nodo = agregar_puntos_fijos(wps + cent, wps[0], base)

            st.session_state.wps        = wps
            st.session_state.cent       = cent
            st.session_state.base_celda = base
            st.session_state.todos      = todos
            st.session_state.base_nodo  = base_nodo
            st.session_state.base_idx   = todos.index(base_nodo)
            st.rerun()

    # ── Resultado ─────────────────────────────────────────────────────────
    if st.session_state.wps is not None:
        _render_resultado_fase3(campo, div)

    # ── Continuar ─────────────────────────────────────────────────────────
    if st.session_state.fase_actual == 3 and st.session_state.wps is not None:
        st.markdown("")
        if st.button(
            "▶  Continuar → Fase 4: Modelo energético y grafo",
            type="primary", key="btn_continuar_f3",
        ):
            st.session_state.fase_actual = 4
            st.rerun()


# ---------------------------------------------------------------------------
# FASE 4 — Modelo energético y grafo
# ---------------------------------------------------------------------------

def _drone_img_html() -> str:
    b64 = base64.b64encode(_SVG_AUV.encode()).decode()
    return (
        '<div style="width:110px;height:70px;flex-shrink:0;display:flex;'
        'align-items:center;justify-content:center;overflow:hidden;'
        'border-radius:6px;background:rgba(46,139,158,0.07);">'
        f'<img src="data:image/svg+xml;base64,{b64}" '
        'style="max-width:100%;max-height:100%;object-fit:contain;"/>'
        '</div>'
    )


@st.cache_resource(show_spinner=False)
def _construir_grafo_cache(
    nc_path: str,
    s: float, eta: float, k_p: float, k_r: float, e_max: float, k_zonas: int,
):
    campo  = cargar_corrientes(nc_path)
    params = ParametrosModelo(s=s, eta=eta, k_p=k_p, k_r=k_r, e_max=e_max, k_zonas=k_zonas)
    return construir_grafo(campo, params), params


@st.dialog("Drone", width="large")
def _dialogo_drone(idx: int) -> None:
    """idx == -1 → nuevo drone; idx >= 0 → editar existente."""
    es_nuevo = idx == -1
    drones   = st.session_state.drones
    base     = {} if es_nuevo else drones[idx]

    st.subheader("Agregar drone" if es_nuevo else f"Editar — {base.get('nombre','')}")
    st.divider()

    nombre = st.text_input(
        "Nombre del drone",
        value=base.get("nombre", ""),
        placeholder="Ej: AUV Bluefin-9",
    )

    st.markdown("**Parámetros hidrodinámicos**")
    c1, c2 = st.columns(2)
    with c1:
        s = st.number_input(
            "Velocidad de crucero v [m/s]",
            min_value=0.1, max_value=3.0,
            value=float(base.get("s", 0.5)), step=0.05, format="%.2f",
            help="Velocidad del AUV respecto al agua. Valores típicos: 0.3–1.5 m/s.",
        )
        st.caption("Velocidad del cuerpo relativa al agua (no respecto al fondo).")
    with c2:
        eta = st.number_input(
            "Eficiencia de regeneración η [0–1]",
            min_value=0.01, max_value=0.99,
            value=float(base.get("eta", 0.30)), step=0.01, format="%.2f",
            help="Fracción de energía cinética convertida en carga eléctrica al regenerar.",
        )
        st.caption("Fracción de energía recuperada al dejarse llevar por la corriente.")

    st.markdown("**Coeficientes energéticos**")
    c3, c4 = st.columns(2)
    with c3:
        k_p = st.number_input(
            "Coeficiente de propulsión kp",
            min_value=0.01, max_value=10.0,
            value=float(base.get("k_p", 1.0)), step=0.1, format="%.2f",
            help="Escala el consumo energético en modo propulsión. Mayor kp → más gasto.",
        )
        st.caption("Escala el consumo en modo propulsión (corriente en contra o lateral).")
    with c4:
        k_r = st.number_input(
            "Coeficiente de regeneración kr",
            min_value=0.01, max_value=10.0,
            value=float(base.get("k_r", 1.0)), step=0.1, format="%.2f",
            help="Escala la energía recuperada en modo regeneración. Mayor kr → más ganancia.",
        )
        st.caption("Escala la ganancia energética en modo regeneración (corriente a favor).")

    st.markdown("**Batería**")
    c5, c6 = st.columns(2)
    with c5:
        e_max_kj = st.number_input(
            "Capacidad máxima [kJ]",
            min_value=1.0, max_value=100_000.0,
            value=float(base.get("e_max", 1_000_000)) / 1_000.0,
            step=10.0, format="%.0f",
            help="Energía almacenada al 100 % de carga.",
        )
        st.caption("Energía total de la batería en kilojulios (1 kJ = 1 000 J).")
    with c6:
        pct_ini = st.number_input(
            "Carga inicial [%]",
            min_value=1, max_value=100,
            value=int(base.get("pct_ini", 100)), step=1,
            help="Estado de carga al inicio de la misión.",
        )
        st.caption("Porcentaje de carga con que el AUV comienza la misión.")

    st.divider()
    bc, bs = st.columns(2)
    with bc:
        if st.button("Cancelar", use_container_width=True, key="dlg_cancel"):
            st.session_state.abrir_drone_idx = None
            st.rerun()
    with bs:
        if st.button("Guardar drone", type="primary", use_container_width=True, key="dlg_save"):
            nuevo = {
                "nombre":  nombre.strip() or (f"Drone {len(drones)+1}" if es_nuevo else base["nombre"]),
                "s":       s,
                "eta":     eta,
                "k_p":     k_p,
                "k_r":     k_r,
                "e_max":   int(e_max_kj * 1_000),
                "pct_ini": int(pct_ini),
            }
            if es_nuevo:
                st.session_state.drones.append(nuevo)
                st.session_state.drone_key = len(st.session_state.drones) - 1
            else:
                st.session_state.drones[idx] = nuevo
            # invalidar grafo y resultado ATSP anteriores
            st.session_state.grafo        = None
            st.session_state.params_f4    = None
            st.session_state.hay_ciclo_f4 = None
            st.session_state.ruta         = None
            st.session_state.orden_f5     = None
            st.session_state.costo_f5     = None
            st.session_state.M_f5         = None
            st.session_state.caminos_f5   = None
            if st.session_state.fase_actual >= 5:
                st.session_state.fase_actual = 4
            st.session_state.abrir_drone_idx = None
            st.rerun()


def _render_tarjeta_drone(idx: int) -> None:
    """Tarjeta compacta con los datos del drone y botones de acción."""
    drone = st.session_state.drones[idx]
    selec = st.session_state.drone_key == idx
    e_kj  = drone["e_max"] / 1_000

    with st.container(border=True):
        # ── Cabecera ──────────────────────────────────────────────────
        h1, h2 = st.columns([5, 1])
        with h1:
            st.markdown(f"**🤖 {drone['nombre']}**")
        with h2:
            if selec:
                st.markdown("**✅**")

        # ── Parámetros + imagen ───────────────────────────────────────
        p_col, img_col = st.columns([3, 1])
        with p_col:
            st.markdown(
                f"**v** `{drone['s']:.2f} m/s`&nbsp;&nbsp;&nbsp;"
                f"**η** `{drone['eta']:.2f}`  \n"
                f"**kp** `{drone['k_p']:.2f}`&nbsp;&nbsp;&nbsp;&nbsp;"
                f"**kr** `{drone['k_r']:.2f}`  \n"
                f"**Bat** `{e_kj:.0f} kJ`&nbsp;&nbsp;"
                f"**Ini** `{drone['pct_ini']} %`"
            )
        with img_col:
            st.markdown(_drone_img_html(), unsafe_allow_html=True)

        # ── Botones de acción ─────────────────────────────────────────
        b1, b2, b3 = st.columns(3)
        with b1:
            lbl  = "✅ Activo" if selec else "Seleccionar"
            tipo = "primary" if selec else "secondary"
            if st.button(lbl, key=f"sel_d_{idx}", type=tipo, use_container_width=True):
                st.session_state.drone_key    = idx
                st.session_state.grafo        = None
                st.session_state.hay_ciclo_f4 = None
                st.session_state.ruta         = None
                st.session_state.orden_f5     = None
                st.session_state.costo_f5     = None
                st.session_state.M_f5         = None
                st.session_state.caminos_f5   = None
                if st.session_state.fase_actual >= 5:
                    st.session_state.fase_actual = 4
                st.rerun()
        with b2:
            if st.button("✏️ Editar", key=f"edit_d_{idx}", use_container_width=True):
                st.session_state.abrir_drone_idx = idx
                st.rerun()
        with b3:
            if st.button("🗑️", key=f"del_d_{idx}", use_container_width=True):
                st.session_state.drones.pop(idx)
                n = len(st.session_state.drones)
                if n == 0:
                    st.session_state.drones = [{
                        "nombre": "AUV Estándar", "s": 0.5, "eta": 0.30,
                        "k_p": 1.0, "k_r": 1.0, "e_max": 1_000_000, "pct_ini": 100,
                    }]
                if st.session_state.drone_key >= len(st.session_state.drones):
                    st.session_state.drone_key = max(0, len(st.session_state.drones) - 1)
                st.session_state.grafo        = None
                st.session_state.hay_ciclo_f4 = None
                st.session_state.ruta         = None
                st.session_state.orden_f5     = None
                st.session_state.costo_f5     = None
                st.session_state.M_f5         = None
                st.session_state.caminos_f5   = None
                if st.session_state.fase_actual >= 5:
                    st.session_state.fase_actual = 4
                st.rerun()


def _render_gestor_drones() -> None:
    """Grid de tarjetas 2 por fila con botón de agregar al final."""
    drones = st.session_state.drones

    # Abrir diálogo si corresponde (debe estar en la parte superior del render)
    if st.session_state.abrir_drone_idx is not None:
        _dialogo_drone(st.session_state.abrir_drone_idx)

    for i in range(0, len(drones), 2):
        c1, c2 = st.columns(2)
        with c1:
            _render_tarjeta_drone(i)
        if i + 1 < len(drones):
            with c2:
                _render_tarjeta_drone(i + 1)

    st.markdown("")
    if st.button("➕ Agregar drone", key="btn_agregar_drone"):
        st.session_state.abrir_drone_idx = -1
        st.rerun()


def _render_fase4() -> None:
    fase = st.session_state.fase_actual

    c_titulo, c_badge = st.columns([7, 1])
    with c_titulo:
        st.markdown("### FASE 4 · Modelo energético y grafo")
    with c_badge:
        if fase > 4:
            st.markdown("**✅ Completada**")
        elif fase == 4:
            st.markdown("**🔵 Activa**")
        else:
            st.caption("Completa Fase 3 primero")

    if fase < 4:
        st.divider()
        return

    if fase > 4:
        with st.expander("Ver modelo energético y grafo construido", expanded=False):
            _render_contenido_fase4()
        st.divider()
        return

    _render_contenido_fase4()
    st.divider()


def _render_contenido_fase4() -> None:
    campo    = st.session_state.campo
    nc_path  = st.session_state.nc_path
    todos    = st.session_state.todos
    base_nodo = st.session_state.base_nodo

    # ── ¿Qué entra? ─────────────────────────────────────────────────────────
    st.markdown("**¿Qué entra en esta fase?**")
    n_nodos_potenciales = int(campo.navegable.sum()) if campo is not None else 0
    st.info(
        f"▸ Campo de corrientes — `{pathlib.Path(nc_path).name}`  \n"
        f"▸ {len(todos)} puntos de misión de Fase 3 "
        f"(waypoints, centinelas y base)  \n"
        f"▸ {n_nodos_potenciales:,} celdas navegables → nodos del grafo  \n"
        f"▸ Parámetros del drone seleccionado"
    )

    # ── ¿Qué se calcula? ────────────────────────────────────────────────────
    st.markdown("**¿Qué se calcula?**")
    with st.container(border=True):
        st.markdown(
            "Se construye un **grafo dirigido** donde cada celda navegable es un nodo "
            "y cada arista representa un desplazamiento entre celdas adyacentes.\n\n"
            "El **peso de cada arista** es la energía neta del movimiento:\n\n"
        )
        st.latex(
            r"E_{u \to v} = \begin{cases}"
            r"k_p \cdot |\mathbf{v}_r|^3 \cdot \frac{L}{s} & \text{propulsión (contra corriente)}\\"
            r"-k_r \cdot \eta \cdot |\mathbf{v}_r|^3 \cdot \frac{L}{s} & \text{regeneración (a favor)}"
            r"\end{cases}"
        )
        st.markdown(
            "Donde **v_r = s·ê − v_c** es la velocidad relativa al agua, "
            "**L** es la distancia entre celdas y **s** la velocidad de crucero.  \n"
            "Pesos negativos → el AUV *recupera* energía (por eso se usa Bellman-Ford, no Dijkstra)."
        )

    # ── Gestor de drones ────────────────────────────────────────────────────
    st.markdown("**Seleccioná el drone para la misión**")
    _render_gestor_drones()

    # ── Botón construir grafo (solo cuando la fase está activa) ─────────────
    if st.session_state.fase_actual == 4:
        st.markdown("")
        drone_sel = st.session_state.drones[st.session_state.drone_key]
        if st.button("⚙️  Construir grafo", type="primary", key="btn_construir_grafo"):
            with st.spinner("Construyendo grafo de energía — puede tardar unos segundos…"):
                grafo, params = _construir_grafo_cache(
                    nc_path,
                    s=drone_sel["s"],
                    eta=drone_sel["eta"],
                    k_p=drone_sel["k_p"],
                    k_r=drone_sel["k_r"],
                    e_max=drone_sel["e_max"],
                    k_zonas=st.session_state.k_zonas_f3,
                )
            with st.spinner("Verificando ciclos de energía negativa…"):
                dist_val, _ = bellman_ford(grafo, base_nodo)
                hay_ciclo   = hay_ciclo_negativo(grafo, dist_val)

            st.session_state.grafo        = grafo
            st.session_state.params_f4    = params
            st.session_state.bat_ini_j    = int(drone_sel["e_max"] * drone_sel["pct_ini"] / 100)
            st.session_state.hay_ciclo_f4 = hay_ciclo
            st.session_state.ruta         = None
            st.session_state.orden_f5     = None
            st.session_state.costo_f5     = None
            st.session_state.M_f5         = None
            st.session_state.caminos_f5   = None
            st.rerun()

    # ── Resultado ────────────────────────────────────────────────────────────
    if st.session_state.grafo is not None:
        grafo     = st.session_state.grafo
        params    = st.session_state.params_f4
        bat_ini   = st.session_state.bat_ini_j
        hay_ciclo = st.session_state.hay_ciclo_f4

        st.markdown("**Resultado**")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodos del grafo",    f"{grafo.num_nodos:,}")
        col2.metric("Aristas dirigidas",  f"{grafo.num_aristas:,}")
        col3.metric("Batería inicial",    f"{bat_ini / 1_000:.0f} kJ")
        col4.metric(
            "Ciclo negativo",
            "⚠️ Detectado" if hay_ciclo else "✅ Ninguno",
        )

        if hay_ciclo:
            st.warning(
                "Se detectó un ciclo de energía negativa. Esto indica que el AUV podría "
                "ganar energía infinita dando vueltas, lo que suele reflejar una calibración "
                "incorrecta de kp, kr o η. Ajustá los parámetros del drone antes de continuar."
            )
        else:
            st.success(
                f"Grafo válido — {grafo.num_nodos:,} nodos, {grafo.num_aristas:,} aristas. "
                "Sin ciclos de energía negativa: el modelo es físicamente consistente."
            )

    # ── Continuar ─────────────────────────────────────────────────────────────
    if (
        st.session_state.fase_actual == 4
        and st.session_state.grafo is not None
        and not st.session_state.hay_ciclo_f4
    ):
        st.markdown("")
        if st.button(
            "▶  Continuar → Fase 5: Optimización ATSP + Bellman-Ford",
            type="primary", key="btn_continuar_f4",
        ):
            st.session_state.fase_actual = 5
            st.rerun()


# ---------------------------------------------------------------------------
# Fase 5 — Optimización ATSP + Bellman-Ford
# ---------------------------------------------------------------------------
def _render_contenido_fase5() -> None:
    campo     = st.session_state.campo
    todos     = st.session_state.todos
    wps       = st.session_state.wps
    cent      = st.session_state.cent
    base_nodo = st.session_state.base_nodo
    base_idx  = st.session_state.base_idx
    grafo     = st.session_state.grafo

    # ── ¿Qué entra? ─────────────────────────────────────────────────────────
    st.markdown("**¿Qué entra en esta fase?**")
    st.info(
        f"▸ Grafo energético de Fase 4 — "
        f"{grafo.num_nodos:,} nodos, {grafo.num_aristas:,} aristas  \n"
        f"▸ {len(todos)} zonas de misión de Fase 3 "
        f"({len(wps)} waypoints de convergencia, {len(cent)} centinelas y base)"
    )

    # ── ¿Qué se calcula? ────────────────────────────────────────────────────
    st.markdown("**¿Qué se calcula?**")
    n_zonas = len(todos)
    n_factorial = math.factorial(n_zonas - 1)
    with st.container(border=True):
        st.markdown(
            "**Paso 1 — Bellman-Ford multi-fuente**  \n"
            f"Se ejecuta Bellman-Ford desde cada una de las {n_zonas} zonas "
            "recorriendo el grafo completo. El resultado es una **matriz "
            f"{n_zonas}×{n_zonas}** donde M[i,j] es la energía mínima [J] "
            "para ir de la zona i a la zona j.\n\n"
            f"**Paso 2 — ATSP fuerza bruta**  \n"
            f"Se enumeran las **(N−1)! = {n_factorial:,}** permutaciones de "
            f"los {n_zonas - 1} nodos intermedios y se elige el tour "
            "Base → … → Base de menor costo total.\n\n"
            "**Paso 3 — Ensamblado de ruta**  \n"
            "Los tramos Bellman-Ford del tour óptimo se concatenan en una "
            "secuencia completa de nodos: Base → … → Base."
        )

    # ── Botón calcular (solo en fase activa) ────────────────────────────────
    if st.session_state.fase_actual == 5:
        st.markdown("")
        if st.button("⚙️  Calcular ruta óptima", type="primary",
                     key="btn_calcular_f5"):
            with st.spinner(
                "Calculando matriz de costos energéticos "
                f"({n_zonas} ejecuciones de Bellman-Ford)…"
            ):
                M, caminos = matriz_costos(grafo, todos)
            with st.spinner("Resolviendo ATSP por fuerza bruta…"):
                orden, costo = atsp_fuerza_bruta(M, base=base_idx)
            ruta = ensamblar_ruta(orden, caminos)
            st.session_state.M_f5       = M
            st.session_state.caminos_f5 = caminos
            st.session_state.orden_f5   = orden
            st.session_state.costo_f5   = costo
            st.session_state.ruta       = ruta
            st.rerun()

    # ── Resultado ────────────────────────────────────────────────────────────
    if st.session_state.ruta is None:
        return

    M     = st.session_state.M_f5
    orden = st.session_state.orden_f5
    costo = st.session_state.costo_f5
    ruta  = st.session_state.ruta

    # Etiquetas por tipo de zona
    etiquetas: dict = {}
    for nodo in todos:
        if nodo == base_nodo:
            etiquetas[nodo] = "Base"
        elif nodo in wps:
            etiquetas[nodo] = f"C{wps.index(nodo) + 1}"
        elif nodo in cent:
            etiquetas[nodo] = f"S{cent.index(nodo) + 1}"
        else:
            etiquetas[nodo] = f"Z{todos.index(nodo)}"
    etiq_lista = [etiquetas[nodo] for nodo in todos]

    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zonas de misión",   str(n_zonas))
    col2.metric("Órdenes evaluados", f"{math.factorial(n_zonas - 1):,}")
    col3.metric("Nodos en ruta",     f"{len(ruta):,}")
    col4.metric("Costo total",       f"{costo:.1f} J")

    # Aviso si algún par es inalcanzable
    inalcanzables = [
        (i, j)
        for i in range(n_zonas)
        for j in range(n_zonas)
        if i != j and not math.isfinite(float(M[i, j]))
    ]
    if inalcanzables:
        pares = ", ".join(
            f"{etiq_lista[i]}→{etiq_lista[j]}" for i, j in inalcanzables[:5]
        )
        st.warning(
            f"{len(inalcanzables)} par(es) de zonas son inalcanzables entre sí: "
            f"{pares}. El tour óptimo puede ser infinito o subóptimo."
        )

    # ── Visualizaciones en tabs ───────────────────────────────────────────
    tab_bf, tab_atsp = st.tabs([
        "📊 Paso 1 · Grafo de costos BF",
        "🔍 Paso 2 · Búsqueda ATSP",
    ])

    with tab_bf:
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        plot_grafo_costos(todos, campo, M, etiquetas, ax=ax1)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        st.caption(
            "Cada arista i → j representa el costo energético mínimo calculado "
            "por Bellman-Ford. Verde = barato, rojo = caro. "
            "La asimetría A→B ≠ B→A es visible en los arcos curvados."
        )

    with tab_atsp:
        fig2 = plot_tours_atsp(M, etiq_lista, orden)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.caption(
            "Izquierda: los (N−1)! tours ordenados por costo total; "
            "el ganador (verde) es el de menor energía. "
            "Derecha: ciclo del tour óptimo con el costo de cada tramo [J]."
        )

    # ── Matriz completa ───────────────────────────────────────────────────
    with st.expander("📋 Matriz de costos completa [J]", expanded=False):
        import pandas as pd
        df = pd.DataFrame(
            [
                [
                    "—" if i == j
                    else ("∞" if not math.isfinite(float(M[i, j]))
                          else f"{M[i, j]:.1f}")
                    for j in range(n_zonas)
                ]
                for i in range(n_zonas)
            ],
            index=etiq_lista,
            columns=etiq_lista,
        )
        st.dataframe(df, use_container_width=True)

    # ── Ruta resumen ──────────────────────────────────────────────────────
    secuencia_str = "  →  ".join(etiq_lista[k] for k in orden)
    with st.container(border=True):
        st.markdown(f"**Ruta óptima:** {secuencia_str}")
        st.markdown(
            f"**Costo total de la misión:** `{costo:.1f} J`  —  "
            f"`{len(ruta):,}` nodos en la secuencia completa"
        )

    # ── Continuar ─────────────────────────────────────────────────────────
    if st.session_state.fase_actual == 5:
        st.markdown("")
        if st.button(
            "▶  Continuar → Fase 6: Resultados + Exportar",
            type="primary", key="btn_continuar_f5",
        ):
            st.session_state.fase_actual = 6
            st.rerun()


def _render_fase5() -> None:
    fase = st.session_state.fase_actual

    c_titulo, c_badge = st.columns([7, 1])
    with c_titulo:
        st.markdown("### FASE 5 · Optimización ATSP + Bellman-Ford")
    with c_badge:
        if fase > 5:
            st.markdown("**✅ Completada**")
        elif fase == 5:
            st.markdown("**🔵 Activa**")
        else:
            st.caption("Completa Fase 4 primero")

    if fase < 5:
        st.divider()
        return

    if fase > 5:
        with st.expander("Ver optimización ATSP y ruta calculada", expanded=False):
            _render_contenido_fase5()
        st.divider()
        return

    _render_contenido_fase5()
    st.divider()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Fase 6 — Resultados de la misión + Exportar
# ---------------------------------------------------------------------------
def _render_contenido_fase6() -> None:
    campo     = st.session_state.campo
    ruta      = st.session_state.ruta
    orden     = st.session_state.orden_f5
    costo     = st.session_state.costo_f5
    M         = st.session_state.M_f5
    grafo     = st.session_state.grafo
    todos     = st.session_state.todos
    wps       = st.session_state.wps
    cent      = st.session_state.cent
    base_nodo = st.session_state.base_nodo
    bat_ini   = st.session_state.bat_ini_j
    params    = st.session_state.params_f4

    # ── ¿Qué entra? ──────────────────────────────────────────────────────────
    st.markdown("**¿Qué entra en esta fase?**")
    st.info(
        f"▸ Ruta completa calculada en Fase 5 — {len(ruta):,} nodos  \n"
        f"▸ Grafo energético (Fase 4) y campo de corrientes (Fase 1)  \n"
        f"▸ Parámetros del drone seleccionado"
    )

    # ── Etiquetas de zonas ───────────────────────────────────────────────────
    etiquetas: dict = {}
    for nodo in todos:
        if nodo == base_nodo:
            etiquetas[nodo] = "Base"
        elif nodo in wps:
            etiquetas[nodo] = f"C{wps.index(nodo) + 1}"
        elif nodo in cent:
            etiquetas[nodo] = f"S{cent.index(nodo) + 1}"
        else:
            etiquetas[nodo] = f"Z{todos.index(nodo)}"
    etiq_lista = [etiquetas[nodo] for nodo in todos]

    # ── Métricas de batería ──────────────────────────────────────────────────
    bat = estado_bateria(ruta, grafo, params.e_max, bateria_inicial=bat_ini)
    pct_min = bat["minimo"] / params.e_max * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Misión viable",       "✅ Sí" if bat["viable"] else "❌ No")
    col2.metric("Energía consumida",   f"{bat['consumido']:.1f} J")
    col3.metric("Energía regenerada",  f"{bat['regenerado']:.1f} J")
    col4.metric("Batería mínima",      f"{pct_min:.1f} %")

    if not bat["viable"]:
        st.error(
            "La batería llega a 0 en algún punto de la misión. "
            "Considerá aumentar la capacidad (e_max) o reducir la distancia entre zonas."
        )

    # ── Tabla de tramos ──────────────────────────────────────────────────────
    import pandas as pd
    tramos_j = [float(M[orden[k], orden[k + 1]]) for k in range(len(orden) - 1)]
    filas = [
        {
            "Tramo": k + 1,
            "De":    etiq_lista[orden[k]],
            "→":     etiq_lista[orden[k + 1]],
            "Costo [J]": f"{tramos_j[k]:.1f}",
        }
        for k in range(len(orden) - 1)
    ]
    filas.append({"Tramo": "", "De": "", "→": "Total", "Costo [J]": f"{costo:.1f}"})
    st.markdown("**Tramos de la misión**")
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

    # ── Tabs de visualización ─────────────────────────────────────────────────
    tab_2d, tab_3d, tab_bat = st.tabs(["🗺️  Ruta 2D", "🧊  Ruta 3D", "🔋  Batería"])

    with tab_2d:
        fig2d, ax2d = plt.subplots(figsize=(9, 7))
        plot_ruta(campo, ruta, waypoints=wps, centinelas=cent, base=base_nodo, ax=ax2d)
        st.pyplot(fig2d, use_container_width=True)
        plt.close(fig2d)
        st.caption(
            "Trayecto nodo a nodo del AUV coloreado por capa de profundidad. "
            "Estrellas rojas = zonas de convergencia · "
            "Triángulos azules = centinelas · Cuadrado verde = base."
        )

    with tab_3d:
        fig3d = plot_3d(campo, ruta, waypoints=wps, centinelas=cent, base=base_nodo)
        st.pyplot(fig3d, use_container_width=True)
        plt.close(fig3d)
        st.caption(
            "Vista 3D con planos de profundidad apilados. "
            "El eje Z usa índices de capa para separación visual."
        )

    with tab_bat:
        fig_bat, ax_bat = plt.subplots(figsize=(12, 4))
        plot_bateria(
            campo, ruta, bat["niveles"], params.e_max,
            waypoints=todos, orden=orden, ax=ax_bat,
        )
        st.pyplot(fig_bat, use_container_width=True)
        plt.close(fig_bat)
        st.caption(
            "Verde = tramos de regeneración · "
            "Naranja = tramos de propulsión · "
            "Zona roja = nivel crítico (<20 % de capacidad)."
        )

    # ── Exportar CSV ──────────────────────────────────────────────────────────
    st.markdown("**Exportar ruta**")
    import io as _io
    import csv as _csv
    buf = _io.StringIO()
    writer = _csv.writer(buf)
    writer.writerow(["paso", "prof_m", "lat_deg", "lon_deg"])
    for paso, (p, i, j) in enumerate(ruta):
        writer.writerow([
            paso,
            round(float(campo.prof[p]), 4),
            round(float(campo.lat[i]), 6),
            round(float(campo.lon[j]), 6),
        ])
    st.download_button(
        label="⬇️  Descargar ruta_auv.csv",
        data=buf.getvalue().encode("utf-8"),
        file_name="ruta_auv.csv",
        mime="text/csv",
    )
    st.caption(
        f"Columnas: paso · prof_m · lat_deg · lon_deg  —  "
        f"{len(ruta):,} filas (una por cada nodo de la ruta completa)"
    )


def _render_fase6() -> None:
    fase = st.session_state.fase_actual

    c_titulo, c_badge = st.columns([7, 1])
    with c_titulo:
        st.markdown("### FASE 6 · Resultados de la misión + Exportar")
    with c_badge:
        if fase == 6:
            st.markdown("**🔵 Activa**")
        else:
            st.caption("Completa Fase 5 primero")

    if fase < 6:
        st.divider()
        return

    _render_contenido_fase6()
    st.divider()


# ---------------------------------------------------------------------------
# Stub genérico (reservado para fases futuras)
# ---------------------------------------------------------------------------
def _render_fase_stub(num: int) -> None:
    nombre, msg_bloqueo = _FASES_DESCRIPCIONES[num]
    fase_actual = st.session_state.fase_actual

    if fase_actual < num:
        c1, c2 = st.columns([7, 1])
        with c1:
            st.markdown(f"### 🔒 FASE {num} · {nombre}")
        with c2:
            st.caption(msg_bloqueo)
    elif fase_actual == num:
        st.markdown(f"### 🔵 FASE {num} · {nombre}")
        st.info("En construcción — próxima fase a implementar.")
    else:
        with st.expander(f"✅ FASE {num} · {nombre}", expanded=False):
            st.caption("Completada.")

    st.divider()


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Planificador AUV — Lima",
    page_icon="🌊",
    layout="wide",
)

st.markdown(_ESTILOS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="auv-hero">
        <h1>🌊 Planificador de rutas AUV · Lima</h1>
        <div class="sub">
            Sistema de planificación de misiones de reconocimiento para Vehículos Submarinos
            Autónomos en el litoral de Lima-Callao. Aprovecha corrientes marinas de Copernicus
            Marine para minimizar el consumo energético.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_init_state()
_render_progreso()

_render_fase1()
_render_fase2()
_render_fase3()
_render_fase4()
_render_fase5()
_render_fase6()
