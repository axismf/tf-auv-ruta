"""Interfaz web del planificador de rutas (RF-06, RF-07, RF-08, RF-09).

Capa de presentación construida con Streamlit. Solo orquesta los módulos del
núcleo (no contiene lógica de algoritmos), de modo que la UI pueda cambiarse
sin tocar el core.

Ejecutar con:
    streamlit run app.py
"""
from __future__ import annotations

import math
import io
import pathlib
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

from src.config import ParametrosModelo
from src.datos import cargar_corrientes, resumen
from src.grafo import construir_grafo
from src.zonas import (
    divergencia,
    seleccionar_waypoints,
    seleccionar_centinelas,
    celda_mas_cercana,
    agregar_puntos_fijos,
)
from src.algoritmos import matriz_costos, atsp_fuerza_bruta, ensamblar_ruta
from src.metricas import resumen_mision, energia_total, estado_bateria, exportar_csv
from src.visualizacion import plot_zonas, plot_ruta, plot_3d, plot_bateria

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_LAT_CALLAO = -12.05
_LON_CALLAO = -77.15
_CAPA_SUPERFICIE = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fig_a_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Planificador de rutas AUV — Lima",
    page_icon="🤿",
    layout="wide",
)

st.title("🤿 Planificador de rutas de mínima energía para AUV")
st.caption(
    "Modela el litoral de Lima como un grafo dirigido y calcula la ruta de "
    "reconocimiento de menor consumo energético, considerando zonas de "
    "convergencia de contaminantes y centinelas offshore."
)

# ---------------------------------------------------------------------------
# Barra lateral — parámetros del modelo (RF-06)
# ---------------------------------------------------------------------------
st.sidebar.header("Parámetros del modelo")

s = st.sidebar.slider(
    "Velocidad de crucero [m/s]", 0.1, 1.5, 0.5, 0.05,
    help="Velocidad del AUV respecto al agua.",
)
eta = st.sidebar.slider(
    "Eficiencia de regeneración η", 0.05, 0.95, 0.30, 0.05,
    help="Fracción de energía recuperada en régimen de regeneración.",
)
k_zonas = st.sidebar.slider(
    "Zonas de convergencia (kc)", 2, 8, 6,
    help="Número de zonas de acumulación a visitar.",
)
k_centinelas = st.sidebar.slider(
    "Centinelas offshore (ks)", 1, 4, 2,
    help="Puntos de detección temprana en la franja oceánica abierta.",
)
dist_min = st.sidebar.slider(
    "Separación mínima entre waypoints [celdas]", 1, 6, 3,
    help="Evita seleccionar celdas contiguas de la misma zona.",
)

st.sidebar.header("Batería del AUV")
e_max = st.sidebar.number_input(
    "Capacidad máxima [J]", min_value=1_000, max_value=10_000_000,
    value=1_000_000, step=50_000,
    help="Energía total almacenable en la batería del AUV.",
)
pct_inicial = st.sidebar.slider(
    "Carga inicial [%]", 10, 100, 100, 5,
    help="Porcentaje de carga al inicio de la misión.",
)
bateria_inicial_j = float(e_max) * pct_inicial / 100

params = ParametrosModelo(s=s, eta=eta, k_zonas=k_zonas, e_max=float(e_max))

# ---------------------------------------------------------------------------
# Carga del dataset (RF-01)
# ---------------------------------------------------------------------------
st.sidebar.header("Dataset")
archivo = st.sidebar.file_uploader(
    "Archivo NetCDF de Copernicus Marine (.nc)",
    type=["nc"],
    help="Producto GLOBAL_ANALYSISFORECAST_PHY_001_024 con variables uo y vo.",
)

_NC_DEFAULT = pathlib.Path(__file__).parent / "data" / "lima3.nc"


@st.cache_data(show_spinner="Cargando campo de corrientes…")
def _cargar(ruta: str) -> object:
    return cargar_corrientes(ruta)


if archivo is not None:
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(archivo.read())
        ruta_nc = tmp.name
    campo = _cargar(ruta_nc)
elif _NC_DEFAULT.exists():
    campo = _cargar(str(_NC_DEFAULT))
    st.sidebar.info(f"Usando dataset por defecto: `{_NC_DEFAULT.name}`")
else:
    st.warning("Sube un archivo NetCDF para comenzar.")
    st.stop()

with st.expander("Resumen del dataset", expanded=False):
    st.text(resumen(campo))

# ---------------------------------------------------------------------------
# Botón de cálculo
# ---------------------------------------------------------------------------
if not st.button("🚀 Calcular ruta óptima", type="primary"):
    st.info("Configurá los parámetros en la barra lateral y presioná **Calcular ruta óptima**.")
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
capa = _CAPA_SUPERFICIE
lat_media = math.radians(float(campo.lat.mean()))
dy = abs(float(campo.lat[1] - campo.lat[0])) * 111_320.0
dx = abs(float(campo.lon[1] - campo.lon[0])) * 111_320.0 * math.cos(lat_media)

uo  = campo.uo[capa]
vo  = campo.vo[capa]
nav = campo.navegable[capa]

with st.spinner("Calculando divergencia y seleccionando waypoints…"):
    div   = divergencia(uo, vo, dx, dy)
    wps   = seleccionar_waypoints(div, nav, k_zonas, capa=capa, dist_min_celdas=dist_min)
    cent  = seleccionar_centinelas(uo, vo, nav, campo.lon, n=k_centinelas, capa=capa)
    base  = celda_mas_cercana(_LAT_CALLAO, _LON_CALLAO, campo.lat, campo.lon, nav, capa=capa)
    todos, base_nodo = agregar_puntos_fijos(wps + cent, wps[0], base)
    base_idx = todos.index(base_nodo)

n_intermedios = len(todos) - 1
st.info(
    f"**{len(wps)}** zonas de convergencia · "
    f"**{len(cent)}** centinelas offshore · "
    f"base en Callao · "
    f"**{math.factorial(n_intermedios):,}** órdenes ATSP a evaluar"
)

@st.cache_resource(show_spinner="Construyendo grafo…")
def _construir_grafo(campo, s, eta, e_max_val, k_zonas_val):
    p = ParametrosModelo(s=s, eta=eta, k_zonas=k_zonas_val, e_max=e_max_val)
    return construir_grafo(campo, p)

@st.cache_data(show_spinner="Calculando rutas con Bellman-Ford…")
def _matriz(_grafo_id, waypoints_tuple):
    return matriz_costos(grafo, list(waypoints_tuple))

grafo = _construir_grafo(campo, s, eta, float(e_max), k_zonas)
st.caption(f"Grafo: {grafo.num_nodos} nodos · {grafo.num_aristas} aristas")

M, caminos = _matriz(id(grafo), tuple(todos))

with st.spinner("Resolviendo ATSP por fuerza bruta…"):
    orden, costo_total = atsp_fuerza_bruta(M, base=base_idx)
    ruta = ensamblar_ruta(orden, caminos)

# Batería
bat = estado_bateria(ruta, grafo, float(e_max), bateria_inicial_j)

# ---------------------------------------------------------------------------
# Resultados numéricos (RF-08)
# ---------------------------------------------------------------------------
if bat["viable"]:
    st.success(f"✅ Ruta calculada — **{len(ruta)} nodos** · costo total **{costo_total:,.1f} J**")
else:
    st.error(
        f"⚠️ Ruta calculada pero la batería se agota durante la misión — "
        f"nivel mínimo: **{bat['minimo']:.1f} J**. "
        f"Aumentá la capacidad o reducí los waypoints."
    )

st.subheader("Resultados de la misión")
st.text(resumen_mision(orden, todos, M, campo))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Energía total", f"{costo_total:,.0f} J")
col2.metric("Consumido (propulsión)", f"{bat['consumido']:,.0f} J")
col3.metric("Recuperado (regeneración)", f"{bat['regenerado']:,.0f} J")
col4.metric("Batería mínima alcanzada",
            f"{bat['minimo']:,.0f} J",
            f"{bat['minimo']/float(e_max)*100:.1f} % de carga",
            delta_color="inverse")

# ---------------------------------------------------------------------------
# Visualizaciones (RF-07)
# ---------------------------------------------------------------------------
st.subheader("Visualizaciones")

tab1, tab2, tab3, tab4 = st.tabs([
    "Zonas y divergencia", "Ruta 2D", "Ruta 3D por capas", "Batería"
])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    plot_zonas(campo, div, wps, centinelas=cent, base=base, capa=capa, ax=ax1)
    st.pyplot(fig1)
    st.download_button("💾 Descargar", _fig_a_bytes(fig1), "zonas.png", "image/png")
    plt.close(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    plot_ruta(campo, ruta, waypoints=wps, centinelas=cent, base=base, ax=ax2)
    st.pyplot(fig2)
    st.download_button("💾 Descargar", _fig_a_bytes(fig2), "ruta_2d.png", "image/png")
    plt.close(fig2)

with tab3:
    fig3 = plot_3d(campo, ruta, waypoints=wps, centinelas=cent, base=base)
    st.pyplot(fig3)
    st.download_button("💾 Descargar", _fig_a_bytes(fig3), "ruta_3d.png", "image/png")
    plt.close(fig3)

with tab4:
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    plot_bateria(campo, ruta, bat["niveles"], float(e_max),
                 waypoints=todos, orden=orden, ax=ax4)
    st.pyplot(fig4)
    st.download_button("💾 Descargar", _fig_a_bytes(fig4), "bateria.png", "image/png")
    plt.close(fig4)

    # Tabla por tramo
    st.markdown("**Energía por tramo**")
    filas = []
    for k, (i, j) in enumerate(zip(orden, orden[1:])):
        _, li, lji = todos[i]
        _, lj, ljj = todos[j]
        costo_tramo = float(M[i, j])
        filas.append({
            "Tramo": f"{k+1}",
            "Desde": f"Z{i} ({campo.lat[li]:.2f}°, {campo.lon[lji]:.2f}°)",
            "Hasta": f"Z{j} ({campo.lat[lj]:.2f}°, {campo.lon[ljj]:.2f}°)",
            "Costo [J]": f"{costo_tramo:,.1f}",
            "Régimen": "🔋 Regenera" if costo_tramo < 0 else "⚡ Propulsión",
        })
    st.table(filas)

# ---------------------------------------------------------------------------
# Exportar CSV (RF-09)
# ---------------------------------------------------------------------------
st.subheader("Exportar resultados")

with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
    exportar_csv(ruta, campo, tmp_csv.name)
    csv_bytes = pathlib.Path(tmp_csv.name).read_bytes()

st.download_button(
    "📥 Descargar ruta como CSV",
    data=csv_bytes,
    file_name="ruta_auv.csv",
    mime="text/csv",
)
