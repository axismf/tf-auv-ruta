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
from src.algoritmos import (
    matriz_costos,
    atsp_fuerza_bruta,
    ensamblar_ruta,
    bellman_ford,
    hay_ciclo_negativo,
)
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


def _mostrar_figura(fig: plt.Figure, modo: str = "medio") -> None:
    """Renderiza una figura centrada y con ancho contenido.

    Evita que las gráficas ocupen todo el ancho de la página (layout wide),
    dándoles un tamaño más sobrio. No altera la figura, solo su contenedor.
    """
    ratios = {
        "compacto": (1, 2, 1),    # ~50 % del ancho útil
        "medio": (1, 3, 1),       # ~60 %
        "panoramico": (1, 8, 1),  # ~80 %, para la gráfica ancha de batería
    }.get(modo, (1, 3, 1))
    _izq, centro, _der = st.columns(ratios)
    with centro:
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Planificador AUV — Lima",
    page_icon="🌊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estilo — identidad visual coherente (complementa .streamlit/config.toml)
# ---------------------------------------------------------------------------
_ESTILOS = """
<style>
/* Botones: forma, peso y espaciado consistentes.
   El COLOR lo aporta el tema (config.toml), por lo que se adapta solo
   al modo claro u oscuro sin sobrescribirlo aquí. */
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

/* Hero del encabezado — hereda el color de texto del tema activo */
.auv-hero {
    border-left: 4px solid #2E8B9E;  /* teal intermedio, legible en ambos modos */
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
    max-width: 62ch;
    opacity: 0.72;  /* matiza el color de texto heredado, sin fijarlo */
}

/* Encabezados de la barra lateral con jerarquía sutil */
section[data-testid="stSidebar"] h2 {
    font-size: 1.0rem;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
</style>
"""
st.markdown(_ESTILOS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="auv-hero">
        <h1>🌊 Planificador de rutas AUV · Lima</h1>
        <div class="sub">
            Grafo dirigido sobre el litoral de Lima para planificar misiones de
            reconocimiento con mínimo consumo energético, aprovechando las
            corrientes marinas. Integra zonas de convergencia de contaminantes
            y puntos centinela offshore.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra lateral — parámetros del modelo (RF-06)
# ---------------------------------------------------------------------------
st.sidebar.header("Modelo energético")

s = st.sidebar.slider(
    "Velocidad de crucero [m/s]", 0.1, 1.5, 0.5, 0.05,
    help="Velocidad del AUV respecto al agua.",
)
eta = st.sidebar.slider(
    "Eficiencia de regeneración η", 0.05, 0.95, 0.30, 0.05,
    help="Fracción de energía recuperada en modo regeneración.",
)
k_p = st.sidebar.slider(
    "Coeficiente de propulsión kp", 0.1, 5.0, 1.0, 0.1,
    help="Escala el coste energético en modo propulsión.",
)
k_r = st.sidebar.slider(
    "Coeficiente de regeneración kr", 0.1, 5.0, 1.0, 0.1,
    help="Escala la energía recuperada en modo regeneración. "
         "Para evitar ciclos negativos, kr·η no debe superar kp.",
)
k_zonas = st.sidebar.slider(
    "Zonas de convergencia", 2, 8, 6,
    help="Número de zonas de acumulación de contaminantes a visitar.",
)
k_centinelas = st.sidebar.slider(
    "Centinelas offshore", 1, 4, 2,
    help="Puntos de detección temprana en la franja oceánica abierta.",
)
dist_min = st.sidebar.slider(
    "Separación mínima entre waypoints [celdas]", 1, 6, 3,
    help="Distancia mínima para evitar seleccionar celdas de la misma zona.",
)

st.sidebar.header("Batería")
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

params = ParametrosModelo(s=s, eta=eta, k_p=k_p, k_r=k_r, k_zonas=k_zonas, e_max=float(e_max))

# ---------------------------------------------------------------------------
# Carga del dataset (RF-01)
# ---------------------------------------------------------------------------
st.sidebar.header("Dataset")
archivo = st.sidebar.file_uploader(
    "NetCDF de Copernicus Marine (.nc)",
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
if not st.button("Calcular ruta óptima", type="primary"):
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
def _construir_grafo(campo, s, eta, k_p_val, k_r_val, e_max_val, k_zonas_val):
    p = ParametrosModelo(s=s, eta=eta, k_p=k_p_val, k_r=k_r_val,
                         k_zonas=k_zonas_val, e_max=e_max_val)
    return construir_grafo(campo, p)

@st.cache_data(show_spinner="Calculando rutas con Bellman-Ford…")
def _matriz(_grafo_id, waypoints_tuple):
    return matriz_costos(grafo, list(waypoints_tuple))

grafo = _construir_grafo(campo, s, eta, k_p, k_r, float(e_max), k_zonas)
st.caption(f"Grafo: {grafo.num_nodos} nodos · {grafo.num_aristas} aristas")

# Validación del modelo: detección de ciclos negativos (RF-08, RNF-07).
with st.spinner("Verificando ausencia de ciclos negativos…"):
    _dist_val, _ = bellman_ford(grafo, base_nodo)
    _hay_ciclo = hay_ciclo_negativo(grafo, _dist_val)
if _hay_ciclo:
    st.warning(
        "⚠️ Se detectó un ciclo de energía negativa en el grafo. Indica una "
        "mala calibración de los parámetros: la regeneración (kr·η) supera al "
        "costo de propulsión (kp), de modo que el AUV ganaría energía dando "
        "vueltas. Los resultados podrían no ser fiables; reducí η o kr."
    )
else:
    st.caption("✓ Sin ciclos negativos — modelo bien calibrado.")

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
    _mostrar_figura(fig1, "medio")
    st.download_button("Descargar PNG", _fig_a_bytes(fig1), "zonas.png",
                       "image/png", key="dl_zonas")
    plt.close(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    plot_ruta(campo, ruta, waypoints=wps, centinelas=cent, base=base, ax=ax2)
    _mostrar_figura(fig2, "medio")
    st.download_button("Descargar PNG", _fig_a_bytes(fig2), "ruta_2d.png",
                       "image/png", key="dl_ruta2d")
    plt.close(fig2)

with tab3:
    fig3 = plot_3d(campo, ruta, waypoints=wps, centinelas=cent, base=base)
    _mostrar_figura(fig3, "medio")
    st.download_button("Descargar PNG", _fig_a_bytes(fig3), "ruta_3d.png",
                       "image/png", key="dl_ruta3d")
    plt.close(fig3)

with tab4:
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    plot_bateria(campo, ruta, bat["niveles"], float(e_max),
                 waypoints=todos, orden=orden, ax=ax4)
    _mostrar_figura(fig4, "panoramico")
    st.download_button("Descargar PNG", _fig_a_bytes(fig4), "bateria.png",
                       "image/png", key="dl_bateria")
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
    "Descargar ruta como CSV",
    data=csv_bytes,
    file_name="ruta_auv.csv",
    mime="text/csv",
    key="dl_csv",
)