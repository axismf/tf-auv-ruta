# Planificador de rutas de mínima energía para un AUV

Calcula la ruta de reconocimiento de menor consumo energético para un Vehículo
Submarino Autónomo (AUV) en el litoral de Lima-Callao, aprovechando las
corrientes marinas. El mar se modela como un grafo dirigido y asimétrico donde
el peso de cada arista es la energía neta de desplazamiento; las rutas de
mínima energía entre zonas se calculan con **Bellman-Ford** y el orden óptimo
de visita con un **ATSP por enumeración exacta**.

## Estructura

```
tf-auv-ruta/
├── data/                 dataset NetCDF de corrientes (CMEMS)
├── src/                  núcleo modular
│   ├── config.py         parámetros del modelo (dataclass inmutable)
│   ├── datos.py          carga del NetCDF y máscara de tierra      (RF-01, RF-02)
│   ├── grafo.py          grafo dirigido + función de costo         (RF-03, RF-04)
│   ├── zonas.py          divergencia y selección de waypoints      (RF-05 a RF-07)
│   ├── algoritmos.py     Bellman-Ford, matriz de costos y ATSP     (RF-08, RF-09)
│   ├── metricas.py       energía total, costos y exportación       (RF-12, RF-13)
│   └── visualizacion.py  plots de matplotlib                       (RF-11)
├── app.py                interfaz Streamlit                        (RF-10, RF-11)
├── notebooks/            pruebas visuales durante el desarrollo
├── outputs/              figuras y rutas exportadas
└── tests/                pruebas unitarias
```

Cada módulo traza a un grupo de requisitos, evidencia del diseño modular
(RNF-08). Las dependencias fluyen en una sola dirección: el núcleo nunca
depende de la interfaz.

## Orden de desarrollo recomendado

1. `datos.py` — cargar el `.nc`, verificar con un quiver del campo.
2. `grafo.py` — construir nodos/aristas y la función de costo.
3. `zonas.py` — divergencia y waypoints, verificar sobre el mapa.
4. `algoritmos.py` — Bellman-Ford + matriz + ATSP, dibujar la ruta.
5. `metricas.py` — energía total, costos por tramo, exportar CSV.
6. `app.py` — envolver todo en Streamlit (al final).

## Instalación y ejecución

```bash
pip install -r requirements.txt
streamlit run app.py        # interfaz web (cuando el core esté listo)
pytest                      # pruebas
```
