"""Núcleo del planificador de rutas de mínima energía para un AUV.

Paquete modular (RNF-08). Las dependencias fluyen en una sola dirección:
datos -> grafo -> zonas -> algoritmos -> metricas, y visualizacion consume el
resto. Ningún módulo del núcleo depende de la interfaz (Streamlit).
"""
