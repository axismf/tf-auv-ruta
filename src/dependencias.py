"""Dependencias centralizadas del paquete src.

Cada módulo importa desde aquí lo que necesita. Así las librerías
se listan una sola vez para todo el programa.
"""
from __future__ import annotations

# ── Stdlib ────────────────────────────────────────────────────────────────────
import csv
import itertools
import math
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Literal

# ── Científicos ───────────────────────────────────────────────────────────────
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import plotly.graph_objects as go

GRADOS_A_METROS: float = 111_320.0

__all__ = [
    "csv", "itertools", "math", "pathlib",
    "ABC", "abstractmethod", "dataclass", "Iterator", "Literal",
    "mcolors", "mpatches", "plt", "np", "xr", "go",
    "GRADOS_A_METROS",
]
