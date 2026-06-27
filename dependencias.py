"""Dependencias centralizadas de la aplicación web (app.py)."""
from __future__ import annotations

# ── Stdlib ────────────────────────────────────────────────────────────────────
import base64
import csv
import io
import math
import pathlib
import tempfile

# ── Científicos / UI ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

GRADOS_A_METROS: float = 111_320.0

__all__ = [
    "base64", "csv", "io", "math", "pathlib", "tempfile",
    "plt", "np", "pd", "st", "xr",
    "GRADOS_A_METROS",
]
