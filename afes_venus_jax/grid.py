"""Gaussian grid utilities and quadrature helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass
class Grid:
    lats: jnp.ndarray
    lons: jnp.ndarray
    weights: jnp.ndarray
    lat2d: jnp.ndarray
    lon2d: jnp.ndarray


def gaussian_grid(nlat: int, nlon: int) -> Grid:
    """Construct a linear Gaussian grid.

    Parameters
    ----------
    nlat, nlon: int
        Number of Gaussian latitudes and longitudes.
    """

    x, w = leggauss(nlat)
    lats = jnp.arcsin(jnp.array(x))
    lons = jnp.linspace(0.0, 2 * jnp.pi, nlon, endpoint=False)
    lon2d, lat2d = jnp.meshgrid(lons, lats, indexing="xy")
    return Grid(lats=lats, lons=lons, weights=jnp.array(w), lat2d=lat2d, lon2d=lon2d)
