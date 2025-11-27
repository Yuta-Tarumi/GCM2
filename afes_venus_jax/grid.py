"""Gaussian grid utilities for the pseudo-spectral core."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss


def gaussian_grid(nlat: int, nlon: int):
    """Construct Gaussian grid.

    Parameters
    ----------
    nlat, nlon : int
        Number of latitude and longitude points. ``nlat`` should match the
        number of Gaussian quadrature points for the desired truncation.

    Returns
    -------
    lats : jnp.ndarray
        Latitudes in radians, shape (nlat,).
    lons : jnp.ndarray
        Longitudes in radians, shape (nlon,).
    weights : jnp.ndarray
        Gaussian quadrature weights, shape (nlat,).
    """

    x, w = leggauss(nlat)
    lats = jnp.arcsin(jnp.array(x))
    lons = jnp.linspace(0.0, 2 * jnp.pi, nlon, endpoint=False)
    return lats, lons, jnp.array(w)


def expand_grid(lats: jnp.ndarray, lons: jnp.ndarray):
    """Return 2D meshgrids of longitude and latitude."""

    lon2d, lat2d = jnp.meshgrid(lons, lats, indexing="xy")
    return lat2d, lon2d
