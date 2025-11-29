"""Gaussian grid utilities and quadrature helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss


def _gaussian_latitudes(nlat: int):
    x, w = leggauss(nlat)
    lats = np.arcsin(x)
    return lats[::-1], w[::-1]


@jax.tree_util.register_pytree_node_class
@dataclass
class Grid:
    lats: jnp.ndarray
    lons: jnp.ndarray
    weights: jnp.ndarray
    lat2d: jnp.ndarray
    lon2d: jnp.ndarray

    def tree_flatten(self):
        children = (self.lats, self.lons, self.weights, self.lat2d, self.lon2d)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        lats, lons, weights, lat2d, lon2d = children
        return cls(lats=lats, lons=lons, weights=weights, lat2d=lat2d, lon2d=lon2d)


def gaussian_grid(nlat: int, nlon: int) -> Grid:
    """Construct a linear Gaussian grid with quadrature weights."""

    lats, w = _gaussian_latitudes(nlat)
    lats = jnp.array(lats)
    w = jnp.array(w)
    lons = jnp.linspace(0.0, 2 * jnp.pi, nlon, endpoint=False)
    lon2d, lat2d = jnp.meshgrid(lons, lats, indexing="xy")
    return Grid(lats=lats, lons=lons, weights=w, lat2d=lat2d, lon2d=lon2d)


def cos_sin_lat_lon(grid: Grid) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return jnp.cos(grid.lat2d), jnp.sin(grid.lat2d), jnp.cos(grid.lon2d), jnp.sin(grid.lon2d)


def area_element(grid: Grid) -> jnp.ndarray:
    return grid.weights[:, None] * jnp.ones((grid.lats.shape[0], grid.lons.shape[0]))
