"""Lightweight xarray I/O for diagnostics."""
from __future__ import annotations

import xarray as xr
import jax.numpy as jnp

from .config import Numerics
from .grid import gaussian_grid
from .spharm import synthesis_spec_to_grid


def write_snapshot(state, path: str, num: Numerics):
    grid = gaussian_grid(num.nlat, num.nlon)
    zeta = synthesis_spec_to_grid(state.zeta, num)
    div = synthesis_spec_to_grid(state.div, num)
    T = synthesis_spec_to_grid(state.T, num)
    lnps = synthesis_spec_to_grid(state.lnps, num)

    ds = xr.Dataset(
        {
            "zeta": ("lat", "lon", jnp.real(zeta)),
            "div": ("lat", "lon", jnp.real(div)),
            "T": ("lev", "lat", "lon", jnp.real(T)),
            "lnps": ("lat", "lon", jnp.real(lnps)),
        },
        coords={"lat": grid.lats, "lon": grid.lons, "lev": jnp.arange(num.L)},
    )
    ds.to_netcdf(path)
