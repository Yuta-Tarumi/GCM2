"""Vertical σ-grid utilities and hydrostatic integration."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet


from functools import partial


@partial(jax.jit, static_argnums=(0,))
def sigma_levels(L: int):
    z_half = jnp.linspace(0.0, 120e3, L + 1)
    H_ref = 15_000.0
    sigma_half = jnp.exp(-z_half / H_ref)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    return sigma_full, sigma_half


@partial(jax.jit, static_argnums=(0,))
def level_altitudes(L: int):
    z_half = jnp.linspace(0.0, 120e3, L + 1)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return z_full, z_half


@partial(jax.jit, static_argnums=(3,))
def hydrostatic_geopotential(T: jnp.ndarray, ps: jnp.ndarray, sigma_half: jnp.ndarray, planet: Planet):
    """Integrate hydrostatic balance to obtain geopotential on full levels.

    Parameters
    ----------
    T: [L, nlat, nlon] temperature
    ps: [nlat, nlon] surface pressure
    sigma_half: [L+1] sigma at half levels
    """

    R = planet.R_gas
    g = planet.g
    L = T.shape[0]
    # pressure at half levels
    p_half = ps[None, :, :] * sigma_half[:, None, None]
    dp = jnp.diff(p_half, axis=0)
    Tv = T  # dry atmosphere
    # Simmons–Burridge coefficients approximated with midpoint rule
    dPhi = -R * Tv * jnp.log(p_half[1:, :, :] / p_half[:-1, :, :])
    Phi = jnp.cumsum(dPhi[::-1, :, :], axis=0)[::-1, :, :]
    return Phi


@jax.jit
def vertical_diffusion(
    field: jnp.ndarray,
    z_half: jnp.ndarray,
    kappa_v: float,
    kappa_scale: jnp.ndarray | None = None,
):
    """Simple vertical diffusion to couple neighbouring layers.

    Parameters
    ----------
    field: [L, nlat, nlon]
        Quantity defined on full levels.
    z_half: [L+1]
        Altitude of half levels (m) used to compute layer thickness.
    kappa_v: float
        Constant vertical diffusivity (m^2/s).
    kappa_scale: [L] or None
        Optional per-layer scaling of diffusivity. Values near zero damp
        vertical mixing, which is useful to prevent surface temperatures
        from diffusing unrealistically into the upper atmosphere.
    """

    dz_full = jnp.diff(z_half)
    # interface spacing uses the mean of neighboring layer thicknesses
    dz_interface = 0.5 * (dz_full[:-1] + dz_full[1:])

    kappa_layer = kappa_v
    if kappa_scale is not None:
        kappa_layer = kappa_layer * kappa_scale[:, None, None]
    kappa_interface = 0.5 * (kappa_layer[:-1, ...] + kappa_layer[1:, ...])

    grad = (field[1:, ...] - field[:-1, ...]) / dz_interface[:, None, None]
    flux = jnp.zeros((field.shape[0] + 1,) + field.shape[1:])
    flux = flux.at[1:-1, ...].set(-kappa_interface * grad)
    tendency = -(flux[1:, ...] - flux[:-1, ...]) / dz_full[:, None, None]
    return tendency
