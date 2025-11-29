"""Sigma-coordinate vertical utilities and hydrostatic geopotential."""
from __future__ import annotations

import jax.numpy as jnp

from .config import Numerics, Planet


def sigma_levels(num: Numerics):
    z_half = jnp.linspace(0.0, 120_000.0, num.L + 1)
    H_ref = 15_000.0
    sigma_half = jnp.exp(-z_half / H_ref)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    return sigma_full, sigma_half, z_half


def reference_temperature_profile(num: Numerics):
    _, _, z_half = sigma_levels(num)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return z_full, jnp.linspace(730.0, 170.0, num.L)


def hydrostatic_geopotential(T: jnp.ndarray, lnps: jnp.ndarray, planet: Planet, num: Numerics) -> jnp.ndarray:
    """Integrate hydrostatic balance to obtain geopotential on full levels."""

    sigma_full, sigma_half, _ = sigma_levels(num)
    ps = jnp.exp(lnps)
    p_full = sigma_full[:, None, None] * ps[None, :, :]
    p_half = sigma_half[:, None, None] * ps[None, :, :]

    # Simmons–Burridge style coefficients
    delta_sigma = jnp.diff(sigma_half)
    ak = delta_sigma[:, None, None]
    bk = jnp.log(p_half[1:, ...] / p_half[:-1, ...])
    # integrate downward
    phi = jnp.zeros((num.L, ps.shape[0], ps.shape[1]))
    running = jnp.zeros_like(ps)
    for k in range(num.L - 1, -1, -1):
        lapse = planet.R_gas * T[k]
        running = running + lapse * bk[k]
        phi = phi.at[k].set(running)
    return phi


def temperature_to_sigma_profile(Tz_func, num: Numerics):
    _, _, z_half = sigma_levels(num)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return z_full, Tz_func(z_full)
