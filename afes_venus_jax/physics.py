"""Diurnal solar heating and Newtonian cooling for Venus."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics
from afes_venus_jax.vertical import level_altitudes


def subsolar_longitude(t: float, planet: Planet) -> float:
    return planet.diurnal_phase0 if hasattr(planet, "diurnal_phase0") else 0.0 + 2 * jnp.pi * t / planet.solar_day


@jax.jit
def diurnal_heating(T: jnp.ndarray, grid, t: float, planet: Planet, num: Numerics):
    z_full, _ = level_altitudes(num.L)
    lam_sun = 2 * jnp.pi * t / planet.solar_day + num.diurnal_phase0
    mu = jnp.maximum(0.0, jnp.cos(grid.lat2d) * jnp.cos(grid.lon2d - lam_sun))
    A = jnp.exp(-((z_full[:, None, None] - 60e3) / 12e3) ** 2) * 2e-4
    return A * mu[None, :, :]


@jax.jit
def newtonian_cooling(T: jnp.ndarray, num: Numerics):
    z_full, _ = level_altitudes(num.L)
    Teq = 730.0 - (730.0 - 170.0) * (z_full / z_full[-1])
    tau = jnp.where(z_full < 30e3, num.tau_rad_surface_days * 86400.0,
                    jnp.where(z_full < 70e3, num.tau_rad_cloud_days * 86400.0,
                               num.tau_rad_upper_days * 86400.0))
    return (Teq[:, None, None] - T) / tau[:, None, None]
