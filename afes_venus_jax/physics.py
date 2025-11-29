"""Simple diurnal shortwave heating and Newtonian cooling."""
from __future__ import annotations

import jax.numpy as jnp

from .config import Planet, Numerics
from .vertical import sigma_levels, reference_temperature_profile


def diurnal_heating(lat2d, lon2d, z_full, t: float, planet: Planet, num: Numerics):
    lambda_s = num.diurnal_phase0 + 2 * jnp.pi * t / planet.solar_day
    hour_angle = lon2d - lambda_s
    coszen = jnp.maximum(0.0, jnp.cos(hour_angle)) * jnp.maximum(0.0, jnp.cos(lat2d))
    peak = 2.0 / 86400.0  # 2 K/day
    center = 70_000.0
    width = 15_000.0
    A = peak * jnp.exp(-((z_full - center) ** 2) / (2 * width ** 2))
    return A[:, None, None] * coszen[None, :, :]


def newtonian_cooling(T_grid: jnp.ndarray, planet: Planet, num: Numerics):
    z_full, T_eq = reference_temperature_profile(num)
    sigma_full, _, _ = sigma_levels(num)
    # piecewise tau profile by height
    tau = jnp.where(z_full < 50_000.0, num.tau_rad_surface_days * 86400.0, num.tau_rad_cloud_days * 86400.0)
    tau = jnp.where(z_full > 90_000.0, num.tau_rad_upper_days * 86400.0, tau)
    T_eq_grid = T_eq[:, None, None]
    tendency = (T_eq_grid - T_grid) / tau[:, None, None]
    return tendency
