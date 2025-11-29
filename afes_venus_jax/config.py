"""Planetary constants and numerical defaults for the Venus spectral core."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Planet:
    """Physical constants for Venus (SI units)."""

    a: float = 6_051_800.0
    g: float = 8.87
    Omega: float = -2 * jnp.pi / (243.0226 * 86400.0)
    Ru: float = 8.314462618
    M_CO2: float = 44.01e-3
    cp: float = 1000.0
    ps_ref: float = 9.2e6
    solar_day: float = 116.75 * 86400.0

    @property
    def R_gas(self) -> float:
        return self.Ru / self.M_CO2


@dataclass
class Numerics:
    """Discretization and time-stepping parameters."""

    Lmax: int = 42
    nlat: int = 64
    nlon: int = 128
    L: int = 60
    dt: float = 600.0
    alpha: float = 0.5
    ra: float = 0.05
    tau_hdiff: float = 0.1 * 86400.0
    order_hdiff: int = 4
    sponge_top_k: int = 10
    tau_rad_surface_days: float = 30.0
    tau_rad_cloud_days: float = 10.0
    tau_rad_upper_days: float = 5.0
    diurnal_phase0: float = 0.0


# Enable x64 globally
jax.config.update("jax_enable_x64", True)


def default_planet() -> Planet:
    return Planet()


def default_numerics() -> Numerics:
    return Numerics()
