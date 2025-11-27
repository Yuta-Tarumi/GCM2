"""Configuration and planetary constants for the AFES-Venus JAX core."""
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class Config:
    """Model configuration parameters.

    Attributes
    ----------
    a : float
        Planetary radius [m].
    g : float
        Gravity [m s^-2].
    Omega : float
        Planetary rotation rate [s^-1].
    R_gas : float
        Specific gas constant [J kg^-1 K^-1].
    cp : float
        Specific heat at constant pressure [J kg^-1 K^-1].
    ps_ref : float
        Reference surface pressure [Pa].
    Lmax : int
        Spectral truncation (triangular T Lmax).
    nlat, nlon : int
        Gaussian grid dimensions.
    L : int
        Number of full levels (Lorenz grid uses L+1 half levels).
    dt : float
        Time step [s].
    alpha : float
        Semi-implicit off-centring coefficient.
    ra : float
        Robert-Asselin filter coefficient.
    tau_hdiff : float
        Hyperdiffusion e-folding time at maximum wavenumber [s].
    order_hdiff : int
        Hyperdiffusion order (4 by default).
    """

    a: float = 6_051_800.0
    g: float = 8.87
    Omega: float = -2 * jnp.pi / (243.0226 * 86400.0)
    R_gas: float = 8.314462618 / 0.04401
    cp: float = 1000.0
    ps_ref: float = 9.2e6

    Lmax: int = 42
    nlat: int = 64
    nlon: int = 128
    L: int = 60
    dt: float = 600.0
    alpha: float = 0.5
    ra: float = 0.05
    tau_hdiff: float = 0.1 * 86400.0
    order_hdiff: int = 4


# Enable 64-bit globally.
jax.config.update("jax_enable_x64", True)


def default_config() -> Config:
    """Return the default Venus configuration."""

    return Config()


def fast_config() -> Config:
    """Return a lightweight configuration for quick unit tests.

    The reduced resolution avoids long JIT compilation while preserving
    code paths for spectral transforms and time stepping.
    """

    return Config(Lmax=4, nlat=8, nlon=16, L=4, dt=600.0)
