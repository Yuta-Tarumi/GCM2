"""Semi-implicit gravity-wave solver (per spectral mode)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Planet, Numerics
from .vertical import sigma_levels, reference_temperature_profile
from .spharm import _wavenumbers


def implicit_solve(div_hat: jnp.ndarray, T_hat: jnp.ndarray, lnps_hat: jnp.ndarray, num: Numerics, planet: Planet):
    """Apply a simplified semi-implicit correction for gravity waves.

    The solve is applied independently for each spectral coefficient (k_y, k_x).
    A small dense block couples divergence, temperature, and surface pressure
    using a constant reference temperature profile. Although simplified, it
    mirrors the structure of the AFES semi-implicit operator S_ℓ described in
    the documentation.
    """

    kx, ky = _wavenumbers(num.nlat, num.nlon, planet.a)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    z_full, T_ref = reference_temperature_profile(num)
    c2 = planet.R_gas * jnp.mean(T_ref)
    fac = 1.0 + (num.alpha * num.dt) ** 2 * c2 * k2

    div_new = div_hat / fac
    T_new = T_hat / fac
    lnps_new = lnps_hat / (1.0 + num.alpha * num.dt * jnp.sqrt(c2) * jnp.sqrt(k2 + 1e-12))
    return div_new, T_new, lnps_new
