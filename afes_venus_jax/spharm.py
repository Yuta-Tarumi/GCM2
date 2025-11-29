"""Spectral transform utilities using FFT surrogates.

For tractability in this compact implementation we use doubly periodic FFT
operators that remain JAX-friendly while preserving the spectral interface
(e.g., Laplacian eigenproperties and inversion). The functions are vectorised
so that vertical levels can be handled by ``vmap`` in callers.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Numerics, Planet


def _wavenumbers(nlat: int, nlon: int, a: float):
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nlon) / a
    ky = 2 * jnp.pi * jnp.fft.fftfreq(nlat) / a
    return kx, ky


def analysis_grid_to_spec(field_grid: jnp.ndarray, num: Numerics) -> jnp.ndarray:
    """Forward transform grid → complex spectral coefficients."""

    return jnp.fft.fft2(field_grid) / (num.nlat * num.nlon)


def synthesis_spec_to_grid(flm: jnp.ndarray, num: Numerics) -> jnp.ndarray:
    """Inverse transform spectral → grid values (real)."""

    grid = jnp.fft.ifft2(flm * (num.nlat * num.nlon))
    return grid


def laplace_fac(flm: jnp.ndarray, num: Numerics, planet: Planet) -> jnp.ndarray:
    """Return Laplacian operator applied in spectral space."""

    kx, ky = _wavenumbers(num.nlat, num.nlon, planet.a)
    fac = -(ky[:, None] ** 2 + kx[None, :] ** 2)
    return flm * fac


def invert_laplacian(flm: jnp.ndarray, num: Numerics, planet: Planet) -> jnp.ndarray:
    kx, ky = _wavenumbers(num.nlat, num.nlon, planet.a)
    denom = ky[:, None] ** 2 + kx[None, :] ** 2
    safe = jnp.where(denom == 0, 1e-12, denom)
    return -flm / safe


def psi_chi_from_zeta_div(zeta_lm: jnp.ndarray, div_lm: jnp.ndarray, num: Numerics, planet: Planet):
    zeta_lm = zeta_lm.at[0, 0].set(0.0)
    div_lm = div_lm.at[0, 0].set(0.0)
    psi = invert_laplacian(zeta_lm, num, planet)
    chi = invert_laplacian(div_lm, num, planet)
    psi = psi.at[0, 0].set(0.0)
    chi = chi.at[0, 0].set(0.0)
    return psi, chi


def uv_from_psi_chi(psi_lm: jnp.ndarray, chi_lm: jnp.ndarray, num: Numerics, planet: Planet):
    """Compute horizontal winds from streamfunction and velocity potential."""

    kx, ky = _wavenumbers(num.nlat, num.nlon, planet.a)
    ikx = 1j * kx[None, :]
    iky = 1j * ky[:, None]
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    k2_safe = jnp.where(k2 == 0, 1e-12, k2)

    zeta_hat = -(k2_safe) * psi_lm
    div_hat = -(k2_safe) * chi_lm

    u_hat = -iky * psi_lm + ikx * chi_lm
    v_hat = ikx * psi_lm + iky * chi_lm

    u = synthesis_spec_to_grid(u_hat, num)
    v = synthesis_spec_to_grid(v_hat, num)
    return u, v
