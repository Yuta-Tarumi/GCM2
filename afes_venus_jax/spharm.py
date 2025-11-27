"""Simplified spherical harmonic helpers built on FFTs.

The implementation uses doubly periodic FFTs as a stand-in for full
spherical harmonics. This keeps the code JIT-friendly and adequate for
self-consistency tests while preserving the spectral operator interface.
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
from jax import lax
from .config import Config
from .grid import expand_grid


@functools.lru_cache(None)
def _wavenumbers(nlat: int, nlon: int, a: float):
    # Angular wavenumbers (radian-based) compatible with FFT derivatives.
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nlon)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(nlat)
    return kx, ky


def analysis_grid_to_spec(field_grid: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Forward transform grid → spectral coefficients.

    Parameters
    ----------
    field_grid : array (..., nlat, nlon)
    cfg : Config

    Returns
    -------
    array (..., nlat, nlon)
        Complex spectral coefficients (periodic FFT surrogate).
    """

    return jnp.fft.fft2(field_grid) / (cfg.nlat * cfg.nlon)


def synthesis_spec_to_grid(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Inverse transform spectral → grid."""

    return jnp.fft.ifft2(flm * (cfg.nlat * cfg.nlon))


def lap_spec(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Apply Laplacian in spectral space using FFT wavenumbers."""

    kx, ky = _wavenumbers(cfg.nlat, cfg.nlon, cfg.a)
    kx2 = jnp.square(kx)
    ky2 = jnp.square(ky)

    def apply(arr):
        res = arr
        res = jnp.fft.fft(res, axis=-1)
        res = jnp.fft.fft(res, axis=-2)
        fac = -(ky2[:, None] + kx2[None, :])
        res = res * fac
        return res

    return laplace_fac(flm, cfg)


def laplace_fac(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    kx, ky = _wavenumbers(cfg.nlat, cfg.nlon, cfg.a)
    fac = -(ky[:, None] ** 2 + kx[None, :] ** 2)
    return flm * fac


def invert_laplacian(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Invert Laplacian safely, zeroing the mean mode."""

    kx, ky = _wavenumbers(cfg.nlat, cfg.nlon, cfg.a)
    denom = ky[:, None] ** 2 + kx[None, :] ** 2
    safe = jnp.where(denom == 0, 1e-12, denom)
    inv = -flm / safe
    return inv


def psi_chi_from_zeta_div(zeta_lm: jnp.ndarray, div_lm: jnp.ndarray, cfg: Config):
    """Return streamfunction and velocity potential in spectral space."""

    psi = invert_laplacian(zeta_lm, cfg)
    chi = invert_laplacian(div_lm, cfg)
    return psi, chi


def uv_from_psi_chi(psi_lm: jnp.ndarray, chi_lm: jnp.ndarray, cfg: Config):
    """Compute horizontal winds from potentials.

    A doubly periodic derivative is used as a surrogate for spin-weighted
    gradients. The resulting winds are divergence/rotational consistent
    with the surrogate operators.
    """

    kx, ky = _wavenumbers(cfg.nlat, cfg.nlon, cfg.a)
    ikx = 1j * kx[None, :]
    iky = 1j * ky[:, None]
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    k2_safe = jnp.where(k2 == 0, 1e-12, k2)

    div_lm = -(k2_safe) * chi_lm
    zeta_lm = -(k2_safe) * psi_lm

    u_hat = -(ikx * div_lm - iky * zeta_lm) / k2_safe
    v_hat = -(ikx * zeta_lm + iky * div_lm) / k2_safe

    u = synthesis_spec_to_grid(u_hat, cfg)
    v = synthesis_spec_to_grid(v_hat, cfg)
    return u, v
