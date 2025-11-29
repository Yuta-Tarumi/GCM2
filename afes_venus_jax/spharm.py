"""Simplified spherical-harmonic utilities using FFT-based spectral space.

This module mimics the interface of an S2FFT-backed transform but internally uses
2-D FFTs for testing convenience. The spectral representation is complex-valued
with shape ``(nlat, nlon//2+1)`` matching ``rfft2`` output.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def analysis_grid_to_spec(field_grid: jnp.ndarray, Lmax: int | None = None) -> jnp.ndarray:
    return jnp.fft.rfft2(field_grid)


@jax.jit
def synthesis_spec_to_grid(flm: jnp.ndarray, nlat: int, nlon: int) -> jnp.ndarray:
    return jnp.fft.irfft2(flm, s=(nlat, nlon))


def _wavenumbers(nlat: int, nlon: int, a: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    kx = jnp.fft.fftfreq(nlon) * 2 * jnp.pi / a
    ky = jnp.fft.fftfreq(nlat) * 2 * jnp.pi / a
    return kx, ky


@jax.jit
def lap_spec(flm: jnp.ndarray, nlat: int, nlon: int, a: float) -> jnp.ndarray:
    kx = jnp.fft.fftfreq(nlon) * 2 * jnp.pi / a
    ky = jnp.fft.fftfreq(nlat) * 2 * jnp.pi / a
    kx2 = kx ** 2
    ky2 = ky ** 2
    ky_grid = ky[:, None]
    kx_grid = kx[None, : flm.shape[1]]
    return -((ky_grid ** 2 + kx_grid ** 2)) * flm


@jax.jit
def invert_laplacian(flm: jnp.ndarray, nlat: int, nlon: int, a: float) -> jnp.ndarray:
    kx = jnp.fft.fftfreq(nlon) * 2 * jnp.pi / a
    ky = jnp.fft.fftfreq(nlat) * 2 * jnp.pi / a
    ky_grid = ky[:, None]
    kx_grid = kx[None, : flm.shape[1]]
    denom = ky_grid ** 2 + kx_grid ** 2
    denom = jnp.where(denom == 0, jnp.inf, denom)
    return flm / (-denom)


@jax.jit
def uv_from_psi_chi(psi_hat: jnp.ndarray, chi_hat: jnp.ndarray, nlat: int, nlon: int, a: float):
    kx = jnp.fft.fftfreq(nlon) * 2 * jnp.pi / a
    ky = jnp.fft.fftfreq(nlat) * 2 * jnp.pi / a
    ikx = 1j * kx[None, : psi_hat.shape[1]]
    iky = 1j * ky[:, None]
    u_hat = iky * psi_hat + ikx * chi_hat
    v_hat = -ikx * psi_hat + iky * chi_hat
    u = jnp.fft.irfft2(u_hat, s=(nlat, nlon))
    v = jnp.fft.irfft2(v_hat, s=(nlat, nlon))
    return u, v


def vmap_levels(fn, axis=0):
    return jax.vmap(fn, in_axes=(axis,), out_axes=axis)
