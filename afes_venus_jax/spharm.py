"""Simplified spherical-harmonic utilities using FFT-based spectral space.

This module mimics the interface of an S2FFT-backed transform but internally uses
2-D FFTs for testing convenience. The spectral representation is complex-valued
with shape ``(nlat, nlon//2+1)`` matching ``rfft2`` output.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def _wavenumbers(nlat: int, nlon: int, a: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Physical wavenumbers using the actual grid spacing.

    The latitude grid spans π radians from pole to pole, while longitude spans
    2π. Using ``d`` in ``fftfreq`` ensures the resulting frequencies are scaled
    to the real-space spacing instead of the implicit unit grid. Without this
    scaling, all meridional derivatives were off by a factor proportional to the
    grid size (≈O(100) at T42), which let FFT wrap-around errors at the poles
    grow unchecked and contaminate the jet.
    """

    dlon = 2 * jnp.pi / nlon
    dlat = jnp.pi / nlat
    kx = jnp.fft.fftfreq(nlon, d=dlon) * 2 * jnp.pi / a
    ky = jnp.fft.fftfreq(nlat, d=dlat) * 2 * jnp.pi / a
    return kx, ky


@functools.partial(jax.jit, static_argnums=(1,))
def analysis_grid_to_spec(field_grid: jnp.ndarray, Lmax: int | None = None) -> jnp.ndarray:
    return jnp.fft.rfft2(field_grid)


@functools.partial(jax.jit, static_argnums=(1, 2))
def synthesis_spec_to_grid(flm: jnp.ndarray, nlat: int, nlon: int) -> jnp.ndarray:
    return jnp.fft.irfft2(flm, s=(nlat, nlon))


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def lap_spec(flm: jnp.ndarray, nlat: int, nlon: int, a: float) -> jnp.ndarray:
    kx, ky = _wavenumbers(nlat, nlon, a)
    ky_grid = ky[:, None]
    kx_grid = kx[None, : flm.shape[1]]
    return -((ky_grid ** 2 + kx_grid ** 2)) * flm


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def invert_laplacian(flm: jnp.ndarray, nlat: int, nlon: int, a: float) -> jnp.ndarray:
    kx, ky = _wavenumbers(nlat, nlon, a)
    ky_grid = ky[:, None]
    kx_grid = kx[None, : flm.shape[1]]
    denom = ky_grid ** 2 + kx_grid ** 2
    denom = jnp.where(denom == 0, jnp.inf, denom)
    return flm / (-denom)


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def uv_from_psi_chi(psi_hat: jnp.ndarray, chi_hat: jnp.ndarray, nlat: int, nlon: int, a: float):
    kx, ky = _wavenumbers(nlat, nlon, a)
    ikx = 1j * kx[None, : psi_hat.shape[1]]
    iky = 1j * ky[:, None]
    u_hat = iky * psi_hat + ikx * chi_hat
    v_hat = -ikx * psi_hat + iky * chi_hat
    u = jnp.fft.irfft2(u_hat, s=(nlat, nlon))
    v = jnp.fft.irfft2(v_hat, s=(nlat, nlon))
    return u, v


def vmap_levels(fn, axis=0):
    return jax.vmap(fn, in_axes=(axis,), out_axes=axis)
