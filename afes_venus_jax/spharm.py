"""Simplified spherical-harmonic utilities using FFT-based spectral space.

This module mimics the interface of an S2FFT-backed transform but internally uses
2-D FFTs for testing convenience. The spectral representation is complex-valued
with shape ``(nlat, nlon//2+1)`` matching ``rfft2`` output. Derivative operators
apply spherical metric factors (``m/(a cosφ)``, ``1/a``) and use the harmonic
eigenvalue ``ℓ(ℓ+1)/a²`` for the Laplacian so advection and pressure-gradient
terms remain properly scaled on the sphere.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from afes_venus_jax.grid import gaussian_grid


@functools.partial(jax.jit, static_argnums=(1,))
def analysis_grid_to_spec(field_grid: jnp.ndarray, Lmax: int | None = None) -> jnp.ndarray:
    return jnp.fft.rfft2(field_grid)


@functools.partial(jax.jit, static_argnums=(1, 2))
def synthesis_spec_to_grid(flm: jnp.ndarray, nlat: int, nlon: int) -> jnp.ndarray:
    return jnp.fft.irfft2(flm, s=(nlat, nlon))


def _angular_wavenumbers(nlat: int, nlon: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return angular wavenumbers for latitude and longitude.

    Longitude samples span ``[0, 2π)`` so the angular frequency associated with
    the rFFT output is the integer zonal wavenumber ``m``. The latitude spacing
    is treated as ``π / nlat`` for the derivative operator; this keeps the
    meridional multiplier in the right units (radians⁻¹) while remaining
    compatible with the simplified FFT-based representation.
    """

    m = jnp.fft.rfftfreq(nlon, d=2 * jnp.pi / nlon) * 2 * jnp.pi
    # Treat latitude sampling as equally spaced in φ with spacing π / nlat.
    # ``fftfreq`` returns signed frequencies (positive then negative) so the
    # second half of the spectrum correctly represents negative ky rather than
    # large positive wavenumbers.
    dphi = jnp.pi / nlat
    ky = jnp.fft.fftfreq(nlat, d=dphi) * 2 * jnp.pi
    return ky, m


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def lap_spec(flm: jnp.ndarray, nlat: int, nlon: int, a: float) -> jnp.ndarray:
    ell_like, _ = _angular_wavenumbers(nlat, nlon)
    ell_grid = jnp.abs(ell_like)[:, None]
    return -(ell_grid * (ell_grid + 1.0) / (a ** 2)) * flm


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def invert_laplacian(flm: jnp.ndarray, nlat: int, nlon: int, a: float) -> jnp.ndarray:
    ell_like, _ = _angular_wavenumbers(nlat, nlon)
    ell_grid = jnp.abs(ell_like)[:, None]
    denom = ell_grid * (ell_grid + 1.0) / (a ** 2)
    denom = jnp.where(denom == 0, jnp.inf, denom)
    return flm / (-denom)


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def uv_from_psi_chi(psi_hat: jnp.ndarray, chi_hat: jnp.ndarray, nlat: int, nlon: int, a: float):
    grid = gaussian_grid(nlat, nlon)
    coslat = jnp.cos(grid.lat2d)
    coslat_safe = jnp.where(jnp.abs(coslat) < 1e-6, 1e-6, coslat)
    ell_like, m = _angular_wavenumbers(nlat, nlon)
    dphi_psi_hat = 1j * (ell_like[:, None]) * psi_hat
    dphi_chi_hat = 1j * (ell_like[:, None]) * chi_hat
    dlam_psi_hat = 1j * m[None, : psi_hat.shape[1]] * psi_hat
    dlam_chi_hat = 1j * m[None, : chi_hat.shape[1]] * chi_hat
    dphi_psi = jnp.fft.irfft2(dphi_psi_hat, s=(nlat, nlon))
    dphi_chi = jnp.fft.irfft2(dphi_chi_hat, s=(nlat, nlon))
    dlam_psi = jnp.fft.irfft2(dlam_psi_hat, s=(nlat, nlon))
    dlam_chi = jnp.fft.irfft2(dlam_chi_hat, s=(nlat, nlon))
    a_inv = 1.0 / a
    u = a_inv * ((-dphi_psi + dlam_chi) / coslat_safe)
    v = a_inv * (dlam_psi + dphi_chi)
    return u, v


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def grad_sphere(field: jnp.ndarray, nlat: int, nlon: int, a: float):
    """Return (d/dx, d/dy) on the sphere with proper metric factors.

    d/dx corresponds to ``(1 / (a cosφ)) ∂/∂λ`` and d/dy to ``(1 / a) ∂/∂φ``
    using the simplified FFT representation.
    """

    grid = gaussian_grid(nlat, nlon)
    coslat = jnp.cos(grid.lat2d)
    coslat_safe = jnp.where(jnp.abs(coslat) < 1e-6, 1e-6, coslat)
    ell_like, m = _angular_wavenumbers(nlat, nlon)
    spec = analysis_grid_to_spec(field)
    dphi_hat = 1j * (ell_like[:, None]) * spec
    dlam_hat = 1j * m[None, : spec.shape[1]] * spec
    dphi = jnp.fft.irfft2(dphi_hat, s=(nlat, nlon))
    dlam = jnp.fft.irfft2(dlam_hat, s=(nlat, nlon))
    a_inv = 1.0 / a
    return a_inv * dlam / coslat_safe, a_inv * dphi


def vmap_levels(fn, axis=0):
    return jax.vmap(fn, in_axes=(axis,), out_axes=axis)
