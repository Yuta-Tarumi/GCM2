"""Spectral hyperdiffusion."""
from __future__ import annotations

import jax.numpy as jnp
from .config import Config
from .spharm import laplace_fac


def hyperdiffusion(state, cfg: Config):
    """Apply ∇⁴ hyperdiffusion in spectral space."""

    ellmax = cfg.nlat // 2
    kxky = laplace_fac(jnp.ones((cfg.nlat, cfg.nlon), dtype=jnp.complex128), cfg)
    eig = -kxky  # positive definite
    eig2 = eig ** 2
    eigmax = jnp.max(eig2)
    eig2_safe = eig2 + 1e-6 * eigmax
    nu4 = 1.0 / (cfg.tau_hdiff * eigmax)

    def damp(arr):
        return arr - cfg.dt * nu4 * eig2_safe * arr

    return state.__class__(
        damp(state.zeta),
        damp(state.div),
        damp(state.T),
        state.lnps,
    )
