"""Spectral hyperdiffusion and sponge layer."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics


def hyperdiffusion_operator(num: Numerics, planet: Planet):
    ell = jnp.arange(num.nlat)
    lam = ell * (ell + 1) / (planet.a ** 2)
    lam_max = lam[-1]
    nu = 1.0 / (num.tau_hdiff * (lam_max ** 2))
    coeff = -nu * (lam[:, None] ** 2)
    return coeff


def apply_hyperdiffusion(state, num: Numerics, planet: Planet):
    coeff = hyperdiffusion_operator(num, planet)
    coef3d = coeff[None, :, :]
    zeta = state.zeta + coef3d * state.zeta
    div = state.div + coef3d * state.div
    T = state.T + coef3d * state.T
    lnps = state.lnps + coeff * state.lnps
    return state.__class__(zeta=zeta, div=div, T=T, lnps=lnps)
