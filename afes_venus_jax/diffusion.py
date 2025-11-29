"""Spectral hyperdiffusion and sponge layers."""
from __future__ import annotations

import jax.numpy as jnp

from .config import Planet, Numerics
from .spharm import _wavenumbers


def hyperdiffuse(q_hat: jnp.ndarray, num: Numerics, planet: Planet) -> jnp.ndarray:
    kx, ky = _wavenumbers(num.nlat, num.nlon, planet.a)
    lam = ky[:, None] ** 2 + kx[None, :] ** 2
    lam_max = jnp.max(lam)
    nu = 1.0 / (num.tau_hdiff * (lam_max ** (num.order_hdiff / 2)))
    return q_hat - nu * (lam ** (num.order_hdiff / 2)) * q_hat * num.dt


def apply_sponge(T_hat: jnp.ndarray, num: Numerics):
    if num.sponge_top_k <= 0:
        return T_hat
    mask = jnp.ones(T_hat.shape[0])
    mask = mask.at[-num.sponge_top_k :].multiply(0.8)
    return T_hat * mask[:, None, None]
