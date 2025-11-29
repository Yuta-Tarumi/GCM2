"""Diagnostics helpers."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics
from afes_venus_jax.spharm import synthesis_spec_to_grid


def mass(ps_grid):
    return jnp.mean(ps_grid)


def ke(u, v):
    return 0.5 * jnp.mean(u ** 2 + v ** 2)


def check_nan(state) -> bool:
    return jnp.any(~jnp.isfinite(state.zeta)) or jnp.any(~jnp.isfinite(state.div)) or jnp.any(~jnp.isfinite(state.T))
