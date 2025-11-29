"""Diagnostic helpers."""
from __future__ import annotations

import jax.numpy as jnp

from .config import Planet, Numerics
from .spharm import synthesis_spec_to_grid


def kinetic_energy(state, num: Numerics):
    u = synthesis_spec_to_grid(state.div, num)
    v = synthesis_spec_to_grid(state.zeta, num)
    return 0.5 * jnp.mean(u ** 2 + v ** 2)


def cfl_estimate(u, v, num: Numerics, planet: Planet):
    umax = jnp.max(jnp.abs(u))
    vmax = jnp.max(jnp.abs(v))
    dx = planet.a * 2 * jnp.pi / num.nlon
    dy = planet.a * jnp.pi / num.nlat
    return umax * num.dt / dx, vmax * num.dt / dy
