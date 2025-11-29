"""One time step of the semi-implicit spectral model."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics
from afes_venus_jax.diffusion import apply_hyperdiffusion
from afes_venus_jax.implicit import si_solve
from afes_venus_jax.tendencies import nonlinear_tendencies


@jax.jit(static_argnums=(2, 3))
def step(state, t: float, num: Numerics, planet: Planet):
    # 1) explicit tendencies
    rhs = nonlinear_tendencies(state, t, num, planet)
    # 2) semi-implicit solve
    zeta_new, div_new, T_new, lnps_new = si_solve(state, rhs, num, planet)
    # 3) leapfrog update
    zeta = state.zeta + num.dt * zeta_new
    div = state.div + num.dt * div_new
    T = state.T + num.dt * T_new
    lnps = state.lnps + num.dt * lnps_new
    # 4) diffusion
    new_state = state.__class__(zeta=zeta, div=div, T=T, lnps=lnps)
    new_state = apply_hyperdiffusion(new_state, num, planet)
    # 5) Robert–Asselin filter (simple relaxation toward mean)
    zeta = (1 - num.ra) * new_state.zeta + num.ra * state.zeta
    div = (1 - num.ra) * new_state.div + num.ra * state.div
    T = (1 - num.ra) * new_state.T + num.ra * state.T
    lnps = (1 - num.ra) * new_state.lnps + num.ra * state.lnps
    return state.__class__(zeta=zeta, div=div, T=T, lnps=lnps)
