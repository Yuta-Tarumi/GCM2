"""One time step of the semi-implicit spectral model."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics
from afes_venus_jax.diffusion import apply_hyperdiffusion
from afes_venus_jax.implicit import si_solve
from afes_venus_jax.spharm import analysis_grid_to_spec, synthesis_spec_to_grid
from afes_venus_jax.tendencies import nonlinear_tendencies


@partial(jax.jit, static_argnums=(2, 3))
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
    lnps_ref = jnp.log(planet.ps_ref)
    lnps_grid = synthesis_spec_to_grid(lnps, num.nlat, num.nlon)
    lnps_grid = lnps_ref + jnp.clip(lnps_grid - lnps_ref, -2.0, 2.0)
    lnps = analysis_grid_to_spec(lnps_grid, num.Lmax)
    T_grid = synthesis_spec_to_grid(T, num.nlat, num.nlon)
    T_grid = jnp.nan_to_num(T_grid)
    T_grid = jnp.clip(T_grid, 100.0, 800.0)
    T = analysis_grid_to_spec(T_grid, num.Lmax)
    zeta_grid = synthesis_spec_to_grid(zeta, num.nlat, num.nlon)
    zeta_grid = jnp.clip(jnp.nan_to_num(zeta_grid), -1e-4, 1e-4)
    zeta = analysis_grid_to_spec(zeta_grid, num.Lmax)
    div_grid = synthesis_spec_to_grid(div, num.nlat, num.nlon)
    div_grid = jnp.clip(jnp.nan_to_num(div_grid), -1e-4, 1e-4)
    div = analysis_grid_to_spec(div_grid, num.Lmax)
    lnps = jnp.nan_to_num(lnps, nan=lnps_ref)
    return state.__class__(zeta=zeta, div=div, T=T, lnps=lnps)
