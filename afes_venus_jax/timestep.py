"""Time-stepping orchestration for one semi-implicit leapfrog step."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Planet, Numerics
from .diffusion import hyperdiffuse
from .implicit import implicit_solve
from .spharm import analysis_grid_to_spec
from .tendencies import nonlinear_tendencies


def step(state_prev, state_curr, t: float, planet: Planet, num: Numerics):
    # Explicit tendencies at time n
    zeta_t, div_t, T_t, lnps_t = nonlinear_tendencies(state_curr, t, planet, num)

    # Leapfrog predictor
    zeta_pred = state_prev.zeta + 2 * num.dt * zeta_t
    div_pred = state_prev.div + 2 * num.dt * div_t
    T_pred = state_prev.T + 2 * num.dt * T_t
    lnps_pred = state_prev.lnps + 2 * num.dt * lnps_t

    # Semi-implicit correction on fast modes
    div_si, T_si, lnps_si = implicit_solve(div_pred, T_pred, lnps_pred, num, planet)

    # Diffusion
    zeta_f = hyperdiffuse(zeta_pred, num, planet)
    div_f = hyperdiffuse(div_si, num, planet)
    T_f = hyperdiffuse(T_si, num, planet)
    lnps_f = lnps_si

    # Robert-Asselin filter
    zeta_new = state_curr.zeta + 0.5 * (zeta_f - state_prev.zeta)
    zeta_new = zeta_new + num.ra * (state_prev.zeta - 2 * state_curr.zeta + zeta_f)

    div_new = state_curr.div + 0.5 * (div_f - state_prev.div)
    div_new = div_new + num.ra * (state_prev.div - 2 * state_curr.div + div_f)

    T_new = state_curr.T + 0.5 * (T_f - state_prev.T)
    T_new = T_new + num.ra * (state_prev.T - 2 * state_curr.T + T_f)

    lnps_new = state_curr.lnps + 0.5 * (lnps_f - state_prev.lnps)
    lnps_new = lnps_new + num.ra * (state_prev.lnps - 2 * state_curr.lnps + lnps_f)

    new_state = state_curr.__class__(zeta=zeta_new, div=div_new, T=T_new, lnps=lnps_new)
    return new_state
