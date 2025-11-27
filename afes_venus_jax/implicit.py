"""Semi-implicit linear gravity-wave step (simplified)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config


def semi_implicit_update(state, tendencies, cfg: Config):
    """Apply a diagonal semi-implicit correction.

    The true AFES SI scheme couples divergence, temperature, and surface
    pressure vertically for each (ℓ, m). For this compact reference
    implementation we use a scalar stability factor that damps the fast
    gravity mode while leaving the interface intact for unit tests.
    """

    zeta_t, div_t, T_t, lnps_t = tendencies
    dt = cfg.dt
    alpha = cfg.alpha

    # simple implicit factor approximating (1 + alpha*dt*omega^2)^{-1}
    stability = 1.0 / (1.0 + alpha * dt * 1e-4)
    div_new = state.div + dt * stability * div_t
    T_new = state.T + dt * stability * T_t
    lnps_new = state.lnps + dt * stability * lnps_t
    zeta_new = state.zeta + dt * zeta_t
    return zeta_new, div_new, T_new, lnps_new


def robert_asselin_filter(state_nm1, state_n, state_np1, ra: float):
    """Apply a Robert–Asselin filter to leapfrog states."""

    def filt(xm1, xn, xp1):
        return xn + 0.5 * ra * (xm1 - 2 * xn + xp1)

    return state_nm1, jax.tree_map(filt, state_nm1, state_n, state_np1)
