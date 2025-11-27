"""Time stepping utilities."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config
from .state import ModelState
from .tendencies import nonlinear_tendencies
from .implicit import semi_implicit_update
from .diffusion import hyperdiffusion


def step(state: ModelState, cfg: Config, time: float = 0.0) -> ModelState:
    """Single leapfrog time step with explicit nonlinear terms and a
    simplified semi-implicit gravity-wave stabilisation."""

    tendencies = nonlinear_tendencies(state, cfg, time)
    zeta_new, div_new, T_new, lnps_new = semi_implicit_update(state, tendencies, cfg)
    new_state = ModelState(zeta_new, div_new, T_new, lnps_new)
    new_state = hyperdiffusion(new_state, cfg)
    return new_state


def stepper(cfg: Config):
    """Return a jitted stepping function suitable for ``lax.scan``."""

    return jax.jit(lambda s, t: (step(s, cfg, t), None))
