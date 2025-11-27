"""Model state container."""
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple
from .config import Config


@dataclass
class ModelState:
    """Spectral prognostic variables.

    Attributes
    ----------
    zeta : array (L, nlat, nlon)
        Vorticity in spectral space.
    div : array (L, nlat, nlon)
        Divergence in spectral space.
    T : array (L, nlat, nlon)
        Temperature in spectral space.
    lnps : array (nlat, nlon)
        Log surface pressure in spectral space.
    """

    zeta: jnp.ndarray
    div: jnp.ndarray
    T: jnp.ndarray
    lnps: jnp.ndarray


jax.tree_util.register_pytree_node(
    ModelState,
    lambda s: ((s.zeta, s.div, s.T, s.lnps), None),
    lambda _, xs: ModelState(*xs),
)


def zeros_state(cfg: Config) -> ModelState:
    """Create a resting state with all spectral coefficients zero."""

    shape = (cfg.L, cfg.nlat, cfg.nlon)
    zero = jnp.zeros(shape, dtype=jnp.complex128)
    lnps = jnp.zeros((cfg.nlat, cfg.nlon), dtype=jnp.complex128)
    return ModelState(zero, zero, zero, lnps)


def combine(old: ModelState, incr: ModelState) -> ModelState:
    return ModelState(old.zeta + incr.zeta, old.div + incr.div, old.T + incr.T, old.lnps + incr.lnps)
