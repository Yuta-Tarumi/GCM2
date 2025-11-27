"""Model state container."""
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Tuple
from .config import Config
from .spharm import analysis_grid_to_spec
from .vertical import sigma_levels


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


def reference_temperature_profile(cfg: Config) -> jnp.ndarray:
    """Construct a vertically varying Venus temperature profile."""

    _, sigma_half, z_half = sigma_levels(cfg.L)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    # Linear interpolation between a hot surface and a cool upper atmosphere
    return jnp.interp(
        z_full,
        jnp.array([0.0, z_half[-1]]),
        jnp.array([cfg.T_surface, cfg.T_top]),
    )


def _temperature_spectrum_from_profile(profile: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Convert a 1D vertical temperature profile to spectral space."""

    tiled = jnp.tile(profile[:, None, None], (1, cfg.nlat, cfg.nlon))
    return jax.vmap(lambda g: analysis_grid_to_spec(g, cfg))(tiled)


def zeros_state(cfg: Config, include_reference_temperature: bool = True) -> ModelState:
    """Create a resting state with optional climatological temperature.

    Parameters
    ----------
    cfg : Config
    include_reference_temperature : bool, optional
        When True, initialise temperature with a simple Venus-like vertical
        profile following Lebonnois et al. (2015) with a hot surface and a cool
        upper atmosphere. When False, temperature coefficients are set to zero.
    """

    shape = (cfg.L, cfg.nlat, cfg.nlon)
    zero = jnp.zeros(shape, dtype=jnp.complex128)
    lnps = jnp.zeros((cfg.nlat, cfg.nlon), dtype=jnp.complex128)
    if include_reference_temperature:
        profile = reference_temperature_profile(cfg)
        T = _temperature_spectrum_from_profile(profile, cfg)
    else:
        T = zero
    return ModelState(zero, zero, T, lnps)


def combine(old: ModelState, incr: ModelState) -> ModelState:
    return ModelState(old.zeta + incr.zeta, old.div + incr.div, old.T + incr.T, old.lnps + incr.lnps)
