"""Model state container."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .config import Numerics, Planet
from .grid import gaussian_grid
from .spharm import analysis_grid_to_spec
from .vertical import reference_temperature_profile


@dataclass
class ModelState:
    zeta: jnp.ndarray
    div: jnp.ndarray
    T: jnp.ndarray
    lnps: jnp.ndarray

    def tree_flatten(self):
        return (self.zeta, self.div, self.T, self.lnps), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        zeta, div, T, lnps = children
        return cls(zeta=zeta, div=div, T=T, lnps=lnps)


jax.tree_util.register_pytree_node(ModelState, ModelState.tree_flatten, ModelState.tree_unflatten)


def make_initial_state(planet: Planet, num: Numerics) -> ModelState:
    grid = gaussian_grid(num.nlat, num.nlon)
    z_full, T_profile = reference_temperature_profile(num)
    T_init = jnp.tile(T_profile[:, None, None], (1, num.nlat, num.nlon))
    lnps_grid = jnp.full((num.nlat, num.nlon), jnp.log(planet.ps_ref))

    key = jax.random.PRNGKey(0)
    noise = 1e-6 * jax.random.normal(key, (num.nlat, num.nlon))
    zeta0 = analysis_grid_to_spec(noise, num)
    div0 = jnp.zeros_like(zeta0)
    T0 = analysis_grid_to_spec(T_init, num)
    lnps0 = analysis_grid_to_spec(lnps_grid, num)
    return ModelState(zeta=zeta0, div=div0, T=T0, lnps=lnps0)
