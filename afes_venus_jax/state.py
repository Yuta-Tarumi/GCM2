"""Model state container and initialisation helpers."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics, default_planet, default_numerics
from afes_venus_jax.grid import gaussian_grid
from afes_venus_jax.spharm import analysis_grid_to_spec
from afes_venus_jax.vertical import sigma_levels, level_altitudes


@dataclass
class ModelState:
    zeta: jnp.ndarray  # [L, nlat, nlon//2+1]
    div: jnp.ndarray
    T: jnp.ndarray
    lnps: jnp.ndarray  # [nlat, nlon//2+1]

    def copy(self):
        return ModelState(self.zeta.copy(), self.div.copy(), self.T.copy(), self.lnps.copy())


@jax.jit(static_argnums=(0,))
def initial_temperature_profile(L: int):
    z_full, _ = level_altitudes(L)
    T0 = 730.0 - (730.0 - 170.0) * (z_full / z_full[-1])
    # introduce a weak inversion near 60 km
    T0 = jnp.where((z_full > 55e3) & (z_full < 65e3), T0 + 10.0, T0)
    return T0


def initial_state_T_profile(num: Numerics | None = None, planet: Planet | None = None) -> ModelState:
    num = num or default_numerics()
    planet = planet or default_planet()
    grid = gaussian_grid(num.nlat, num.nlon)
    sigma_full, _ = sigma_levels(num.L)
    T0 = initial_temperature_profile(num.L)
    T_grid = jnp.array(T0)[:, None, None] * jnp.ones((num.L, num.nlat, num.nlon))
    T_spec = jax.vmap(lambda x: analysis_grid_to_spec(x, num.Lmax))(T_grid)
    lnps_spec = analysis_grid_to_spec(jnp.ones((num.nlat, num.nlon)) * jnp.log(planet.ps_ref), num.Lmax)
    zeros = jnp.zeros((num.L, num.nlat, num.nlon // 2 + 1), dtype=jnp.complex128)
    return ModelState(zeta=zeros, div=zeros, T=T_spec, lnps=lnps_spec)
