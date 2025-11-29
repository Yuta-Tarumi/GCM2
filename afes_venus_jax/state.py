"""Model state container and initialisation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics, default_planet, default_numerics
from afes_venus_jax.grid import gaussian_grid
from afes_venus_jax.spharm import analysis_grid_to_spec, lap_spec
from afes_venus_jax.vertical import sigma_levels, level_altitudes


@jax.tree_util.register_pytree_node_class
@dataclass
class ModelState:
    zeta: jnp.ndarray  # [L, nlat, nlon//2+1]
    div: jnp.ndarray
    T: jnp.ndarray
    lnps: jnp.ndarray  # [nlat, nlon//2+1]

    def copy(self):
        return ModelState(self.zeta.copy(), self.div.copy(), self.T.copy(), self.lnps.copy())

    def tree_flatten(self):
        children = (self.zeta, self.div, self.T, self.lnps)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        zeta, div, T, lnps = children
        return cls(zeta=zeta, div=div, T=T, lnps=lnps)


@partial(jax.jit, static_argnums=(0,))
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
    T0 = initial_temperature_profile(num.L)
    T_grid = jnp.array(T0)[:, None, None] * jnp.ones((num.L, num.nlat, num.nlon))
    T_spec = jax.vmap(lambda x: analysis_grid_to_spec(x, num.Lmax))(T_grid)
    lnps_spec = analysis_grid_to_spec(jnp.ones((num.nlat, num.nlon)) * jnp.log(planet.ps_ref), num.Lmax)
    z_full, _ = level_altitudes(num.L)
    # Solid-body rotation that ramps from 0 m/s at the surface to 100 m/s at 70 km,
    # following a cos(latitude) profile and held constant above.
    peak_height = 70_000.0
    peak_speed = 100.0
    wind_speed_levels = peak_speed * jnp.minimum(z_full / peak_height, 1.0)
    psi_lat_profile = -(planet.a * (grid.lat2d / 2.0 + jnp.sin(2.0 * grid.lat2d) / 4.0))
    psi_grid = wind_speed_levels[:, None, None] * psi_lat_profile[None, :, :]
    chi_grid = jnp.zeros_like(psi_grid)
    psi_spec = jax.vmap(lambda x: analysis_grid_to_spec(x, num.Lmax))(psi_grid)
    chi_spec = jax.vmap(lambda x: analysis_grid_to_spec(x, num.Lmax))(chi_grid)
    zeta_spec = jax.vmap(lambda x: lap_spec(x, num.nlat, num.nlon, planet.a))(psi_spec)
    div_spec = jax.vmap(lambda x: lap_spec(x, num.nlat, num.nlon, planet.a))(chi_spec)
    return ModelState(zeta=zeta_spec, div=div_spec, T=T_spec, lnps=lnps_spec)
