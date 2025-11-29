"""Eulerian tendencies for vorticity–divergence form."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics
from afes_venus_jax.grid import gaussian_grid
from afes_venus_jax.physics import diurnal_heating, newtonian_cooling
from afes_venus_jax.spharm import (
    analysis_grid_to_spec,
    grad_sphere,
    invert_laplacian,
    lap_spec,
    synthesis_spec_to_grid,
    uv_from_psi_chi,
)
from afes_venus_jax.vertical import (
    hydrostatic_geopotential,
    level_altitudes,
    sigma_levels,
    vertical_diffusion,
)


def spectral_to_grid(state, num: Numerics, planet: Planet, grid=None):
    grid = grid or gaussian_grid(num.nlat, num.nlon)
    zeta_g = jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.zeta)
    div_g = jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.div)
    T_g = jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.T)
    lnps_g = synthesis_spec_to_grid(state.lnps, num.nlat, num.nlon)
    lnps_ref = jnp.log(planet.ps_ref)
    lnps_g = jnp.nan_to_num(lnps_g, nan=lnps_ref)
    lnps_g = lnps_ref + jnp.clip(lnps_g - lnps_ref, -2.0, 2.0)
    psi_hat = jax.vmap(lambda z: invert_laplacian(z, num.nlat, num.nlon, planet.a))(state.zeta)
    chi_hat = jax.vmap(lambda d: invert_laplacian(d, num.nlat, num.nlon, planet.a))(state.div)
    u = []
    v = []
    for k in range(num.L):
        uk, vk = uv_from_psi_chi(psi_hat[k], chi_hat[k], num.nlat, num.nlon, planet.a)
        u.append(uk)
        v.append(vk)
    u = jnp.stack(u, axis=0)
    v = jnp.stack(v, axis=0)
    sigma_full, sigma_half = sigma_levels(num.L)
    ps = jnp.exp(lnps_g)
    Phi = hydrostatic_geopotential(T_g, ps, sigma_half, planet)
    return zeta_g, div_g, T_g, lnps_g, u, v, Phi, ps


def nonlinear_tendencies(state, t: float, num: Numerics, planet: Planet, grid=None):
    grid = grid or gaussian_grid(num.nlat, num.nlon)
    zeta_g, div_g, T_g, lnps_g, u, v, Phi, ps = spectral_to_grid(state, num, planet, grid)
    sigma_full, _ = sigma_levels(num.L)
    _, z_half = level_altitudes(num.L)
    Phi_spec = jax.vmap(lambda x: analysis_grid_to_spec(x, num.Lmax))(Phi)
    lap_Phi_spec = jax.vmap(lambda x: lap_spec(x, num.nlat, num.nlon, planet.a))(Phi_spec)
    lap_Phi = jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(lap_Phi_spec)
    f = 2 * planet.Omega * jnp.sin(grid.lat2d)

    zeta_dot = []
    div_dot = []
    T_adv = []
    for k in range(num.L):
        dlon_zeta, dlat_zeta = grad_sphere(zeta_g[k] + f, num.nlat, num.nlon, planet.a)
        dlon_div, dlat_div = grad_sphere(div_g[k], num.nlat, num.nlon, planet.a)
        dlon_T, dlat_T = grad_sphere(T_g[k], num.nlat, num.nlon, planet.a)
        zeta_dot.append(-(u[k] * dlon_zeta + v[k] * dlat_zeta))
        div_nonlin = -(u[k] * dlon_div + v[k] * dlat_div)
        metric_stretch = -(div_g[k] * (div_g[k] + 0.5 * (jnp.tan(grid.lat2d) * v[k] / planet.a)) + (zeta_g[k] + f) * (v[k] * jnp.tan(grid.lat2d) / planet.a))
        div_dot.append(div_nonlin + metric_stretch - lap_Phi[k])
        T_adv.append(-(u[k] * dlon_T + v[k] * dlat_T))
    zeta_dot = jnp.stack(zeta_dot)
    div_dot = jnp.stack(div_dot)
    T_dot = jnp.stack(T_adv)
    # vertical coupling via diffusive fluxes
    kappa_scale = jnp.square(sigma_full)
    div_dot = div_dot + vertical_diffusion(div_g, z_half, num.kappa_v, kappa_scale)
    T_dot = T_dot + vertical_diffusion(T_g, z_half, num.kappa_v, kappa_scale)
    # physics
    T_dot = T_dot + diurnal_heating(T_g, grid, t, planet, num) + newtonian_cooling(T_g, num)
    lnps_dot = -jnp.mean(div_g, axis=0)
    zeta_spec = jax.vmap(lambda x: analysis_grid_to_spec(x, num.Lmax))(zeta_dot)
    div_spec = jax.vmap(lambda x: analysis_grid_to_spec(x, num.Lmax))(div_dot)
    T_spec = jax.vmap(lambda x: analysis_grid_to_spec(x, num.Lmax))(T_dot)
    lnps_spec = analysis_grid_to_spec(lnps_dot, num.Lmax)
    return zeta_spec, div_spec, T_spec, lnps_spec
