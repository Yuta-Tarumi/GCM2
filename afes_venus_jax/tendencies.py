"""Nonlinear tendency calculations on the grid."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config
from .spharm import analysis_grid_to_spec, synthesis_spec_to_grid, psi_chi_from_zeta_div, uv_from_psi_chi
from .grid import expand_grid


def _grad_central(field: jnp.ndarray, dlat: float, dlon: float):
    df_dlon = (jnp.roll(field, -1, axis=-1) - jnp.roll(field, 1, axis=-1)) / (2 * dlon)
    df_dlat = (jnp.roll(field, -1, axis=-2) - jnp.roll(field, 1, axis=-2)) / (2 * dlat)
    return df_dlat, df_dlon


def nonlinear_tendencies(state, cfg: Config):
    """Compute Eulerian nonlinear tendencies in spectral space.

    The formulation is intentionally simple: centred finite differences on
    the Gaussian grid with periodic longitudes. It is self-consistent with
    the surrogate spectral operators used in :mod:`afes_venus_jax.spharm`.
    """

    lats, lons = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, cfg.nlat), jnp.linspace(0, 2 * jnp.pi, cfg.nlon, endpoint=False)
    lat2d, lon2d = jnp.meshgrid(lats, lons, indexing="xy")
    dlat = lats[1] - lats[0]
    dlon = lons[1] - lons[0]

    # synthesize prognostics
    zeta_g = jax.vmap(lambda lev: synthesis_spec_to_grid(lev, cfg))(state.zeta)
    div_g = jax.vmap(lambda lev: synthesis_spec_to_grid(lev, cfg))(state.div)
    T_g = jax.vmap(lambda lev: synthesis_spec_to_grid(lev, cfg))(state.T)
    lnps_g = synthesis_spec_to_grid(state.lnps, cfg)

    psi_lm, chi_lm = psi_chi_from_zeta_div(state.zeta, state.div, cfg)
    u_g, v_g = jax.vmap(lambda p, c: uv_from_psi_chi(p, c, cfg))(psi_lm, chi_lm)

    def advect(field, u, v):
        df_dlat, df_dlon = _grad_central(field, dlat, dlon)
        return -(u * df_dlon + v * df_dlat)

    zeta_t = jax.vmap(advect)(zeta_g, u_g, v_g)
    div_t = jax.vmap(advect)(div_g, u_g, v_g)
    T_t = jax.vmap(advect)(T_g, u_g, v_g)

    # surface pressure tendency from mass continuity (very simple surrogate)
    u_bar = jnp.mean(u_g, axis=0)
    v_bar = jnp.mean(v_g, axis=0)
    div_u = _grad_central(u_bar, dlat, dlon)[1]
    div_v = _grad_central(v_bar, dlat, dlon)[0]
    lnps_t_grid = -(div_u + div_v)

    # back to spectral space
    zeta_t_spec = jax.vmap(lambda g: analysis_grid_to_spec(g, cfg))(zeta_t)
    div_t_spec = jax.vmap(lambda g: analysis_grid_to_spec(g, cfg))(div_t)
    T_t_spec = jax.vmap(lambda g: analysis_grid_to_spec(g, cfg))(T_t)
    lnps_t_spec = analysis_grid_to_spec(lnps_t_grid, cfg)

    return zeta_t_spec, div_t_spec, T_t_spec, lnps_t_spec
