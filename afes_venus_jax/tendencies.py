"""Nonlinear tendency calculations on the grid."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config
from .spharm import analysis_grid_to_spec, synthesis_spec_to_grid, psi_chi_from_zeta_div, uv_from_psi_chi
from .state import reference_temperature_profile
from .vertical import sigma_levels


def _grad_central(field: jnp.ndarray, dlat: float, dlon: float):
    df_dlon = (jnp.roll(field, -1, axis=-1) - jnp.roll(field, 1, axis=-1)) / (2 * dlon)
    df_dlat = (jnp.roll(field, -1, axis=-2) - jnp.roll(field, 1, axis=-2)) / (2 * dlat)
    return df_dlat, df_dlon


def _temperature_forcing(lat2d, lon2d, time: float, cfg: Config):
    """Compute diurnally varying solar heating in grid space."""

    _, _, z_half = sigma_levels(cfg.L)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])

    local_time = lon2d - cfg.Omega * time
    diurnal = jnp.maximum(0.0, jnp.cos(local_time))
    meridional = jnp.cos(lat2d)
    vertical = jnp.exp(-0.5 * ((z_full - cfg.heating_peak_height) / cfg.heating_width) ** 2)

    base = cfg.heating_rate * vertical[:, None, None]
    return base * diurnal[None, :, :] * meridional[None, :, :]


def _newtonian_cooling(T_g: jnp.ndarray, cfg: Config):
    """Relax toward the reference vertical profile."""

    T_ref = reference_temperature_profile(cfg)[:, None, None]
    return -(T_g - T_ref) / cfg.tau_newton


def nonlinear_tendencies(state, cfg: Config, time: float = 0.0):
    """Compute Eulerian nonlinear tendencies in spectral space.

    The formulation is intentionally simple: centred finite differences on
    the Gaussian grid with periodic longitudes. It is self-consistent with
    the surrogate spectral operators used in :mod:`afes_venus_jax.spharm`.
    """

    lats, lons = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, cfg.nlat), jnp.linspace(0, 2 * jnp.pi, cfg.nlon, endpoint=False)
    lat2d, lon2d = jnp.meshgrid(lats, lons, indexing="ij")
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

    # Diabatic tendencies in grid space
    heating = _temperature_forcing(lat2d, lon2d, time, cfg)
    cooling = _newtonian_cooling(T_g, cfg)
    T_t = T_t + heating + cooling

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
