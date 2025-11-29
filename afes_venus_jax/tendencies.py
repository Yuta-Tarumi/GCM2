"""Explicit nonlinear and physics tendencies in grid space."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Planet, Numerics
from .grid import gaussian_grid
from .spharm import analysis_grid_to_spec, synthesis_spec_to_grid, psi_chi_from_zeta_div, uv_from_psi_chi, laplace_fac
from .vertical import hydrostatic_geopotential, sigma_levels
from .physics import diurnal_heating, newtonian_cooling


def spectral_to_grid(state, num: Numerics):
    zeta = synthesis_spec_to_grid(state.zeta, num)
    div = synthesis_spec_to_grid(state.div, num)
    T = synthesis_spec_to_grid(state.T, num)
    lnps = synthesis_spec_to_grid(state.lnps, num)
    return zeta, div, T, lnps


def grid_to_spectral(zeta, div, T, lnps, num: Numerics):
    return (
        analysis_grid_to_spec(zeta, num),
        analysis_grid_to_spec(div, num),
        analysis_grid_to_spec(T, num),
        analysis_grid_to_spec(lnps, num),
    )


def nonlinear_tendencies(state, t: float, planet: Planet, num: Numerics):
    grid = gaussian_grid(num.nlat, num.nlon)
    sigma_full, _, _ = sigma_levels(num)

    zeta_g, div_g, T_g, lnps_g = spectral_to_grid(state, num)
    psi, chi = psi_chi_from_zeta_div(state.zeta, state.div, num, planet)
    u, v = uv_from_psi_chi(psi, chi, num, planet)

    phi = hydrostatic_geopotential(T_g, lnps_g, planet, num)

    # Horizontal derivatives via spectral multiplication
    def ddx(f):
        spec = analysis_grid_to_spec(f, num)
        kx = 1j * 2 * jnp.pi * jnp.fft.fftfreq(num.nlon)
        return synthesis_spec_to_grid(spec * kx[None, :], num)

    def ddy(f):
        spec = analysis_grid_to_spec(f, num)
        ky = 1j * 2 * jnp.pi * jnp.fft.fftfreq(num.nlat)
        return synthesis_spec_to_grid(spec * ky[:, None], num)

    zeta_tend = -(u * ddx(zeta_g) + v * ddy(zeta_g))
    phi_spec = analysis_grid_to_spec(phi, num)
    div_tend = -(u * ddx(div_g) + v * ddy(div_g)) + synthesis_spec_to_grid(
        -laplace_fac(phi_spec, num, planet), num
    )
    T_tend = -(u * ddx(T_g) + v * ddy(T_g))
    lnps_tend = jnp.zeros_like(state.lnps)

    # Physics tendencies
    z_full, _ = hydrostatic_levels(num)
    Q = diurnal_heating(grid.lat2d, grid.lon2d, z_full, t, planet, num)
    cool = newtonian_cooling(T_g, planet, num)
    T_tend = T_tend + Q + cool

    return grid_to_spectral(zeta_tend, div_tend, T_tend, lnps_tend, num)


def hydrostatic_levels(num: Numerics):
    sigma_full, _, z_half = sigma_levels(num)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return z_full, sigma_full
