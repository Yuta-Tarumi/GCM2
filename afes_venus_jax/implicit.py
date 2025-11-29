"""Semi-implicit linear solver for gravity-wave terms.

The solver couples divergence, temperature, and surface pressure tendencies
through a hydrostatic gravity-wave operator. A reference temperature derived
from the current state sets an equivalent depth ``c² = R Tref`` so the block
system becomes a Helmholtz solve in spectral space. Divergence and surface
pressure share this implicit gravity-wave step; temperature receives the
resulting compressional heating implicitly via ``∂T/∂t ∝ -γ div``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from afes_venus_jax.config import Planet, Numerics
from afes_venus_jax.spharm import _wavenumbers


def si_matrices(num: Numerics, planet: Planet):
    L = num.L
    ell = jnp.arange(num.nlat)
    Lam = ell * (ell + 1) / (planet.a ** 2)
    base = jnp.eye(L + 1)
    mats = []
    for lam in Lam:
        diag = 1 + num.alpha * num.dt * lam
        block = jnp.eye(2 * L + 1) * diag
        mats.append(block)
    return jnp.stack(mats)


def si_solve(state, rhs, num: Numerics, planet: Planet, mats=None):
    """Solve the semi-implicit gravity-wave system in spectral space.

    The implicit step treats the fast gravity-wave coupling between divergence
    and surface pressure using a Helmholtz operator per spectral coefficient,
    while temperature receives compressional heating based on the updated
    divergence. Vorticity is untouched by the SI solve.
    """

    zeta_rhs, div_rhs, T_rhs, lnps_rhs = rhs

    # Equivalent depth from the vertically averaged base temperature.
    T_ref_levels = jnp.real(state.T[:, 0, 0])
    T_ref = jnp.clip(jnp.mean(T_ref_levels), 150.0, 800.0)
    c2 = planet.R_gas * T_ref

    # Spectral Helmholtz operator for the Laplacian eigenvalues.
    kx, ky = _wavenumbers(num.nlat, num.nlon, planet.a)
    lam = ky[:, None] ** 2 + kx[None, : div_rhs.shape[2]] ** 2
    alpha_dt = num.alpha * num.dt

    mean_div_rhs = jnp.mean(div_rhs, axis=0)
    denom = 1.0 + (alpha_dt ** 2) * c2 * lam
    lnps_new = (lnps_rhs - alpha_dt * mean_div_rhs) / denom
    div_new = div_rhs - alpha_dt * c2 * lam[None, :, :] * lnps_new[None, :, :]

    gamma = (planet.R_gas / planet.cp) * T_ref
    T_new = T_rhs - alpha_dt * gamma * div_new

    return zeta_rhs, div_new, T_new, lnps_new
