"""Vertical grid construction and hydrostatic utilities."""
from __future__ import annotations

import jax.numpy as jnp
from .config import Config


def sigma_levels(L: int, H_ref: float = 15_000.0):
    """Construct Lorenz-staggered sigma coordinate.

    Parameters
    ----------
    L : int
        Number of full levels.
    H_ref : float
        Reference scale height [m] used to map a target height grid to
        sigma = exp(-z/H_ref).

    Returns
    -------
    sigma_full : jnp.ndarray
        Full-level sigma values, shape (L,).
    sigma_half : jnp.ndarray
        Half-level sigma values, shape (L+1,).
    z_half : jnp.ndarray
        Half-level geometric heights [m], shape (L+1,).
    """

    z_half = jnp.linspace(0.0, 120_000.0, L + 1)
    sigma_half = jnp.exp(-z_half / H_ref)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    return sigma_full, sigma_half, z_half


def hydrostatic_phi(
    T: jnp.ndarray,
    lnps: jnp.ndarray,
    cfg: Config,
    sigma_full: jnp.ndarray,
    sigma_half: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate hydrostatic balance to compute geopotential.

    Parameters
    ----------
    T : array (..., L, nlat, nlon)
        Temperature [K].
    lnps : array (..., nlat, nlon)
        Log surface pressure.
    cfg : Config
    sigma_full, sigma_half : array
        Sigma coordinates.

    Returns
    -------
    phi : array (..., L, nlat, nlon)
        Geopotential [m^2 s^-2].
    """

    ps = jnp.exp(lnps)[..., None, None]
    # pressure at full levels
    p_full = ps * sigma_full[:, None, None]
    # delta ln p between half levels
    dlnp = jnp.log(ps * sigma_half[:-1, None, None]) - jnp.log(ps * sigma_half[1:, None, None])
    # Hypsometric relation: dPhi = R*T*dlnp
    layer_phi = cfg.R_gas * T * dlnp
    # integrate from top downward
    phi = jnp.cumsum(layer_phi[::-1], axis=-3)[::-1]
    return phi
