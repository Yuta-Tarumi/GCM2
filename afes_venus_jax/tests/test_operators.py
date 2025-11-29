import jax
import jax.numpy as jnp

from afes_venus_jax import spharm
from afes_venus_jax.config import default_planet


def test_zeta_div_cycle(num):
    planet = default_planet()
    key = jax.random.PRNGKey(1)
    zeta = jax.random.normal(key, (num.nlat, num.nlon)) + 1j * 0
    div = jax.random.normal(key, (num.nlat, num.nlon)) + 1j * 0
    psi, chi = spharm.psi_chi_from_zeta_div(zeta, div, num, planet)
    u, v = spharm.uv_from_psi_chi(psi, chi, num, planet)
    # recompute
    kx, ky = spharm._wavenumbers(num.nlat, num.nlon, planet.a)
    u_lm = jnp.fft.fft2(u) / (num.nlat * num.nlon)
    v_lm = jnp.fft.fft2(v) / (num.nlat * num.nlon)
    vort = (1j * kx[None, :] * v_lm - 1j * ky[:, None] * u_lm)
    div2 = (1j * kx[None, :] * u_lm + 1j * ky[:, None] * v_lm)
    err_vort = jnp.linalg.norm(vort - zeta) / jnp.linalg.norm(zeta)
    err_div = jnp.linalg.norm(div2 - div) / jnp.linalg.norm(div)
    assert err_vort < 2e-2
    assert err_div < 2e-2
