import jax
import jax.numpy as jnp
from afes_venus_jax import spharm


def test_zeta_div_cycle(fast_cfg):
    cfg = fast_cfg
    key = jax.random.PRNGKey(1)
    zeta = jax.random.normal(key, (cfg.nlat, cfg.nlon)) + 1j * 0
    div = jax.random.normal(key, (cfg.nlat, cfg.nlon)) + 1j * 0
    zeta = zeta.at[0, 0].set(0.0)
    div = div.at[0, 0].set(0.0)
    psi, chi = spharm.psi_chi_from_zeta_div(zeta, div, cfg)
    u, v = spharm.uv_from_psi_chi(psi, chi, cfg)
    # recompute vorticity/divergence from u,v using spectral derivatives
    kx, ky = spharm._wavenumbers(cfg.nlat, cfg.nlon, cfg.a)
    ikx = 1j * kx
    iky = 1j * ky
    u_lm = jnp.fft.fft2(u) / (cfg.nlat * cfg.nlon)
    v_lm = jnp.fft.fft2(v) / (cfg.nlat * cfg.nlon)
    vort = (ikx[None, :] * v_lm - iky[:, None] * u_lm)
    div2 = (ikx[None, :] * u_lm + iky[:, None] * v_lm)
    err_vort = jnp.linalg.norm(vort - zeta) / jnp.linalg.norm(zeta)
    err_div = jnp.linalg.norm(div2 - div) / jnp.linalg.norm(div)
    assert err_vort < 1e-6
    assert err_div < 1e-6
