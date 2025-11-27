import jax
import jax.numpy as jnp
from afes_venus_jax import config, spharm


def test_round_trip(fast_cfg):
    cfg = fast_cfg
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (cfg.nlat, cfg.nlon))
    spec = spharm.analysis_grid_to_spec(field, cfg)
    back = spharm.synthesis_spec_to_grid(spec, cfg)
    rel_err = jnp.linalg.norm(field - back) / jnp.linalg.norm(field)
    assert rel_err < 1e-10


def test_laplacian_eigen():
    cfg = config.fast_config()
    key = jax.random.PRNGKey(42)
    spec = jax.random.normal(key, (cfg.nlat, cfg.nlon)) + 1j * 0
    lap_spec = spharm.laplace_fac(spec, cfg)
    kx, ky = spharm._wavenumbers(cfg.nlat, cfg.nlon, cfg.a)
    expected = spec * (-(ky[:, None] ** 2 + kx[None, :] ** 2))
    rel_err = jnp.linalg.norm(lap_spec - expected) / jnp.linalg.norm(expected)
    assert rel_err < 1e-12
