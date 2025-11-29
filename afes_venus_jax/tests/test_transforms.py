import jax
import jax.numpy as jnp

from afes_venus_jax import spharm
from afes_venus_jax.config import default_numerics, default_planet


def test_round_trip(num):
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (num.nlat, num.nlon))
    spec = spharm.analysis_grid_to_spec(field, num)
    back = spharm.synthesis_spec_to_grid(spec, num)
    rel = jnp.linalg.norm(field - back) / jnp.linalg.norm(field)
    assert rel < 1e-10


def test_laplacian_eigen(num):
    planet = default_planet()
    key = jax.random.PRNGKey(1)
    field = jax.random.normal(key, (num.nlat, num.nlon))
    spec = spharm.analysis_grid_to_spec(field, num)
    lap_spec = spharm.laplace_fac(spec, num, planet)
    back = spharm.synthesis_spec_to_grid(lap_spec, num)
    # Compare against finite-difference approximation via spectral derivative
    kx, ky = spharm._wavenumbers(num.nlat, num.nlon, planet.a)
    d2 = spharm.synthesis_spec_to_grid(spec * (-(kx[None, :] ** 2 + ky[:, None] ** 2)), num)
    rel = jnp.linalg.norm(back - d2) / jnp.linalg.norm(d2)
    assert rel < 1e-12
