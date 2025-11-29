import jax
import jax.numpy as jnp

from afes_venus_jax import spharm
from afes_venus_jax.config import default_planet, default_numerics


def test_round_trip():
    num = default_numerics()
    key = jax.random.PRNGKey(0)
    field = jax.random.normal(key, (num.nlat, num.nlon))
    spec = spharm.analysis_grid_to_spec(field, num.Lmax)
    back = spharm.synthesis_spec_to_grid(spec, num.nlat, num.nlon)
    rel = jnp.linalg.norm(field - back) / jnp.linalg.norm(field)
    assert rel < 1e-10


def test_laplacian_eigen():
    num = default_numerics()
    planet = default_planet()
    spec = jnp.zeros((num.nlat, num.nlon // 2 + 1), dtype=jnp.complex128)
    for ell in (0, 1, 5, 10):
        spec = spec.at[ell, 0].set(1.0 + 0j)
        lap_spec = spharm.lap_spec(spec, num.nlat, num.nlon, planet.a)
        expected = -(ell * (ell + 1) / (planet.a ** 2))
        assert jnp.isclose(lap_spec[ell, 0], expected)
        spec = spec.at[ell, 0].set(0.0 + 0j)
