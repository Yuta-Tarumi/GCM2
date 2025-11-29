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
    key = jax.random.PRNGKey(1)
    field = jax.random.normal(key, (num.nlat, num.nlon))
    spec = spharm.analysis_grid_to_spec(field, num.Lmax)
    lap_spec = spharm.lap_spec(spec, num.nlat, num.nlon, planet.a)
    lap_grid = spharm.synthesis_spec_to_grid(lap_spec, num.nlat, num.nlon)
    # Finite difference using spectral derivatives as reference
    kx = jnp.fft.fftfreq(num.nlon) * 2 * jnp.pi / planet.a
    ky = jnp.fft.fftfreq(num.nlat) * 2 * jnp.pi / planet.a
    d2 = spharm.synthesis_spec_to_grid(spec * (-(ky[:, None] ** 2 + kx[None, : spec.shape[1]] ** 2)), num.nlat, num.nlon)
    rel = jnp.linalg.norm(lap_grid - d2) / jnp.linalg.norm(d2)
    assert rel < 1e-12
