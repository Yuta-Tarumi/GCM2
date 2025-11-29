import jax
import jax.numpy as jnp

from afes_venus_jax.config import default_numerics, default_planet
from afes_venus_jax.implicit import implicit_solve


def test_si_stability(num):
    planet = default_planet()
    key = jax.random.PRNGKey(0)
    div = jax.random.normal(key, (num.nlat, num.nlon)) + 1j * 0
    T = jax.random.normal(key, (num.nlat, num.nlon)) + 1j * 0
    lnps = jax.random.normal(key, (num.nlat, num.nlon)) + 1j * 0
    div_new, T_new, lnps_new = implicit_solve(div, T, lnps, num, planet)
    assert jnp.max(jnp.abs(div_new)) <= jnp.max(jnp.abs(div))
    assert jnp.max(jnp.abs(lnps_new)) <= jnp.max(jnp.abs(lnps))
