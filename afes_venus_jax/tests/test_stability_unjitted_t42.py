import jax
import jax.numpy as jnp

from afes_venus_jax.config import default_numerics, default_planet
from afes_venus_jax.state import initial_state_T_profile
from afes_venus_jax.timestep import step
from afes_venus_jax.tendencies import spectral_to_grid


def test_short_stability():
    num = default_numerics()
    planet = default_planet()
    state = initial_state_T_profile(num, planet)
    t = 0.0
    for _ in range(5):
        state = step(state, t, num, planet)
        t += num.dt
    zeta_g, div_g, T_g, lnps_g, u, v, Phi, ps = spectral_to_grid(state, num, planet)
    assert jnp.all(jnp.isfinite(zeta_g))
    assert jnp.all(jnp.isfinite(T_g))
    mass_mean = jnp.mean(ps)
    assert mass_mean > 0
