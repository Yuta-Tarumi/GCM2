import jax.numpy as jnp

from afes_venus_jax.config import default_numerics, default_planet
from afes_venus_jax.state import make_initial_state
from afes_venus_jax.timestep import step


def test_short_stability():
    planet = default_planet()
    num = default_numerics()
    state_prev = make_initial_state(planet, num)
    state_curr = state_prev
    steps = 3
    for n in range(steps):
        state_next = step(state_prev, state_curr, n * num.dt, planet, num)
        state_prev, state_curr = state_curr, state_next
    assert not (jnp.isnan(state_curr.zeta).any() or jnp.isnan(state_curr.T).any())
