from afes_venus_jax.config import default_planet, default_numerics
from afes_venus_jax.state import make_initial_state
from afes_venus_jax.timestep import step


def test_step_runs(num):
    planet = default_planet()
    state0 = make_initial_state(planet, num)
    state1 = step(state0, state0, 0.0, planet, num)
    assert state1.zeta.shape == state0.zeta.shape
