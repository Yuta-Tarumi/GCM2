import jax.numpy as jnp

from afes_venus_jax.examples import t42l60_venus_dry_diurnal as example


def test_diurnal_example_runs_one_step():
    num = example.default_numerics()
    planet = example.default_planet()
    state = example.initial_state_T_profile(num, planet)
    next_state = example.step(state, 0.0, num, planet)

    assert jnp.all(jnp.isfinite(next_state.T))
