import jax
import jax.numpy as jnp

from afes_venus_jax.config import default_planet, default_numerics
from afes_venus_jax.implicit import si_matrices, si_solve
from afes_venus_jax.state import initial_state_T_profile


def test_si_stable_linear():
    num = default_numerics()
    planet = default_planet()
    mats = si_matrices(num, planet)
    state = initial_state_T_profile(num, planet)
    rhs = (state.zeta, state.div, state.T, state.lnps)
    zeta, div, T, lnps = si_solve(state, rhs, num, planet, mats)
    assert jnp.all(jnp.isfinite(div))
    assert jnp.all(jnp.isfinite(T))
    assert jnp.all(jnp.isfinite(lnps))
