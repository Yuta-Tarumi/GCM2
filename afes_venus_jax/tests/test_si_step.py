import jax
import jax.numpy as jnp
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import step


def test_semi_implicit_stability(fast_cfg):
    cfg = fast_cfg
    state = zeros_state(cfg)
    key = jax.random.PRNGKey(2)
    perturb = 1e-3 * (jax.random.normal(key, (cfg.nlat, cfg.nlon)) + 1j * 0)
    state = state.__class__(state.zeta, state.div, state.T, state.lnps + perturb)
    for _ in range(6):
        state = step(state, cfg)
        assert jnp.all(jnp.isfinite(state.lnps))
        assert jnp.all(jnp.isfinite(state.zeta))
