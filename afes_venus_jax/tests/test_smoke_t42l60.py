import jax
import jax.numpy as jnp
from afes_venus_jax.state import zeros_state, ModelState
from afes_venus_jax.timestep import step


def test_short_spinup_no_nans(fast_cfg):
    cfg = fast_cfg
    state = zeros_state(cfg)
    key = jax.random.PRNGKey(3)
    perturb = 1e-4 * (jax.random.normal(key, (cfg.nlat, cfg.nlon)) + 1j * 0)
    state = ModelState(state.zeta, state.div, state.T, state.lnps + perturb)

    nsteps = 6
    for _ in range(nsteps):
        state = step(state, cfg)
    assert jnp.all(jnp.isfinite(state.lnps))
    assert jnp.all(jnp.isfinite(state.zeta))
