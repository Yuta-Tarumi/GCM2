import jax.numpy as jnp
from afes_venus_jax.state import ModelState
from afes_venus_jax.diffusion import hyperdiffusion


def test_hyperdiffusion_decay(fast_cfg):
    cfg = fast_cfg
    spec = jnp.zeros((cfg.L, cfg.nlat, cfg.nlon), dtype=jnp.complex128)
    spec = spec.at[0, 0, 0].set(1.0 + 0j)
    state = ModelState(spec, spec, spec, jnp.zeros((cfg.nlat, cfg.nlon), dtype=jnp.complex128))
    decayed = hyperdiffusion(state, cfg)
    factor = jnp.abs(decayed.zeta[0, 0, 0])
    assert factor < 1.0
