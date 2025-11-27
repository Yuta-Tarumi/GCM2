"""T42L60 Venus dry spin-up demo."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from afes_venus_jax import Config, default_config
from afes_venus_jax.state import ModelState, zeros_state
from afes_venus_jax.timestep import stepper


def main():
    cfg = default_config()
    state = zeros_state(cfg)

    # small random perturbations in surface pressure
    key = jax.random.PRNGKey(0)
    noise = 1e-4 * (jax.random.normal(key, (cfg.nlat, cfg.nlon)) + 1j * 0)
    state = ModelState(state.zeta, state.div, state.T, state.lnps + noise)

    step_fn = stepper(cfg)
    nsteps = int(6 * 3600 / cfg.dt)
    state, _ = jax.lax.scan(step_fn, state, None, length=nsteps)

    lnps_grid = jnp.real(jnp.fft.ifft2(state.lnps * (cfg.nlat * cfg.nlon)))
    print("Final mass mean (Pa):", jnp.mean(jnp.exp(lnps_grid)))
    print("Max |lnps|:", jnp.max(jnp.abs(lnps_grid)))


if __name__ == "__main__":
    main()
