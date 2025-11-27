"""T42L60 Venus dry spin-up demo."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from afes_venus_jax import Config, default_config
from afes_venus_jax.state import ModelState, zeros_state
from afes_venus_jax.timestep import stepper
from afes_venus_jax.spharm import psi_chi_from_zeta_div, synthesis_spec_to_grid, uv_from_psi_chi


def main():
    cfg = default_config()
    state = zeros_state(cfg)

    # small random perturbations in surface pressure
    key = jax.random.PRNGKey(0)
    noise = 1e-4 * (jax.random.normal(key, (cfg.nlat, cfg.nlon)) + 1j * 0)
    state = ModelState(state.zeta, state.div, state.T, state.lnps + noise)

    step_fn = stepper(cfg)
    nsteps = int(6 * 3600 / cfg.dt)

    for istep in range(1, nsteps + 1):
        state, _ = step_fn(state, None)
        if istep % 10 == 0:
            log_snapshot(state, cfg, istep)

    lnps_grid = jnp.real(jnp.fft.ifft2(state.lnps * (cfg.nlat * cfg.nlon)))
    print("Final mass mean (Pa):", jnp.mean(jnp.exp(lnps_grid)))
    print("Max |lnps|:", jnp.max(jnp.abs(lnps_grid)))


def log_snapshot(state: ModelState, cfg: Config, istep: int) -> None:
    """Print a diagnostic snapshot every few steps."""

    psi, chi = psi_chi_from_zeta_div(state.zeta, state.div, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)

    u_grid = jnp.real(u)
    v_grid = jnp.real(v)
    T_grid = jnp.real(synthesis_spec_to_grid(state.T, cfg))
    lnps_grid = jnp.real(jnp.fft.ifft2(state.lnps * (cfg.nlat * cfg.nlon)))
    p_grid = jnp.exp(lnps_grid)

    def bounds(arr):
        return jnp.min(arr), jnp.max(arr)

    u_min, u_max = bounds(u_grid)
    v_min, v_max = bounds(v_grid)
    T_min, T_max = bounds(T_grid)
    p_min, p_max = bounds(p_grid)

    print(
        f"Step {istep:04d}: "
        f"u[min,max]=({u_min:.3e}, {u_max:.3e}), "
        f"v[min,max]=({v_min:.3e}, {v_max:.3e}), "
        f"T[min,max]=({T_min:.3e}, {T_max:.3e}), "
        f"p[min,max]=({p_min:.3e}, {p_max:.3e})"
    )


if __name__ == "__main__":
    main()
