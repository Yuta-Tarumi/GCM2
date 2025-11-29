"""T42L60 Venus dry spin-up demo."""
from __future__ import annotations

import time

import jax

from afes_venus_jax.config import default_planet, default_numerics
from afes_venus_jax.state import make_initial_state
from afes_venus_jax.timestep import step
from afes_venus_jax.io_utils import write_snapshot


def main():
    planet = default_planet()
    num = default_numerics()
    state_prev = make_initial_state(planet, num)
    state_curr = state_prev

    sim_hours = 12
    steps = int(sim_hours * 3600 / num.dt)
    t0 = time.time()
    for n in range(steps):
        t = n * num.dt
        state_next = step(state_prev, state_curr, t, planet, num)
        state_prev, state_curr = state_curr, state_next
        if n % int(3 * 3600 / num.dt) == 0:
            print(f"t={t/3600:.1f} h")
    write_snapshot(state_curr, "venus_spinup.nc", num)
    print(f"Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
