"""T42L60 Venus demo with diurnal heating and Newtonian cooling."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp

from afes_venus_jax.config import default_numerics, default_planet
from afes_venus_jax.state import initial_state_T_profile
from afes_venus_jax.timestep import step
from afes_venus_jax.io_utils import plot_snapshot, write_netcdf_snapshot


def main():
    num = default_numerics()
    planet = default_planet()
    state = initial_state_T_profile(num, planet)
    t = 0.0
    outdir = pathlib.Path("outputs")
    outdir.mkdir(exist_ok=True)
    nsteps = 500
    snapshot_steps = {0, nsteps - 1}
    for n in range(nsteps):
        state = step(state, t, num, planet)
        t += num.dt
        if n in snapshot_steps:
            print(f"step {n}, t={t/86400:.2f} days")
            plot_snapshot(state, t, n, outdir, num, planet)
            write_netcdf_snapshot(state, t, outdir / f"snapshot_{n:05d}.nc", num, planet)
        elif (n + 1) % 50 == 0:
            print(f"step {n}, t={t/86400:.2f} days")
    write_netcdf_snapshot(state, t, outdir / "final.nc", num, planet)
    plot_snapshot(state, t, nsteps, outdir, num, planet)


if __name__ == "__main__":
    main()
