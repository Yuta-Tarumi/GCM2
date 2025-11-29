"""T42L60 Venus demo with diurnal heating and Newtonian cooling."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp

from afes_venus_jax.config import default_numerics, default_planet
from afes_venus_jax.state import initial_state_T_profile
from afes_venus_jax.timestep import step
from afes_venus_jax.io_utils import (
    plot_snapshot,
    state_extrema,
    temperature_std_by_level,
    write_netcdf_snapshot,
)


def main():
    num = default_numerics()
    planet = default_planet()
    state = initial_state_T_profile(num, planet)
    t = 0.0
    outdir = pathlib.Path("outputs")
    outdir.mkdir(exist_ok=True)
    nsteps = 500
    snapshot_steps = {0, nsteps - 1}
    snapshot_steps.add(nsteps)

    def log_extrema(step_idx: int, extrema: dict[str, tuple[float, float]]):
        units = {"u": "m/s", "v": "m/s", "T": "K", "p": "Pa"}
        print(f"  extrema @ step {step_idx}:")
        for key in ("u", "v", "T", "p"):
            vmin, vmax = extrema[key]
            print(f"    {key}: min {vmin: .3g} {units[key]}, max {vmax: .3g} {units[key]}")

    def log_temperature_std(step_idx: int):
        std = temperature_std_by_level(state, num)
        print(f"  temperature std @ step {step_idx} (by level, K):")
        for k, sigma in enumerate(std):
            print(f"    level {k:02d}: {float(sigma): .3g}")
    for n in range(nsteps):
        state = step(state, t, num, planet)
        t += num.dt
        if n in snapshot_steps:
            print(f"step {n}, t={t/86400:.2f} days")
            plot_snapshot(state, t, n, outdir, num, planet)
            write_netcdf_snapshot(state, t, outdir / f"snapshot_{n:05d}.nc", num, planet)
            log_extrema(n, state_extrema(state, num, planet))
            log_temperature_std(n)
        elif (n + 1) % 50 == 0:
            print(f"step {n}, t={t/86400:.2f} days")
    write_netcdf_snapshot(state, t, outdir / "final.nc", num, planet)
    plot_snapshot(state, t, nsteps, outdir, num, planet)
    log_extrema(nsteps, state_extrema(state, num, planet))
    log_temperature_std(nsteps)


if __name__ == "__main__":
    main()
