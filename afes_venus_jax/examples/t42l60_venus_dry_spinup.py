"""T42L60 Venus dry spin-up demo."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from afes_venus_jax import Config, default_config
from afes_venus_jax.grid import expand_grid, gaussian_grid
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
    times = jnp.arange(1, nsteps + 1, dtype=float) * cfg.dt

    for istep, t in enumerate(times, start=1):
        state, _ = step_fn(state, t)
        if istep % 10 == 0:
            log_snapshot(state, cfg, istep)
            plot_snapshot(state, cfg, istep)

    lnps_grid = jnp.real(jnp.fft.ifft2(state.lnps * (cfg.nlat * cfg.nlon)))
    print("Final mass mean (Pa):", jnp.mean(jnp.exp(lnps_grid)))
    print("Max |lnps|:", jnp.max(jnp.abs(lnps_grid)))


def log_snapshot(state: ModelState, cfg: Config, istep: int) -> None:
    """Print a diagnostic snapshot every few steps."""

    u_grid, v_grid, T_grid, p_grid = snapshot_fields(state, cfg)

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


def snapshot_fields(state: ModelState, cfg: Config):
    """Convert spectral fields to grid space for diagnostics and plotting."""

    psi, chi = psi_chi_from_zeta_div(state.zeta, state.div, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)

    u_grid = jnp.real(u)
    v_grid = jnp.real(v)
    T_grid = jnp.real(synthesis_spec_to_grid(state.T, cfg))
    lnps_grid = jnp.real(jnp.fft.ifft2(state.lnps * (cfg.nlat * cfg.nlon)))
    p_grid = jnp.exp(lnps_grid)

    return u_grid, v_grid, T_grid, p_grid


def plot_snapshot(state: ModelState, cfg: Config, istep: int, outdir: str | Path = "figures") -> None:
    """Plot diagnostic maps every few steps.

    A small set of evenly spaced vertical levels is plotted for the zonal
    wind, meridional wind, and temperature. Surface pressure is shown once on
    the top row. Figures are saved to ``outdir`` with a step-indexed filename.
    """

    u_grid, v_grid, T_grid, p_grid = snapshot_fields(state, cfg)

    lats, lons, _ = gaussian_grid(cfg.nlat, cfg.nlon)
    lat2d, lon2d = expand_grid(lats, lons)
    lat_deg = np.degrees(np.asarray(lat2d))
    lon_deg = np.degrees(np.asarray(lon2d))

    # Pick up to five evenly spaced levels through the column.
    nlevels = min(5, cfg.L)
    levels = np.unique(np.linspace(0, cfg.L - 1, num=nlevels, dtype=int))

    fig, axes = plt.subplots(len(levels), 4, figsize=(20, 3 * len(levels)), constrained_layout=True)
    if len(levels) == 1:
        axes = np.array([axes])

    u_abs = np.max(np.abs(np.asarray(u_grid[levels])))
    v_abs = np.max(np.abs(np.asarray(v_grid[levels])))
    T_min, T_max = np.min(np.asarray(T_grid[levels])), np.max(np.asarray(T_grid[levels]))

    for idx, lev in enumerate(levels):
        plots = [
            (u_grid, axes[idx, 0], "u [m/s]"),
            (v_grid, axes[idx, 1], "v [m/s]"),
            (T_grid, axes[idx, 2], "T [K]"),
        ]

        for data, ax, label in plots:
            grid = np.asarray(data[lev])
            if label.startswith("u"):
                vmin, vmax = -u_abs, u_abs
                cmap = "coolwarm"
            elif label.startswith("v"):
                vmin, vmax = -v_abs, v_abs
                cmap = "coolwarm"
            else:
                vmin, vmax = T_min, T_max
                cmap = "plasma"

            pcm = ax.pcolormesh(lon_deg, lat_deg, grid, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} (level {lev})")
            ax.set_ylabel("Latitude [deg]")
            if idx == len(levels) - 1:
                ax.set_xlabel("Longitude [deg]")
            fig.colorbar(pcm, ax=ax, orientation="vertical")

        pressure_ax = axes[idx, 3]
        if idx == 0:
            pcm = pressure_ax.pcolormesh(lon_deg, lat_deg, np.asarray(p_grid), shading="auto")
            pressure_ax.set_title("p [Pa] (surface)")
            pressure_ax.set_ylabel("Latitude [deg]")
            pressure_ax.set_xlabel("Longitude [deg]")
            fig.colorbar(pcm, ax=pressure_ax, orientation="vertical")
        else:
            pressure_ax.axis("off")

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"snapshot_{istep:04d}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
