"""I/O utilities for snapshots."""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from afes_venus_jax.grid import gaussian_grid
from afes_venus_jax.spharm import invert_laplacian, synthesis_spec_to_grid, uv_from_psi_chi


def write_netcdf_snapshot(state, t: float, path: str, num, planet):
    grid = gaussian_grid(num.nlat, num.nlon)
    zeta_g = np.asarray(jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.zeta))
    div_g = np.asarray(jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.div))
    T_g = np.asarray(jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.T))
    lnps_g = np.asarray(synthesis_spec_to_grid(state.lnps, num.nlat, num.nlon))
    ds = xr.Dataset(
        {
            "zeta": (("level", "lat", "lon"), zeta_g),
            "div": (("level", "lat", "lon"), div_g),
            "T": (("level", "lat", "lon"), T_g),
            "lnps": (("lat", "lon"), lnps_g),
        },
        coords={"level": np.arange(num.L), "lat": np.asarray(grid.lats), "lon": np.asarray(grid.lons)},
    )
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def plot_snapshot(state, t: float, step_idx: int, outdir: str, num, planet):
    import jax

    grid = gaussian_grid(num.nlat, num.nlon)
    levels = [0, 15, 30, 45, num.L - 1]
    u_panels = []
    v_panels = []
    T_panels = []
    p_panels = []
    psi_hat = jax.vmap(lambda z: invert_laplacian(z, num.nlat, num.nlon, planet.a))(state.zeta)
    chi_hat = jax.vmap(lambda d: invert_laplacian(d, num.nlat, num.nlon, planet.a))(state.div)
    for k in levels:
        zeta = synthesis_spec_to_grid(state.zeta[k], num.nlat, num.nlon)
        div = synthesis_spec_to_grid(state.div[k], num.nlat, num.nlon)
        u, v = uv_from_psi_chi(psi_hat[k], chi_hat[k], num.nlat, num.nlon, planet.a)
        T = synthesis_spec_to_grid(state.T[k], num.nlat, num.nlon)
        ps = jnp.exp(synthesis_spec_to_grid(state.lnps, num.nlat, num.nlon))
        u_panels.append(np.asarray(u))
        v_panels.append(np.asarray(v))
        T_panels.append(np.asarray(T))
        p_panels.append(np.asarray(ps))
    fig, axes = plt.subplots(len(levels), 4, figsize=(12, 10), constrained_layout=True)
    for i, k in enumerate(levels):
        for j, (field, title, cmap) in enumerate(
            [
                (u_panels[i], f"u [m/s] (level {k})", "coolwarm"),
                (v_panels[i], f"v [m/s] (level {k})", "coolwarm"),
                (T_panels[i], f"T [K] (level {k})", "inferno"),
                (p_panels[i], f"p [Pa] (level {k})", "viridis"),
            ]
        ):
            ax = axes[i, j]
            pcm = ax.pcolormesh(np.asarray(grid.lons), np.asarray(grid.lats), field, shading="auto", cmap=cmap)
            ax.set_title(title)
            fig.colorbar(pcm, ax=ax, shrink=0.75)
    outdir_path = pathlib.Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir_path / f"snapshot_step_{step_idx:05d}.png", dpi=150)
    plt.close(fig)
