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
from afes_venus_jax.vertical import hydrostatic_geopotential, sigma_levels


def write_netcdf_snapshot(state, t: float, path: str, num, planet):
    grid = gaussian_grid(num.nlat, num.nlon)
    zeta_g = np.asarray(jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.zeta))
    div_g = np.asarray(jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.div))
    T_g = np.asarray(jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.T))
    lnps_g = np.asarray(synthesis_spec_to_grid(state.lnps, num.nlat, num.nlon))
    sigma_full, sigma_half = sigma_levels(num.L)
    ps_g = np.exp(lnps_g)
    p_full = sigma_full[:, None, None] * ps_g[None, :, :]
    Phi_g = np.asarray(hydrostatic_geopotential(T_g, ps_g, sigma_half, planet))
    ds = xr.Dataset(
        {
            "zeta": (("level", "lat", "lon"), zeta_g),
            "div": (("level", "lat", "lon"), div_g),
            "T": (("level", "lat", "lon"), T_g),
            "p": (("level", "lat", "lon"), p_full),
            "Phi": (("level", "lat", "lon"), Phi_g),
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
    sigma_full, sigma_half = sigma_levels(num.L)
    ps = jnp.exp(synthesis_spec_to_grid(state.lnps, num.nlat, num.nlon))
    psi_hat = jax.vmap(lambda z: invert_laplacian(z, num.nlat, num.nlon, planet.a))(state.zeta)
    chi_hat = jax.vmap(lambda d: invert_laplacian(d, num.nlat, num.nlon, planet.a))(state.div)
    for k in levels:
        zeta = synthesis_spec_to_grid(state.zeta[k], num.nlat, num.nlon)
        div = synthesis_spec_to_grid(state.div[k], num.nlat, num.nlon)
        u, v = uv_from_psi_chi(psi_hat[k], chi_hat[k], num.nlat, num.nlon, planet.a)
        T = synthesis_spec_to_grid(state.T[k], num.nlat, num.nlon)
        p_full = sigma_full[k] * ps
        u_panels.append(np.asarray(u))
        v_panels.append(np.asarray(v))
        T_panels.append(np.asarray(T))
        p_panels.append(np.asarray(p_full))
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


def state_extrema(state, num, planet):
    """Return domain-wide extrema for u, v, T, and p across all levels."""

    psi_hat = jax.vmap(lambda z: invert_laplacian(z, num.nlat, num.nlon, planet.a))(state.zeta)
    chi_hat = jax.vmap(lambda d: invert_laplacian(d, num.nlat, num.nlon, planet.a))(state.div)
    u, v = jax.vmap(lambda psi, chi: uv_from_psi_chi(psi, chi, num.nlat, num.nlon, planet.a))(psi_hat, chi_hat)
    T = jax.vmap(lambda x: synthesis_spec_to_grid(x, num.nlat, num.nlon))(state.T)

    ps = jnp.exp(synthesis_spec_to_grid(state.lnps, num.nlat, num.nlon))
    sigma_full, _ = sigma_levels(num.L)
    p = sigma_full[:, None, None] * ps

    def minmax(field):
        return float(field.min()), float(field.max())

    return {"u": minmax(u), "v": minmax(v), "T": minmax(T), "p": minmax(p)}
