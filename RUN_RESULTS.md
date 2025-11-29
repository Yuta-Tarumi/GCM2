# Venus diurnal demo run (500 steps)

- Command: `python -m afes_venus_jax.examples.t42l60_venus_dry_diurnal`.
- Outputs: NetCDF and PNG snapshots for steps 0 and 499 plus `final.nc` under `outputs/`. Pressure (`p`) and geopotential (`Phi`) are now written on every model level for hydrostatic checks.
- Physics: nonlinear vorticity/divergence advection on every level, diurnal solar heating centred near 60 km with a cosine day–night mask, and multi-layer Newtonian cooling; semi-implicit vertical coupling and hyperdiffusion are active with `dt=600 s`.
- Initial condition: vertically sheared zonal flow that ramps from 0 m/s at the surface to ~140 m/s near 70 km at the equator, tapered sharply toward the poles (retrograde polar speeds stay within ~60 m/s in magnitude) atop the Venus temperature profile. The zonal pattern follows a high-power cosine profile so that winds are strongest at the equator and nearly vanish at the poles.
- Basic diagnostics from this run:
  - Temperature: first snapshot 170.0–725.29 K (mean 448.31 K); 500th snapshot 170.0–725.29 K (mean 448.32 K).
  - Vorticity/divergence: |ζ|max stays O(10⁻⁶) s⁻¹ (2.67e-06 → 2.54e-06); |∇·v|max remains numerically zero because the initialization is divergence-free and forcing does not inject divergence here.
  - Mass/energy proxies: mean surface pressure holds at 9.20e6 Pa with level pressures dropping from ~8.6e6 Pa at the bottom to ~3.3e3 Pa at the top; mean kinetic energy eases from 1.86e3 to 1.68e3 m²/s² over 500 steps.
  - Winds: zonal-mean u at the equator is ~2 m/s at the surface and ~139→132 m/s near 70 km, while polar magnitudes are under 1 m/s at the surface and ~58 m/s near 70 km with opposite sign, keeping the profile focused toward low latitudes.

The strengthened equatorial jet plus explicit pressure/geopotential outputs highlight the vertical stratification (hydrostatic) and the latitudinal wind shear under the applied diurnal heating and Newtonian cooling while preserving mass.
