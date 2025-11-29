# Venus diurnal demo run (1000 steps)

- Command: `python -m afes_venus_jax.examples.t42l60_venus_dry_diurnal`.
- Outputs: NetCDF and PNG snapshots for steps 0 and 999 plus `final.nc` and a final PNG under `outputs/`. Pressure (`p`) and geopotential (`Phi`) remain written on every model level for hydrostatic checks.
- Physics: nonlinear vorticity/divergence advection on every level, diurnal solar heating centred near 60 km with a cosine day–night mask, and multi-layer Newtonian cooling; semi-implicit vertical coupling and hyperdiffusion are active with `dt=600 s`.
- Initial condition: vertically sheared zonal flow that ramps from 0 m/s at the surface to ~140 m/s near 70 km at the equator, tapered sharply toward the poles (retrograde polar speeds stay within ~60 m/s in magnitude) atop the Venus temperature profile. The zonal pattern follows a high-power cosine profile so that winds are strongest at the equator and nearly vanish at the poles.
- Basic diagnostics from this run (domain-wide extrema):
  - Step 0: u ≈ −58.7 to 141 m/s, v ≈ 0 to 0 m/s, T ≈ 170 to 725 K, p ≈ 3.31×10³ to 8.63×10⁶ Pa.
  - Step 999: u ≈ −641 to 567 m/s, v ≈ −912 to 832 m/s, T ≈ 170 to 725 K, p ≈ 3.3×10³ to 8.63×10⁶ Pa.
  - Step 1000: u ≈ −641 to 567 m/s, v ≈ −912 to 832 m/s, T ≈ 170 to 725 K, p ≈ 3.3×10³ to 8.63×10⁶ Pa.

The strengthened equatorial jet plus explicit pressure/geopotential outputs highlight the vertical stratification (hydrostatic) and the latitudinal wind shear under the applied diurnal heating and Newtonian cooling while preserving mass over the 6.94-day integration.
