# Venus diurnal demo run (500 steps)

- Command: `python -m afes_venus_jax.examples.t42l60_venus_dry_diurnal`.
- Outputs: NetCDF and PNG snapshots for steps 0 and 499 plus `final.nc` under `outputs/`.
- Physics: nonlinear vorticity/divergence advection on every level, diurnal solar heating centred near 60 km with a cosine day–night mask, and multi-layer Newtonian cooling; semi-implicit vertical coupling and hyperdiffusion are active with `dt=600 s`.
- Initial condition: solid-body-style zonal flow that ramps linearly from 0 m/s at the surface to 100 m/s at 70 km (cosine-latitude dependence, held fixed above 70 km) atop the Venus temperature profile.
- Basic diagnostics from this run:
  - Temperature: first snapshot 170.0–725.29 K (mean 448.31 K); 500th snapshot 170.0–725.29 K (mean 448.32 K).
  - Vorticity/divergence: |ζ|max drifts slightly from 4.27e-05 to 4.22e-05 s⁻¹; |∇·v|max remains machine-zero because the initial solid-body wind is non-divergent and forcing does not inject divergence here.
  - Mass/energy proxies: mean surface pressure holds at 9.20e6 Pa to machine precision, and mean kinetic energy softens from 1.62e2 to 1.55e2 m²/s² over 500 steps.

The stronger vertically sheared solid-body wind produces finite but bounded evolution under the applied diurnal heating and Newtonian cooling while preserving mass.
