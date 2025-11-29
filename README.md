# AFES-Venus-JAX

A simplified AFES-style hydrostatic primitive-equation core for Venus written in JAX. The model advances vorticity–divergence (ζ–D) form on the sphere with a σ (p/ps) Lorenz vertical coordinate, spectral transforms for horizontal operators, semi-implicit gravity-wave treatment, Eulerian advection, and ∇⁴ hyperdiffusion. Diurnal solar heating and Newtonian infrared cooling approximate Venusian forcing. A T42L60 example writes figure snapshots every 10 steps.

## Equation set
The solver integrates the full hydrostatic primitive equations without shallow-water reduction. Vorticity and divergence are prognosed spectrally; temperature and surface pressure share the same representation. Hydrostatic geopotential is obtained using Simmons–Burridge-style integration on the Lorenz grid.

## Semi-implicit solve
Linear gravity-wave coupling between divergence, temperature, and surface pressure is handled with a semi-implicit matrix per total wavenumber ℓ. Off-centering uses `alpha`; the resulting block system is inverted per ℓ and applied to all m.

## Venus constants
Key constants follow Venus observations: radius 6051.8 km, gravity 8.87 m/s², retrograde rotation, cp=1000 J/kg/K, and reference surface pressure ~92 bar. The reference temperature profile ranges from ~730 K at the surface to ~170 K at 120 km altitude.

## Physics
- **Diurnal shortwave heating:** Longitude-dependent term peaks near 60–70 km and follows the moving subsolar point.
- **Newtonian cooling:** Relaxation toward the reference T(z) with altitude-dependent radiative timescales.
- **Hyperdiffusion:** ∇⁴ damping calibrated so the ℓmax mode decays with e-folding time `tau_hdiff`.

## Running
Install dependencies and run tests:
```bash
pip install -r requirements.txt
pytest -q
```
Run the T42L60 Venus demo (writes PNG every 10 steps):
```bash
python -m afes_venus_jax.examples.t42l60_venus_dry_diurnal
```
Figure snapshots show u, v, T, and pressure on selected vertical levels.
