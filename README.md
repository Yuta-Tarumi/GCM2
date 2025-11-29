# AFES-Venus-JAX

A simplified AFES-style hydrostatic primitive-equation core for Venus written in JAX. The model advances vorticity–divergence (ζ–D) form on the sphere with a σ (p/ps) Lorenz vertical coordinate, spectral transforms for horizontal operators, semi-implicit gravity-wave treatment, Eulerian advection, and ∇⁴ hyperdiffusion. Diurnal solar heating and Newtonian infrared cooling approximate Venusian forcing. A T42L60 example writes figure snapshots at start/end of the 500-step demo integration.

## Equation set
The original intent was to integrate the full hydrostatic primitive equations without shallow-water reduction. The current code paths, however, only include horizontal advection of vorticity, divergence, and temperature plus simple Newtonian cooling/diurnal heating. A diagnostic hydrostatic geopotential feeds the horizontal pressure-gradient force into the divergence tendency, and vertical layers share information through diffusive coupling, while a semi-implicit Helmholtz solve damps fast gravity waves. The model remains idealised but now carries the primary linear gravity-wave response of the primitive equations.

## Semi-implicit solve
Linear gravity-wave coupling between divergence, temperature, and surface pressure is handled with a semi-implicit Helmholtz solve. A reference temperature drawn from the state sets an equivalent depth `c² = R Tref`; divergence and surface pressure share the implicit gravity-wave damping, and temperature receives the corresponding compressional heating. Off-centering uses `alpha`, and the solve operates directly on every spectral coefficient.

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
For CUDA-enabled environments, install the JAX wheels compiled for CUDA 12 before the remaining dependencies:
```bash
pip install --upgrade "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```
Run the T42L60 Venus demo (writes PNGs at initialization, step 499, and the final state):
```bash
python -m afes_venus_jax.examples.t42l60_venus_dry_diurnal
```
Figure snapshots show u, v, T, and pressure on selected vertical levels.

## 500-step diagnostic (T42L60 diurnal example)
Running `python -m afes_venus_jax.examples.t42l60_venus_dry_diurnal` for the default 500 steps (≈3.5 model days) produces the following domain-wide extrema at the final step:

- Zonal wind **u**: min ≈ −38 m/s, max ≈ 48 m/s
- Meridional wind **v**: min ≈ −55 m/s, max ≈ 56 m/s
- Temperature **T**: min ≈ 100 K, max ≈ 800 K
- Pressure **p**: min ≈ 4.5×10² Pa (aloft), max ≈ 6.4×10⁷ Pa (surface)
