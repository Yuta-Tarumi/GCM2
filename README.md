# AFES-Venus-JAX

A simplified AFES-style hydrostatic primitive-equation core for Venus written in JAX. The model advances vorticity–divergence (ζ–D) form on the sphere with a σ (p/ps) Lorenz vertical coordinate, spectral transforms for horizontal operators, Eulerian advection, and ∇⁴ hyperdiffusion. Diurnal solar heating and Newtonian infrared cooling approximate Venusian forcing. A T42L60 example writes figure snapshots at start/end of the 500-step demo integration. The semi-implicit gravity-wave treatment is temporarily disabled while investigating excessive layerwise temperature variance.

## Equation set
The original intent was to integrate the full hydrostatic primitive equations without shallow-water reduction. The current code paths, however, only include horizontal advection of vorticity, divergence, and temperature plus simple Newtonian cooling/diurnal heating. A diagnostic hydrostatic geopotential feeds the horizontal pressure-gradient force into the divergence tendency, and vertical layers share information through diffusive coupling. The semi-implicit Helmholtz step for fast gravity waves is currently turned off during debugging, so the run uses a purely explicit leapfrog update.

## Semi-implicit solve (disabled)
Linear gravity-wave coupling between divergence, temperature, and surface pressure is implemented with a semi-implicit Helmholtz solve. A reference temperature drawn from the state sets an equivalent depth `c² = R Tref`; divergence and surface pressure share the implicit gravity-wave damping, and temperature receives the corresponding compressional heating. Off-centering uses `alpha`, and the solve operates directly on every spectral coefficient. This step is temporarily bypassed in `afes_venus_jax.timestep.step` pending investigation of large per-level temperature variance.

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
- Zonal wind **u**: min ≈ −641 m/s, max ≈ 567 m/s
- Meridional wind **v**: min ≈ −912 m/s, max ≈ 832 m/s
- Temperature **T**: min ≈ 170 K, max ≈ 725 K
- Pressure **p**: min ≈ 3.3×10³ Pa (aloft), max ≈ 8.6×10⁶ Pa (surface)

## Why the pressure gradient and winds blow up
- **Surface-pressure equation lacks advection:** the surface-pressure tendency uses only the column-mean divergence (`lnps_dot = -mean(div)`) with no horizontal advection or filtering, so any column convergence piles mass up locally while nothing transports it away. That allows `ps` to diverge by orders of magnitude until the hard log-pressure clamp is hit, producing large latitudinal gradients that then feed the hydrostatic pressure field on every level.【F:afes_venus_jax/tendencies.py†L56-L87】【F:afes_venus_jax/timestep.py†L32-L57】
- **Semi-implicit gravity-wave coupling is disabled:** the Helmholtz step that normally couples divergence, temperature, and surface pressure is bypassed; the model integrates everything explicitly and only applies a weak Robert–Asselin filter. Without the implicit damping, fast gravity-wave modes freely amplify pressure gradients and winds when the explicit step becomes marginally stable.【F:afes_venus_jax/timestep.py†L14-L57】
- **Horizontal derivatives ignore spherical geometry:** gradient, Laplacian, and Helmholtz operators use plain FFT wavenumbers on the latitude–longitude grid rather than true spherical harmonics. Metrics such as 1/`cos(lat)` are only partially handled, so the pressure-gradient force and advection terms are mis-scaled—especially at high latitudes—letting u and v attain unrealistically large values.【F:afes_venus_jax/spharm.py†L1-L57】【F:afes_venus_jax/tendencies.py†L34-L70】

