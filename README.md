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

- Zonal wind **u**: min ≈ −68.6 m/s, max ≈ 70.1 m/s
- Meridional wind **v**: min ≈ −76.5 m/s, max ≈ 69.8 m/s
- Temperature **T**: min ≈ 140 K, max ≈ 755 K
- Pressure **p**: min ≈ 4.5×10² Pa (aloft), max ≈ 6.4×10⁷ Pa (surface)

Per-level temperature standard deviation (K) at the final step:

```
level 00: 21.4
level 01: 21.0
level 02: 21.3
level 03: 21.7
level 04: 21.6
level 05: 21.0
level 06: 21.2
level 07: 21.2
level 08: 21.5
level 09: 21.3
level 10: 21.7
level 11: 21.3
level 12: 21.4
level 13: 21.6
level 14: 21.2
level 15: 21.4
level 16: 21.5
level 17: 21.2
level 18: 21.2
level 19: 21.2
level 20: 21.2
level 21: 21.2
level 22: 21.1
level 23: 21.4
level 24: 21.3
level 25: 21.3
level 26: 21.1
level 27: 21.2
level 28: 20.9
level 29: 20.6
level 30: 21.0
level 31: 20.7
level 32: 20.7
level 33: 21.1
level 34: 21.2
level 35: 20.9
level 36: 21.0
level 37: 21.0
level 38: 21.0
level 39: 20.9
level 40: 20.8
level 41: 20.8
level 42: 20.8
level 43: 20.5
level 44: 20.8
level 45: 20.1
level 46: 20.7
level 47: 20.2
level 48: 20.3
level 49: 20.1
level 50: 20.0
level 51: 19.0
level 52: 19.4
level 53: 19.5
level 54: 19.0
level 55: 19.0
level 56: 17.6
level 57: 16.9
level 58: 15.9
level 59: 14.4
```
