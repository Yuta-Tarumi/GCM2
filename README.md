# AFES-Venus JAX core

A lightweight hydrostatic primitive-equation spectral core written in JAX. The code follows an AFES-style vorticity–divergence formulation with a sigma-coordinate vertical grid and semi-implicit time stepping for gravity waves. Horizontal transforms use FFT-based spectral surrogates to keep the implementation compact and easily testable in this environment.

## Features
- Hydrostatic primitive-equation variables (ζ, D, temperature, surface pressure) on a Lorenz sigma grid.
- Gaussian grid with pseudo-spectral nonlinear advection and exact metric factors on the sphere surrogate.
- Leapfrog time integration with semi-implicit gravity-wave damping and Robert–Asselin filtering.
- Spectral ∇⁴ hyperdiffusion calibrated to an e-folding time at the truncation wavenumber.
- Diurnal shortwave heating tied to the Venus solar day and height-dependent Newtonian cooling.

## Running

```
pip install -r requirements.txt
pytest -q
python -m afes_venus_jax.examples.t42l60_venus_spinup
```

## Notes
This repository enables x64 in JAX by default and disables JIT inside the unit tests to keep the suite fast. The example run writes a NetCDF snapshot `venus_spinup.nc` after 12 hours of model time.
