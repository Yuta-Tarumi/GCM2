# AFES-Venus JAX (simplified)

This repository provides a compact, JAX-friendly scaffold of an
AFES/Venus-style hydrostatic spectral core. The focus is on providing a
clean, well-documented layout that mirrors the major components of the
original model while keeping dependencies minimal for testing.

## Running the demo

```bash
python -m afes_venus_jax.examples.t42l60_venus_dry_spinup
```

## Running tests

```bash
pytest -q
```
