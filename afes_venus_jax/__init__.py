"""AFES-Venus style hydrostatic spectral core in JAX.

This lightweight implementation mirrors the structure of a classic
AFES/Venus hydrostatic primitive equation model while staying
self‑contained for unit testing. The code favours clarity and
JAX-friendliness over strict numerical fidelity.
"""

from .config import Config, default_config, fast_config
from .state import ModelState
from .timestep import stepper

__all__ = ["Config", "default_config", "fast_config", "ModelState", "stepper"]
