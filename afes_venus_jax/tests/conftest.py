"""Pytest configuration for faster, jit-free unit runs."""

import sys
from pathlib import Path

import jax
import pytest


# Ensure repository root is importable before loading project modules.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afes_venus_jax import config


# Avoid costly XLA compilation in the unit suite.
jax.config.update("jax_disable_jit", True)


@pytest.fixture(scope="session")
def fast_cfg() -> config.Config:
    """Lightweight configuration for quick tests."""

    return config.fast_config()
