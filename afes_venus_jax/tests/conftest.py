import sys
from pathlib import Path

import jax
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from afes_venus_jax.config import default_numerics

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", True)


@pytest.fixture(scope="session")
def num():
    return default_numerics()
