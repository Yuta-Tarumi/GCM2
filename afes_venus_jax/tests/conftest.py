from jax import config as _cfg
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_cfg.update("jax_enable_x64", True)
_cfg.update("jax_disable_jit", True)
