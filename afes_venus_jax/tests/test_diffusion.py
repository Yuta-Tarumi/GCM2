import jax
import jax.numpy as jnp

from afes_venus_jax.config import default_numerics, default_planet
from afes_venus_jax.diffusion import hyperdiffuse
from afes_venus_jax.spharm import _wavenumbers


def test_hyperdiffusion_decay(num):
    planet = default_planet()
    kx, ky = _wavenumbers(num.nlat, num.nlon, planet.a)
    lam_field = ky[:, None] ** 2 + kx[None, :] ** 2
    idx = jnp.unravel_index(jnp.argmax(lam_field), lam_field.shape)
    lam = lam_field[idx]
    mode = jnp.zeros((num.nlat, num.nlon), dtype=complex)
    mode = mode.at[idx].set(1.0 + 0j)
    decayed = hyperdiffuse(mode, num, planet)
    nu = 1.0 / (num.tau_hdiff * (lam ** (num.order_hdiff / 2)))
    expected = mode - nu * (lam ** (num.order_hdiff / 2)) * mode * num.dt
    rel = jnp.linalg.norm(decayed - expected) / jnp.linalg.norm(expected)
    assert rel < 1e-12
