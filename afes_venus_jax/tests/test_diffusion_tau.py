import jax
import jax.numpy as jnp

from afes_venus_jax.config import default_planet, default_numerics
from afes_venus_jax.diffusion import hyperdiffusion_operator


def test_tau_matches():
    num = default_numerics()
    planet = default_planet()
    coeff = hyperdiffusion_operator(num, planet)
    lam_max = ((num.nlat - 1) * num.nlat) / (planet.a ** 2)
    nu = 1.0 / (num.tau_hdiff * (lam_max ** 2))
    decay = -coeff[-1, 0]
    assert jnp.isclose(decay, nu * lam_max ** 2)
