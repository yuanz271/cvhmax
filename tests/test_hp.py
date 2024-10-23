from functools import partial

import numpy as np
from jax import config
from jax import numpy as jnp

config.update("jax_enable_x64", True)

from cvhmax import hm, hp
from cvhmax.hm import sample_matern


def test_spectral_loss():
    n = 100
    dt = 1.
    sigma = 1.
    rho = .5
    m = sample_matern(n, dt, sigma, rho)
    m = np.expand_dims(m, 1)
    v = np.ones((n, 1, 1))
    # print(m.shape, v.shape)
    spec = [[{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 1}], [{'sigma': 1., 'rho': 1., 'omega': 0., 'order': 0}, {'sigma': 1., 'rho': 1., 'omega': 1., 'order': 1}]]
    filter = [[{'sigma': True, 'rho': True, 'omega': True, 'order': False}], [{'sigma': True, 'rho': True, 'omega': True, 'order': False}, {'sigma': True, 'rho': True, 'omega': True, 'order': False}]]
    
    paramflat, paramdef, static = hp.spec2vec(spec, filter)
    log_p_s = hp.spectral_loss(paramflat, paramdef, static, hm.spectral_density, m, v, dt)
    print(log_p_s)
    
    arr = jnp.array([jnp.array(1.), jnp.array(2.)])
    print(arr, arr.shape)
    new_spec = hp.whittle(spec, filter, m, v, dt)
    print(new_spec)

# def test_whittle():
#     n = 100
#     dt = 1.
#     sigma = 1.
#     rho = .5
#     m = sample_matern(n, dt, sigma, rho)
#     m = np.expand_dims(m, 1)
#     v = np.ones((n, 1, 1))
#     print(m.shape, v.shape)
#     hyperparams = np.zeros(3)
#     sigma, rho, b = hp.whittle(hyperparams, m, v, dt)
#     print(sigma, rho, b)
