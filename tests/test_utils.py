from functools import partial
import pytest
import numpy as np
from jax import vmap
import jax.numpy as jnp
from cvhmax import utils


def poisson_nell(params, y, m):
    C, d = params
    
    def _nell(y_t, m_t):
        eta = C @ m_t + d
        lam = jnp.exp(eta)
        return jnp.sum(lam - eta * y_t, axis=-1)
    
    return jnp.mean(vmap(_nell)(y, m))


def test_lbfgs(capsys):
    T = 100
    L = 5
    N = 10
    rng = np.random.default_rng()
    x = rng.normal(0, 1, size=(T, L))
    C = rng.uniform(0, 1, size=(N, L))
    d = np.ones((N,))
    
    eta = x @ C.T + d[None, ...]
    lam = np.exp(eta)

    y = rng.poisson(lam)

    with capsys.disabled():
        params = (jnp.array(rng.normal(size=(N, L)) / N), jnp.zeros(N))
        params, _ = utils.lbfgs_solve(params, partial(poisson_nell, y=y, m=x), max_iter=15000) 
        print(params)
        print(C, d)
