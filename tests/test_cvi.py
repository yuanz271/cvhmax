import numpy as np
from jax import numpy as jnp
from cvhmax.cvi import Params, poisson_cvi_stats, Poisson
import chex


def dimensions():
    return 10, 5


def test_poisson_cvi_stats():
    N, L = dimensions()

    j = jnp.zeros(L)
    J = -jnp.eye(L)
    y = jnp.zeros(N)
    C = jnp.ones((N, L))
    d = jnp.ones(N)

    poisson_cvi_stats(j, J, y, C, d)


def test_Poisson(capsys):
    T = 20
    N = 10
    L = 5
    rng = np.random.default_rng()
    y = jnp.array(rng.poisson(5, size=(T, N)))
    C = jnp.ones((N, L))
    d = jnp.ones(N)

    A = jnp.eye(L)
    Q = jnp.eye(L)

    params = Params(C=C, d=d, R=None, M=jnp.eye(L))

    j, J = Poisson.init_info(params, y, A, Q)
    chex.assert_shape([j, J], [(T, L), (T, L, L)])

    m = jnp.array(rng.normal(size=(T, L)))
    V = jnp.stack([jnp.eye(L)] * T)
    
    with capsys.disabled():
        chex.assert_equal_shape((y, m, V), dims=0)
        print(f"{y.shape=} {m.shape=}, {V.shape=}")
        params, nell = Poisson.update_readout(params, [y], [m], [V])

        z = jnp.zeros((T, L))
        Z = jnp.stack([jnp.eye(L)] * T)
        chex.assert_equal_shape((z, Z, j, J, y), dims=0)
        chex.assert_shape((z, Z), ((T, L), (T, L, L)))
        j, J = Poisson.update_pseudo(params, y, z, Z, j, J, 0.1)
        chex.assert_shape([j, J], [(T, L), (T, L, L)])
