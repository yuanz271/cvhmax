import numpy as np
from jax import numpy as jnp, vmap
from cvhmax.cvi import Params, poisson_cvi_bin_stats, Poisson
import chex


def dimensions():
    return 10, 5


def test_poisson_cvi_stats():
    N, L = dimensions()

    j = jnp.zeros(L)
    J = -jnp.eye(L)
    y = jnp.zeros(N)
    ymask = jnp.ones(1)
    C = jnp.ones((N, L))
    d = jnp.ones(N)

    poisson_cvi_bin_stats(j, J, y, ymask, C, d)


def test_Poisson(capsys):
    T = 20
    N = 10
    L = 5
    rng = np.random.default_rng()
    y = jnp.array(rng.poisson(5, size=(1, T, N)))
    ymask = jnp.ones((1, T))
    C = jnp.ones((N, L))
    d = jnp.ones(N)

    A = jnp.eye(L)
    Q = jnp.eye(L)

    params = Params(C=C, d=d, R=jnp.zeros((N, N)), M=jnp.eye(L))

    j, J = vmap(Poisson.initialize_info, in_axes=(None, 0, 0, None, None))(params, y, ymask, A, Q)
    chex.assert_shape([j, J], [(1, T, L), (1,T, L, L)])

    m = jnp.array(rng.normal(size=(1, T, L)))
    V = jnp.expand_dims(jnp.tile(jnp.eye(L), (T, 1, 1)), 0)
    
    with capsys.disabled():
        chex.assert_equal_shape((y, m, V), dims=0)
        print(f"{y.shape=} {m.shape=}, {V.shape=}")
        params, nell = Poisson.update_readout(params, y, ymask, m, V)

        z = jnp.zeros((1, T, L))
        Z = jnp.expand_dims(jnp.tile(jnp.eye(L), (T, 1, 1)), 0)
        chex.assert_equal_shape((z, Z, j, J, y), dims=0)

        chex.assert_shape((z, Z), ((1, T, L), (1, T, L, L)))
        j, J = Poisson.update_pseudo(params, y, ymask, z, Z, j, J, 0.1)
        chex.assert_shape([j, J], [(1, T, L), (1, T, L, L)])
