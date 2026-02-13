from functools import partial

import numpy as np
from jax import random as jrnd, vmap
import jax.numpy as jnp
import chex
import numpy.testing as npt

from cvhmax import utils
from cvhmax.utils import real_repr, symm, conjtrans, norm_loading, trial_info_repr


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
        params = utils.lbfgs_solve(
            params, partial(poisson_nell, y=y, m=x), max_iter=15000
        )
        print(params)
        print(C, d)


def test_filter_array(capsys):
    S = 10
    T = 50
    D = 5
    y = jrnd.normal(jrnd.key(0), shape=(S, T, D))
    ymask = jrnd.bernoulli(jrnd.key(1), shape=(S, T))

    filtered = utils.filter_array(y, ymask)
    chex.assert_shape(filtered, (ymask.sum(), D))


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------


def test_real_repr_structure(rng):
    """real_repr of a complex matrix has [[Re, -Im], [Im, Re]] block layout."""
    # 1x1 case
    a, b = 3.0, -2.0
    C = jnp.array([[a + b * 1j]])
    R = real_repr(C)
    expected = jnp.array([[a, -b], [b, a]])
    npt.assert_allclose(np.asarray(R), np.asarray(expected), atol=1e-14)

    # 2x2 case
    vals = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    C2 = jnp.array(vals)
    R2 = real_repr(C2)
    assert R2.shape == (4, 4)
    npt.assert_allclose(np.asarray(R2[:2, :2]), np.asarray(C2.real), atol=1e-14)
    npt.assert_allclose(np.asarray(R2[:2, 2:]), -np.asarray(C2.imag), atol=1e-14)
    npt.assert_allclose(np.asarray(R2[2:, :2]), np.asarray(C2.imag), atol=1e-14)
    npt.assert_allclose(np.asarray(R2[2:, 2:]), np.asarray(C2.real), atol=1e-14)


def test_real_repr_preserves_eigenvalues(rng):
    """Eigenvalues of real_repr(C) are conjugate pairs of C's eigenvalues."""
    vals = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    C = jnp.array(vals)
    eigs_c = np.sort_complex(np.asarray(jnp.linalg.eigvals(C)))
    eigs_r = np.sort_complex(np.asarray(jnp.linalg.eigvals(real_repr(C))))

    # real_repr doubles dimension; eigenvalues appear as conjugate pairs
    expected = np.sort_complex(np.concatenate([eigs_c, eigs_c.conj()]))
    npt.assert_allclose(eigs_r, expected, atol=1e-10)


def test_symm_produces_symmetric(rng):
    """symm(X) is exactly symmetric."""
    X = jnp.array(rng.standard_normal((5, 5)))
    S = symm(X)
    npt.assert_array_equal(np.asarray(S), np.asarray(S.T))


def test_conjtrans_involution(rng):
    """conjtrans is an involution: applying it twice returns the original."""
    vals = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    X = jnp.array(vals)
    npt.assert_allclose(np.asarray(conjtrans(conjtrans(X))), np.asarray(X), atol=1e-14)


def test_norm_loading_unit_rows(rng):
    """norm_loading(C, axis=0) normalises each row to unit norm."""
    C = jnp.array(rng.standard_normal((10, 3)))
    Cn = norm_loading(C)
    row_norms = np.asarray(jnp.linalg.norm(Cn, axis=1))
    npt.assert_allclose(row_norms, 1.0, atol=1e-5)


def test_trial_info_repr_analytic(rng):
    """trial_info_repr matches hand-computed j = C^T R^{-1}(y-d), J = C^T R^{-1} C.

    Note: trial_info_repr does ``y.T - d`` which requires ``d`` to have shape
    ``(N, 1)`` for broadcasting with ``y.T`` of shape ``(N, T)``.
    """
    T, N, L = 50, 8, 3
    C = jnp.array(rng.standard_normal((N, L)))
    d = jnp.array(rng.standard_normal((N, 1)))  # column vector for broadcast
    R = jnp.eye(N) * 0.5
    y = jnp.array(rng.standard_normal((T, N)))
    ymask = jnp.ones(T)

    j, J = trial_info_repr(y, ymask, C, d, R)

    # Expected
    Rinv = jnp.linalg.inv(R)
    j_exp = (C.T @ Rinv @ (y.T - d)).T  # (T, L)
    J_exp = C.T @ Rinv @ C  # (L, L)
    J_exp_tiled = jnp.tile(J_exp, (T, 1, 1))

    npt.assert_allclose(np.asarray(j), np.asarray(j_exp), atol=1e-10)
    npt.assert_allclose(np.asarray(J), np.asarray(J_exp_tiled), atol=1e-10)
