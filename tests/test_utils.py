from functools import partial

import numpy as np
from jax import random as jrnd, vmap
import jax.numpy as jnp
import chex
import numpy.testing as npt

from cvhmax import utils
from cvhmax.utils import (
    real_repr,
    symm,
    conjtrans,
    norm_loading,
    bin_info_repr,
    trial_info_repr,
    natural_to_moment,
    moment_to_natural,
)


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
    """trial_info_repr matches hand-computed j = C^T R^{-1}(y-d), J = C^T R^{-1} C."""
    T, N, L = 50, 8, 3
    C = jnp.array(rng.standard_normal((N, L)))
    d = jnp.array(rng.standard_normal(N))
    R = jnp.eye(N) * 0.5
    y = jnp.array(rng.standard_normal((T, N)))
    ymask = jnp.ones(T)

    j, J = trial_info_repr(y, ymask, C, d, R)

    # Expected
    Rinv = jnp.linalg.inv(R)
    # Expected: per-bin j_t = C^T R^{-1} (y_t - d), J = C^T R^{-1} C tiled
    j_exp = (C.T @ Rinv @ (y.T - d[:, None])).T  # (T, L)
    J_exp = C.T @ Rinv @ C  # (L, L)
    J_exp_tiled = jnp.tile(J_exp, (T, 1, 1))

    npt.assert_allclose(np.asarray(j), np.asarray(j_exp), atol=1e-10)
    npt.assert_allclose(np.asarray(J), np.asarray(J_exp_tiled), atol=1e-10)


def test_bin_info_repr_mask(rng):
    """Masked bins contribute zero information (j=0, J=0)."""
    N, L = 5, 2
    C = jnp.array(rng.standard_normal((N, L)))
    d = jnp.array(rng.standard_normal(N))
    R = jnp.eye(N) * 0.5
    y = jnp.array(rng.standard_normal(N))

    # Unmasked: normal computation
    j_on, J_on = bin_info_repr(y, jnp.array(1.0), C, d, R)
    assert jnp.any(j_on != 0)
    assert jnp.any(J_on != 0)

    # Masked: zero information
    j_off, J_off = bin_info_repr(y, jnp.array(0.0), C, d, R)
    npt.assert_array_equal(np.asarray(j_off), 0.0)
    npt.assert_array_equal(np.asarray(J_off), 0.0)


def test_trial_info_repr_mask(rng):
    """trial_info_repr zeros out masked bins, keeps unmasked bins."""
    T, N, L = 10, 5, 2
    C = jnp.array(rng.standard_normal((N, L)))
    d = jnp.array(rng.standard_normal(N))
    R = jnp.eye(N) * 0.5
    y = jnp.array(rng.standard_normal((T, N)))

    # Mask out bins 0, 3, 7
    ymask = jnp.ones(T).at[jnp.array([0, 3, 7])].set(0.0)

    j, J = trial_info_repr(y, ymask, C, d, R)

    # Masked bins should be zero
    for t in [0, 3, 7]:
        npt.assert_array_equal(np.asarray(j[t]), 0.0)
        npt.assert_array_equal(np.asarray(J[t]), 0.0)

    # Unmasked bins should match direct computation
    Rinv = jnp.linalg.inv(R)
    J_exp = C.T @ Rinv @ C
    for t in [1, 2, 4, 5, 6, 8, 9]:
        j_exp_t = C.T @ Rinv @ (y[t] - d)
        npt.assert_allclose(np.asarray(j[t]), np.asarray(j_exp_t), atol=1e-10)
        npt.assert_allclose(np.asarray(J[t]), np.asarray(J_exp), atol=1e-10)


# ---------------------------------------------------------------------------
# natural_to_moment / moment_to_natural
# ---------------------------------------------------------------------------


def test_natural_moment_roundtrip_moment_start(rng):
    """moment -> natural -> moment is identity."""
    D = 4
    A = rng.standard_normal((D, D))
    Sigma = jnp.array(A @ A.T + 0.1 * np.eye(D))  # positive definite
    mu = jnp.array(rng.standard_normal(D))

    eta1, eta2 = moment_to_natural(mu, Sigma)
    mu_rec, Sigma_rec = natural_to_moment(eta1, eta2)

    npt.assert_allclose(np.asarray(mu_rec), np.asarray(mu), atol=1e-12)
    npt.assert_allclose(np.asarray(Sigma_rec), np.asarray(Sigma), atol=1e-12)


def test_natural_moment_roundtrip_natural_start(rng):
    """natural -> moment -> natural is identity."""
    D = 3
    A = rng.standard_normal((D, D))
    prec = A @ A.T + 0.1 * np.eye(D)
    eta2 = jnp.array(-0.5 * prec)  # negative definite
    eta1 = jnp.array(rng.standard_normal(D))

    mu, Sigma = natural_to_moment(eta1, eta2)
    eta1_rec, eta2_rec = moment_to_natural(mu, Sigma)

    npt.assert_allclose(np.asarray(eta1_rec), np.asarray(eta1), atol=1e-12)
    npt.assert_allclose(np.asarray(eta2_rec), np.asarray(eta2), atol=1e-12)


def test_natural_moment_1d_analytic():
    """1-D analytic check: Sigma=2, mu=3 -> eta1=1.5, eta2=-0.25."""
    mu = jnp.array([3.0])
    Sigma = jnp.array([[2.0]])

    eta1, eta2 = moment_to_natural(mu, Sigma)
    npt.assert_allclose(np.asarray(eta1), [1.5], atol=1e-14)
    npt.assert_allclose(np.asarray(eta2), [[-0.25]], atol=1e-14)

    mu_rec, Sigma_rec = natural_to_moment(eta1, eta2)
    npt.assert_allclose(np.asarray(mu_rec), [3.0], atol=1e-14)
    npt.assert_allclose(np.asarray(Sigma_rec), [[2.0]], atol=1e-14)


def test_natural_to_moment_matches_sde2gp(rng):
    """natural_to_moment on (z, -0.5*Z) matches sde2gp (without mask M)."""
    from cvhmax.cvhm import sde2gp

    D = 3
    # Build valid information-form parameters
    A = rng.standard_normal((D, D))
    Z = jnp.array(A @ A.T + 0.5 * np.eye(D))  # precision, positive definite
    z = jnp.array(rng.standard_normal(D))

    # sde2gp expects (trials, time, ...) — use identity mask and single trial/time
    M = jnp.eye(D)
    m_sde, V_sde = sde2gp(z[None, None, :], Z[None, None, :, :], M)
    m_sde = m_sde[0, 0]  # strip trial/time dims
    V_sde = V_sde[0, 0]

    # Convert info-form (z, Z) to natural params (z, -0.5*Z) then to moments
    mu, Sigma = natural_to_moment(z, -0.5 * Z)

    npt.assert_allclose(np.asarray(mu), np.asarray(m_sde), atol=1e-12)
    npt.assert_allclose(np.asarray(Sigma), np.asarray(V_sde), atol=1e-12)
