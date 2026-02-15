import numpy as np
from jax import random as jrnd
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
    pad_trials,
    unpad_trials,
)


def test_filter_array():
    """filter_array keeps exactly the rows where valid_y == 1."""
    S = 10
    T = 50
    D = 5
    y = jrnd.normal(jrnd.key(0), shape=(S, T, D))
    valid_y = jrnd.bernoulli(jrnd.key(1), shape=(S, T))

    filtered = utils.filter_array(y, valid_y)

    n_valid = int(valid_y.sum())
    assert filtered.shape == (n_valid, D)

    # Verify values: manually collect unmasked rows
    expected = y[valid_y.astype(bool)]
    npt.assert_array_equal(np.asarray(filtered), np.asarray(expected))


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
    valid_y = jnp.ones(T)

    j, J = trial_info_repr(y, valid_y, C, d, R)

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
    valid_y = jnp.ones(T).at[jnp.array([0, 3, 7])].set(0.0)

    j, J = trial_info_repr(y, valid_y, C, d, R)

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
# pad_trials / unpad_trials
# ---------------------------------------------------------------------------


class TestPadTrials:
    """Tests for pad_trials and unpad_trials."""

    def _make_trials(self, lengths, N=4, rng=None):
        """Helper: create random per-trial observation arrays."""
        if rng is None:
            rng = np.random.default_rng(0)
        return [jnp.asarray(rng.standard_normal((T_i, N))) for T_i in lengths]

    def test_shapes(self):
        y_list = self._make_trials([300, 500, 250])
        y, valid_y, trial_lengths = pad_trials(y_list)
        assert y.shape == (3, 500, 4)
        assert valid_y.shape == (3, 500)
        assert trial_lengths.shape == (3,)
        npt.assert_array_equal(np.asarray(trial_lengths), [300, 500, 250])

    def test_padded_bins_masked(self):
        y_list = self._make_trials([10, 20])
        y, valid_y, _ = pad_trials(y_list)
        # First trial is shorter; bins 10..19 should be masked
        npt.assert_array_equal(np.asarray(valid_y[0, 10:]), 0)
        # Second trial fills entirely
        npt.assert_array_equal(np.asarray(valid_y[1, :]), 1)

    def test_padded_values_zero(self):
        y_list = self._make_trials([10, 20])
        y, _, _ = pad_trials(y_list)
        npt.assert_array_equal(np.asarray(y[0, 10:]), 0.0)

    def test_no_valid_y_default(self):
        y_list = self._make_trials([5, 8, 3])
        _, valid_y, _ = pad_trials(y_list)
        for i, T_i in enumerate([5, 8, 3]):
            npt.assert_array_equal(np.asarray(valid_y[i, :T_i]), 1)
            npt.assert_array_equal(np.asarray(valid_y[i, T_i:]), 0)

    def test_preserves_original_mask(self):
        y_list = self._make_trials([10, 15])
        # Mark bins 0 and 3 as missing in trial 0
        mask0 = jnp.ones(10, dtype=jnp.uint8).at[0].set(0).at[3].set(0)
        mask1 = jnp.ones(15, dtype=jnp.uint8)
        _, valid_y, _ = pad_trials(y_list, valid_y_list=[mask0, mask1])
        # Original missing values preserved
        assert int(valid_y[0, 0]) == 0
        assert int(valid_y[0, 3]) == 0
        # Other original bins still observed
        assert int(valid_y[0, 1]) == 1
        # Padded bins masked
        npt.assert_array_equal(np.asarray(valid_y[0, 10:]), 0)

    def test_equal_lengths(self):
        y_list = self._make_trials([20, 20, 20])
        y, valid_y, trial_lengths = pad_trials(y_list)
        assert y.shape == (3, 20, 4)
        npt.assert_array_equal(np.asarray(valid_y), 1)
        npt.assert_array_equal(np.asarray(trial_lengths), [20, 20, 20])

    def test_single_trial(self):
        y_list = self._make_trials([30])
        y, valid_y, trial_lengths = pad_trials(y_list)
        assert y.shape == (1, 30, 4)
        npt.assert_array_equal(np.asarray(valid_y), 1)

    def test_unpad_roundtrip(self):
        lengths = [10, 20, 15]
        y_list = self._make_trials(lengths)
        y, _, trial_lengths = pad_trials(y_list)
        recovered = unpad_trials(y, trial_lengths)
        assert len(recovered) == 3
        for orig, rec in zip(y_list, recovered):
            npt.assert_array_equal(np.asarray(rec), np.asarray(orig))

    def test_unpad_tuple(self):
        lengths = [10, 20]
        rng = np.random.default_rng(1)
        K = 2
        m = jnp.asarray(rng.standard_normal((2, 20, K)))
        V = jnp.asarray(rng.standard_normal((2, 20, K, K)))
        recovered = unpad_trials((m, V), jnp.asarray(lengths))
        assert len(recovered) == 2
        # Each element is a tuple of (m_i, V_i)
        m0, V0 = recovered[0]
        assert m0.shape == (10, K)
        assert V0.shape == (10, K, K)
        m1, V1 = recovered[1]
        assert m1.shape == (20, K)
        assert V1.shape == (20, K, K)
        # Values match the original slices
        npt.assert_array_equal(np.asarray(m0), np.asarray(m[0, :10]))
        npt.assert_array_equal(np.asarray(V1), np.asarray(V[1, :20]))
