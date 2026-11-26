from typing import override

from jax import Array, numpy as jnp

from .cvi import CVI, fa_init, initialization_data
from .readout import (
    NONFINITE_VARIANCE,
    Loading,
    LoadingTerms,
    Params,
    ard_fit_conjugate,
    conjugate_solve,
    fitted_alpha,
    penalty,
)
from .utils import sequence_info_repr


class Gaussian(CVI):
    """Linear-Gaussian readout with closed-form updates."""

    @classmethod
    @override
    def update_pseudo(
        cls,
        params: Params,
        y: Array,
        valid_y: Array,
        m: Array,
        V: Array,
        j: Array,
        J: Array,
        lr: float,
    ) -> tuple[Array, Array]:
        """Return unmodified pseudo-observations for Gaussian readouts.

        Gaussian readouts are conjugate so pseudo-observations are fully
        determined by ``initialize_info``.  All arrays are in latent space.

        Returns
        -------
        tuple[Array, Array]
            Same pseudo-observation parameters ``(j, J)``.
        """
        return j, J

    @classmethod
    @override
    def initialize_info(
        cls, params: Params, y: Array, valid_y: Array
    ) -> tuple[Array, Array]:
        """Compute Gaussian observation information in latent space.

        Parameters
        ----------
        params : Params
            Current readout parameter state.
        y : Array
            Observation tensor shaped ``(time, obs_dim)``.
        valid_y : Array
            Observation mask, shape ``(time, obs_dim)``.

        Returns
        -------
        tuple[Array, Array]
            Observation information vectors and matrices in latent space
            with shapes ``(time, latent_dim (K))`` and
            ``(time, latent_dim (K), latent_dim (K))``.
        """
        C = params.C
        d = params.d
        R = params.R
        return sequence_info_repr(y, valid_y, C, d, R)

    supported_loadings = frozenset({"ridge", "ard", "fixed"})

    @classmethod
    def loading_terms(
        cls, y: Array, valid_y: Array, m: Array, V: Array
    ) -> LoadingTerms:
        """Conjugate representation of the Gaussian readout; see :class:`LoadingTerms`.

        Arguments are flattened over sequences and time. The bias is index 0.
        """
        design = jnp.concatenate((jnp.ones((m.shape[0], 1), dtype=m.dtype), m), axis=1)
        outer = design[:, :, None] * design[:, None, :]
        outer = outer.at[:, 1:, 1:].add(V)
        observed = valid_y.astype(m.dtype)
        return LoadingTerms(
            precision=jnp.einsum("tn,tij->nij", observed, outer),
            linear=jnp.einsum("tn,ti->ni", jnp.where(valid_y, y, 0.0), design),
            counts=jnp.sum(observed, axis=0),
            squared=jnp.sum(jnp.where(valid_y, y**2, 0.0), axis=0),
        )

    @classmethod
    @override
    def solve_ridge(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """Closed-form ridge solve; the loadings are conjugate to the Gaussian."""
        terms = cls.loading_terms(y, valid_y, m, V)
        penalty_diagonal = penalty(settings["alpha"], m.shape[-1])
        return conjugate_solve(terms, penalty_diagonal, params, y, valid_y, m, V), jnp.nan

    @classmethod
    @override
    def solve_ard(
        cls, settings: dict, params: Params, y: Array, valid_y: Array, m: Array, V: Array
    ) -> tuple[Params, float | Array]:
        """Fit Gaussian loadings against the exact conditional evidence."""
        terms = cls.loading_terms(y, valid_y, m, V)
        return ard_fit_conjugate(
            terms, params, y, valid_y, m, V,
            alpha=fitted_alpha(params),
            alpha_shape=settings["alpha_shape"],
            alpha_rate=settings["alpha_rate"],
            iterations=settings["iterations"],
        )

    @classmethod
    @override
    def readout_failure(
        cls, loading: Loading, params: Params, score: float | Array
    ) -> Array:
        """Add the check on the fitted observation variances."""
        variance = jnp.diag(params.R)
        bad_variance = jnp.any(~jnp.isfinite(variance) | (variance <= 0))
        return jnp.where(
            bad_variance,
            NONFINITE_VARIANCE,
            super().readout_failure(loading, params, score),
        )

    @classmethod
    @override
    def check_params(cls, params: Params, n_features: int) -> None:
        """Validate a caller-supplied diagonal observation covariance."""
        R = jnp.asarray(params.R)
        if R.shape != (n_features, n_features):
            raise ValueError("Gaussian R must have shape (features, features)")
        diagonal = jnp.diag(R)
        if not bool(jnp.all(jnp.isfinite(diagonal) & (diagonal > 0))) or not bool(
            jnp.all(R == jnp.diag(diagonal))
        ):
            raise ValueError("Gaussian R must be a positive diagonal covariance")

    @classmethod
    @override
    def check_ard_fit(cls, y: Array, valid_y: Array, n_components: int) -> None:
        """ARD needs more observations than readout parameters per feature."""
        counts = jnp.sum(valid_y, axis=(0, 1))
        minimum = n_components + 1
        if not bool(jnp.all(counts > minimum)):
            raise ValueError(
                "ARD requires every feature to be observed at more than "
                f"n_components + 1 = {minimum} time steps; "
                f"minimum observed count is {int(jnp.min(counts))}"
            )

    @classmethod
    @override
    def initialize_params(
        cls,
        y: Array,
        valid_y: Array,
        n_factors: int,
        *,
        random_state: int,
        penalty: Array | None = None,
        params: Params | None = None,
    ) -> Params:
        """Initialise Gaussian readout via FA.

        Parameters
        ----------
        y : Array
            Observation tensor shaped ``(sequences, time, obs_dim)``.
        valid_y : Array
            Observation mask aligned with ``y``.
        n_factors : int
            Number of latent factors.
        random_state : int
            Seed used for factor analysis.
        penalty : Array, optional
            Loading penalty supplied by the fit. Unused here: the
            Gaussian start is data-driven and is refined by the selected
            loading kind.
        params : Params or None, optional
            If provided, returned as-is (skipping initialisation).

        Returns
        -------
        Params
            Initial Gaussian readout parameters.
        """
        if params is not None:
            return params

        _, _, filled = initialization_data(y, valid_y)
        _, C, d = fa_init(filled, n_factors, random_state)
        return Params(C=C, d=d, R=jnp.eye(y.shape[-1]))
