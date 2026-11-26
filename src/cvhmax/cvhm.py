import math
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, NamedTuple

import jax
from jax import Array, vmap
from jax import numpy as jnp
from jax.scipy.linalg import block_diag

from .cvi import CVI, Gaussian
from .readout import (
    FAILURE_MESSAGES,
    READOUT_OK,
    Params,
    loading_settings,
    penalty,
    resolve_alpha,
)
from .serialization import portable_config
from .filtering import bifilter, information_filter
from .hm import _dynamics_from_covariances
from .utils import cho_inv, real_repr, symm, training_progress


class _ScaledDynamics(NamedTuple):
    """Correlation-scaled state-space quantities for one kernel."""

    stationary: Array
    forward: Array
    forward_noise: Array
    backward: Array
    backward_noise: Array


class _InferenceContext(NamedTuple):
    Af: Array
    Pf: Array
    Ab: Array
    Pb: Array
    z0: Array
    Z0: Array
    M: Array


class _FitState(NamedTuple):
    """EM loop state with scalar counters and flags."""

    iteration: int
    params: Any
    patience: int | Array
    converged: bool
    failure: int | Array


def _prepare_observations(y: Array, valid_y: Array | None) -> tuple[Array, Array]:
    """Broadcast the observation mask and add a sequence axis when needed."""
    y = jnp.asarray(y)
    if valid_y is None:
        valid_y = jnp.ones_like(y, dtype=bool)
    else:
        valid_y = jnp.asarray(valid_y, dtype=bool)
        if valid_y.shape == y.shape[:-1]:
            valid_y = valid_y[..., None]
        valid_y = jnp.broadcast_to(valid_y, y.shape)
    if y.ndim == 2:
        y, valid_y = y[None, ...], valid_y[None, ...]
    return jnp.where(valid_y, y, 0.0), valid_y


def _relative_change(value_new: Array, value_old: Array) -> Array:
    """Normalized change between successive readout parameters.

    Uses the Frobenius norm for matrices and the 2-norm for vectors,
    normalized by ``max(1, ||reference||)``.
    """
    denom = jnp.maximum(jnp.linalg.norm(value_old), 1.0)
    return jnp.linalg.norm(value_new - value_old) / denom


def readout_change(params_new: Params, params_old: Params, *, ard: bool = False) -> Array:
    """Return the largest normalized readout change.

    Ridge compares the loading Gram matrices modulo rotations, using a reduced
    QR decomposition instead of feature-by-feature matrices. ARD also compares
    the loadings in their current basis and log precisions: rotations change
    the column-wise prior and hence relevance, even when the Gram is unchanged.

    Parameters
    ----------
    params_new, params_old : Params
        Successive readout parameters.
    ard : bool, default=False
        Include basis-sensitive loading and precision changes.

    Returns
    -------
    Array
        Largest readout or relevance change.
    """
    new, old = params_new.C, params_old.C
    k = old.shape[1]
    _, r = jnp.linalg.qr(jnp.concatenate((new - old, new + old), axis=1), mode="reduced")
    diff, total = r[:, :k], r[:, k:]
    gram_delta = 0.5 * (diff @ total.T + total @ diff.T)
    scale = (jnp.linalg.norm(new) + jnp.linalg.norm(old)) * jnp.maximum(
        1.0, jnp.linalg.norm(old)
    )
    delta_C = jnp.linalg.norm(gram_delta) / jnp.where(scale > 0, scale, 1.0)
    change = jnp.maximum(
        jnp.maximum(delta_C, _relative_change(params_new.d, params_old.d)),
        _relative_change(params_new.R, params_old.R),
    )
    if ard:
        alpha_new, alpha_old = params_new.alpha, params_old.alpha
        if alpha_new is None or alpha_old is None:
            raise ValueError(
                "ARD convergence needs fitted loading precisions in both states"
            )
        change = jnp.maximum(change, _relative_change(new, old))
        change = jnp.maximum(
            change, jnp.max(jnp.abs(jnp.log(alpha_new) - jnp.log(alpha_old)))
        )
    return change


def _validated_integer(value: Any, name: str, *, minimum: int) -> int:
    """Validate and normalize an integer configuration field."""
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


@dataclass
class CVHM:
    """Variational CVHM model wrapper for latent state inference and smoothing.

    Sequences have independent latent trajectories and share the kernels,
    time step, and readout. Feature indices identify the same variables across
    sequences.

    Parameters
    ----------
    n_components : int
        Number of latent components to infer.
    dt : float
        Time step used to discretize the latent SDE.
    kernels : Sequence[Any]
        Sequence of kernel objects providing SSM parameters.
    params : optional
        Initial CVI parameter state. Defaults to ``None``.
    observation : str, default="Gaussian"
        Name of the CVI observation model registered in `CVI.registry`.
    loading : str, default="ridge"
        How the readout loadings are fitted: ``"ridge"`` holds a fixed
        precision, ``"ard"`` fits one precision per latent from conditional
        evidence, and ``"fixed"`` leaves supplied parameters untouched. The
        observation model decides which kinds it supports; see
        `CVI.supported_loadings`.
    loading_config : dict or None, default=None
        Overrides for the loading kind's defaults, for example
        ``{"alpha": 0.5}`` or ``{"iterations": 20}``.
    lr : float, default=1.0
        Step size of the CVI update of the pseudo-observations. The fixed point
        does not depend on it; it only sets how fast the inner loop gets there.
    max_iter : int, default=200
        Maximum number of outer EM iterations.
    cvi_iter : int, default=20
        Fixed number of inner CVI smoothing iterations per EM step. The built-in
        Gaussian readout requires no refinement, so 0 is sufficient for it.
    tol : float, default=0.05
        Convergence tolerance on the readout-parameter change.
    min_iter : int, default=2
        Minimum number of completed outer iterations before a stop is allowed.
    convergence_patience : int, default=2
        Number of consecutive passing outer iterations required for convergence.

    Attributes
    ----------
    posterior : tuple[Array, Array]
        Posterior mean and covariance. Shapes are
        `(sequences, time, latent_dim (K))` and `(sequences, time, latent_dim (K), latent_dim (K))`
        after calling :meth:`fit`.

    Raises
    ------
    ValueError
        If the observation model does not support the loading kind, a fit setting
        is invalid, or ``loading_config`` contains invalid values or keys.
    """

    n_components: int
    dt: float
    kernels: Sequence[Any]
    params: Any = None
    observation: str = "Gaussian"
    loading: str = "ridge"
    loading_config: dict | None = None
    lr: float = 1.0
    cvi: type[CVI] = field(init=False, default=Gaussian)
    max_iter: int = 200
    cvi_iter: int = 20
    tol: float = 0.05
    min_iter: int = 2
    convergence_patience: int = 2
    posterior: tuple[Array, Array] = field(init=False)

    def __post_init__(self):
        """Resolve the selected CVI subclass and validate fit settings."""
        self.cvi = CVI.registry[self.observation]
        if self.loading not in self.cvi.supported_loadings:
            raise ValueError(
                f"loading {self.loading!r} is not supported by observation "
                f"model {self.observation!r}"
            )
        self._loading_settings()  # fail at construction, not at the first fit
        tol = float(self.tol)
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError(f"tol must be finite and strictly positive, got {self.tol!r}")
        self.tol = tol
        self.max_iter = _validated_integer(self.max_iter, "max_iter", minimum=1)
        self.cvi_iter = _validated_integer(self.cvi_iter, "cvi_iter", minimum=0)
        self.min_iter = _validated_integer(self.min_iter, "min_iter", minimum=1)
        self.convergence_patience = _validated_integer(
            self.convergence_patience, "convergence_patience", minimum=1
        )
        if self.min_iter > self.max_iter:
            raise ValueError(
                f"min_iter ({self.min_iter}) must not exceed max_iter ({self.max_iter})"
            )
        if self.params is not None:
            self.cvi.check_params(self.params, self.params.C.shape[0])

    def _loading_settings(self) -> dict:
        """Settings of the current loading kind, read from the live configuration."""
        return loading_settings(self.loading, self.loading_config, self.n_components)

    def _scaled_kernel_dynamics(self, tau):
        """Return correlation-scaled, Cholesky-stabilized dynamics."""
        dynamics = []
        for kernel in self.kernels:
            scale = kernel.state_scale()
            K0 = kernel.K(0.0)
            Kt = kernel.K(tau)
            K0 = scale[:, None] * K0 * scale[None, :]
            Kt = scale[:, None] * Kt * scale[None, :]
            raw = _dynamics_from_covariances(K0, Kt)
            dynamics.append(
                _ScaledDynamics(
                    stationary=K0,
                    forward=raw.forward,
                    forward_noise=raw.forward_noise,
                    backward=raw.backward,
                    backward_noise=raw.backward_noise,
                )
            )
        return dynamics

    def _matrices_from_dynamics(self, dynamics):
        """Assemble real block matrices from one prepared dynamics bundle."""
        Af = real_repr(block_diag(*[item.forward for item in dynamics]))
        Qf = symm(real_repr(block_diag(*[item.forward_noise for item in dynamics])))
        Ab = real_repr(block_diag(*[item.backward for item in dynamics]))
        Qb = symm(real_repr(block_diag(*[item.backward_noise for item in dynamics])))
        Q0 = symm(real_repr(block_diag(*[item.stationary for item in dynamics])))
        return Af, Qf, Ab, Qb, Q0

    def Af(self):
        """Forward transition matrix for the normalized latent SSM.

        Returns
        -------
        Array
            Block-diagonal real-valued transition matrix.
        """
        return self._matrices_from_dynamics(self._scaled_kernel_dynamics(self.dt))[0]

    def Qf(self):
        """Forward process noise covariance for the latent SSM.

        Returns
        -------
        Array
            Block-diagonal process noise covariance.
        """
        return self._matrices_from_dynamics(self._scaled_kernel_dynamics(self.dt))[1]

    def Ab(self):
        """Backward transition matrix for the latent SSM.

        Returns
        -------
        Array
            Block-diagonal real-valued transition matrix.
        """
        return self._matrices_from_dynamics(self._scaled_kernel_dynamics(self.dt))[2]

    def Qb(self):
        """Backward process noise covariance for the latent SSM.

        Returns
        -------
        Array
            Block-diagonal process noise covariance.
        """
        return self._matrices_from_dynamics(self._scaled_kernel_dynamics(self.dt))[3]

    def Q0(self):
        """Stationary prior covariance of the latent process.

        Returns
        -------
        Array
            Block-diagonal stationary covariance.
        """
        return self._matrices_from_dynamics(self._scaled_kernel_dynamics(0.0))[4]

    def latent_mask(self):
        """Construct the block-diagonal selection matrix from latent to state space.

        Returns
        -------
        Array
            Mask of shape ``(latent_dim (K), state_dim (L))`` selecting the
            GP-value coordinate of each kernel in the real-valued SDE state.
        """
        ssm_dim = sum(kernel.nple for kernel in self.kernels)
        M = jnp.zeros((self.n_components, 2 * ssm_dim))
        offset = 0
        for i, kernel in enumerate(self.kernels):
            # The normalized state is x_z = D x. Recover f(t) from the
            # function coordinate with D^{-1}; the imaginary coordinate is
            # not observed for the real GP.
            M = M.at[i, offset].set(1.0 / kernel.state_scale()[0])
            offset += kernel.nple

        return M

    def _inference_context(self, n_sequences: int) -> _InferenceContext:
        """Prepare the fixed dynamics and stationary state for inference."""
        dynamics = self._scaled_kernel_dynamics(self.dt)
        Af, Qf, Ab, Qb, Q0 = self._matrices_from_dynamics(dynamics)
        L = Af.shape[0]
        return _InferenceContext(
            Af, cho_inv(Qf), Ab, cho_inv(Qb),
            jnp.tile(jnp.zeros(L), (n_sequences, 1)),
            jnp.tile(cho_inv(Q0), (n_sequences, 1, 1)),
            self.latent_mask(),
        )

    def _infer_state(self, params, y, valid_y, context):
        """Infer SDE information and projected GP moments for fixed readout."""
        Af, Pf, Ab, Pb, z0, Z0, M = context
        jl, Jl = vmap(self.cvi.initialize_info, in_axes=(None, 0, 0))(
            params, y, valid_y
        )
        smooth_batch = vmap(
            lambda jk, Jk, zk0, Zk0: bifilter(jk, Jk, zk0, Zk0, Af, Pf, Ab, Pb)
        )
        fwd_batch = vmap(
            lambda jk, Jk, zk0, Zk0: information_filter(
                (zk0, Zk0), (jk, Jk), Af, Pf
            )
        )
        j, J = lift(jl, Jl, M)
        zp, Zp, _, _ = fwd_batch(j, J, z0, Z0)
        m, V = project(zp, Zp, M)
        jl, Jl = self.cvi.update_pseudo(params, y, valid_y, m, V, jl, Jl, 1.0)

        def cvi_step(_, carry):
            jl, Jl = carry
            j, J = lift(jl, Jl, M)
            z, Z = smooth_batch(j, J, z0, Z0)
            m, V = project(z, Z, M)
            return self.cvi.update_pseudo(params, y, valid_y, m, V, jl, Jl, self.lr)

        jl, Jl = jax.lax.fori_loop(0, self.cvi_iter, cvi_step, (jl, Jl))
        j, J = lift(jl, Jl, M)
        z, Z = smooth_batch(j, J, z0, Z0)
        m, V = project(z, Z, M)
        return z, Z, m, V

    def fit(self, y: Array, valid_y: Array | None = None, *, random_state=None):
        """Fit the CVHM model to observations using CVI-EM.

        Parameters
        ----------
        y : Array
            Observations shaped `(sequences, time, features)` or `(time, features)`.
        valid_y : Array, optional
            Boolean mask broadcastable to `y`, or a time-step mask shaped
            `(sequences, time)` (or `(time,)` for one sequence). Defaults to all observed.
        random_state : int | None, optional
            Seed used for initialization. Drawn from `secrets` when absent.

        Returns
        -------
        CVHM
            Fitted instance for chaining.

        Raises
        ------
        ValueError
            If a feature is never observed, if the observation model rejects the
            loading kind for this data through `CVI.check_ard_fit`, or if
            ARD's ``params.alpha`` is not finite and strictly positive.
        FloatingPointError
            If Gaussian variances or ARD precisions become non-positive or
            non-finite, or ARD evidence becomes non-finite.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from cvhmax.cvhm import CVHM
        >>> from cvhmax.hm import HidaMatern
        >>> y = jnp.asarray(...)  # (sequences, time, features)
        >>> valid_y = jnp.ones_like(y, dtype=bool)
        >>> kernels = [HidaMatern(order=0) for _ in range(2)]
        >>> model = CVHM(n_components=2, dt=1.0, kernels=kernels, observation="Gaussian")
        >>> model.fit(y, valid_y=valid_y, random_state=0)
        """
        y, valid_y = _prepare_observations(y, valid_y)
        if not bool(jnp.all(jnp.any(valid_y, axis=(0, 1)))):
            raise ValueError("Every feature must be observed at least once during fitting")
        fits_ard = self.loading == "ard"
        if fits_ard:
            self.cvi.check_ard_fit(y, valid_y, self.n_components)
        settings = self._loading_settings()
        if random_state is None:
            random_state = secrets.randbits(32)
        if self.params is not None:
            self.cvi.check_params(self.params, y.shape[-1])
        params = self.cvi.initialize_params(
            y,
            valid_y,
            self.n_components,
            random_state=random_state,
            penalty=penalty(jnp.asarray(settings["alpha"]), self.n_components),
            params=self.params,
        )
        alpha = resolve_alpha(self.loading, settings, params.alpha, self.n_components)
        params = Params(C=params.C, d=params.d, R=params.R, alpha=alpha)
        context = self._inference_context(y.shape[0])

        with training_progress() as pbar:
            task_id = pbar.add_task(
                "Training", total=self.max_iter, nell=jnp.nan, nell_display="n/a"
            )

            initial = _FitState(0, params, 0, False, jnp.asarray(READOUT_OK, dtype=jnp.int32))

            def cond(state):
                return (
                    (state.iteration < self.max_iter)
                    & ~state.converged
                    & (state.failure == READOUT_OK)
                )

            def body(state):
                _, _, m, V = self._infer_state(state.params, y, valid_y, context)
                updated, nell = self.cvi.solve_readout(
                    self.loading, settings, state.params, y, valid_y, m, V
                )
                metric = jnp.where(
                    state.iteration > 0,
                    readout_change(
                        updated, state.params, ard=fits_ard
                    ),
                    jnp.inf,
                )
                iteration = state.iteration + 1
                passing = (
                    jnp.isfinite(metric) & (metric <= self.tol) & (iteration >= self.min_iter)
                )
                patience = jnp.where(passing, state.patience + 1, 0)
                converged = passing & (patience >= self.convergence_patience)
                failure = self.cvi.readout_failure(self.loading, updated, nell).astype(
                    jnp.int32
                )
                jax.debug.callback(
                    lambda step_i, x: pbar.update(
                        task_id,
                        completed=int(step_i) + 1,
                        nell=float(x),
                        nell_display=(
                            f"{float(x):.3f}" if math.isfinite(float(x)) else "n/a"
                        ),
                    ),
                    state.iteration,
                    nell,
                    ordered=True,
                )
                return _FitState(iteration, updated, patience, converged, failure)

            result = jax.lax.while_loop(cond, body, initial)

        if int(result.failure) != READOUT_OK:
            raise FloatingPointError(FAILURE_MESSAGES[int(result.failure)])
        z, Z, m, V = self._infer_state(result.params, y, valid_y, context)
        self.params = result.params
        self.latent = (z, Z)
        self.posterior = (m, V)
        self.converged_ = bool(result.converged)
        self.n_iter_ = int(result.iteration)
        return self

    def get_config(self) -> dict:
        """Return a JSON-compatible configuration dict.

        Returns
        -------
        dict
            Configuration with keys ``n_components``, ``dt``, ``lr``, ``max_iter``,
            ``cvi_iter``, ``tol``, ``min_iter``, ``convergence_patience``,
            ``observation``, ``loading``, ``loading_config``, and ``kernels``
            (each kernel via its ``get_config()``). ``loading_config`` is
            emitted as finite JSON primitives, so a numpy or JAX ``alpha`` array
            becomes a list while keeping its logical shape.
        """
        return {
            "n_components": int(self.n_components),
            "dt": float(self.dt),
            "lr": float(self.lr),
            "max_iter": int(self.max_iter),
            "cvi_iter": int(self.cvi_iter),
            "tol": float(self.tol),
            "min_iter": int(self.min_iter),
            "convergence_patience": int(self.convergence_patience),
            "observation": str(self.observation),
            "loading": str(self.loading),
            "loading_config": portable_config(self.loading_config),
            "kernels": [k.get_config() for k in self.kernels],
        }

    @classmethod
    def from_config(cls, config: dict) -> "CVHM":
        """Reconstruct a CVHM from a configuration dict.

        Parameters
        ----------
        config : dict
            Configuration dict with keys ``n_components``, ``dt``, ``lr``,
            ``max_iter``, ``cvi_iter``, ``tol``, ``min_iter``,
            ``convergence_patience``, ``observation``, and ``kernels``.

        Returns
        -------
        CVHM
            Reconstructed model without posterior or fitted params.

        Raises
        ------
        ValueError
            If required keys are missing, observation is unknown, or kernel
            config is invalid.
        """
        from .hm import HidaMatern

        required = (
            "n_components", "dt", "lr", "max_iter", "cvi_iter",
            "tol", "min_iter", "convergence_patience",
            "observation", "loading", "loading_config", "kernels",
        )
        for key in required:
            if key not in config:
                raise ValueError(f"CVHM.from_config missing required key: {key!r}")
        kernels = [HidaMatern.from_config(k) for k in config["kernels"]]
        return cls(
            n_components=config["n_components"],
            dt=config["dt"],
            kernels=kernels,
            observation=config["observation"],
            loading=config["loading"],
            loading_config=config["loading_config"],
            lr=config["lr"],
            max_iter=config["max_iter"],
            cvi_iter=config["cvi_iter"],
            tol=config["tol"],
            min_iter=config["min_iter"],
            convergence_patience=config["convergence_patience"],
        )

    def save(self, path: str | PathLike[str]) -> None:
        """Save model configuration and fitted readout parameters.

        Posterior and latent inference caches are intentionally excluded.
        The archive is a data-only format; custom kernels and observation
        models are not supported by the initial serializer.
        """
        from .serialization import save

        save(self, path)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> "CVHM":
        """Load a model saved by :meth:`save`."""
        from .serialization import load

        return load(path)

    def infer(self, y: Array, valid_y: Array | None = None):
        """Compute a posterior using fitted parameters without refitting.

        Parameters
        ----------
        y : Array
            Observations shaped `(sequences, time, features)` or `(time, features)`.
        valid_y : Array, optional
            Boolean mask broadcastable to `y`, or a time-step mask shaped
            `(sequences, time)` (or `(time,)` for one sequence). Defaults to all observed.

        Returns
        -------
        tuple[Array, Array]
            Posterior mean and covariance in latent space.

        Raises
        ------
        RuntimeError
            If the model has no fitted readout parameters.
        """
        if self.params is None:
            raise RuntimeError("Cannot infer with an unfitted model")
        y, valid_y = _prepare_observations(y, valid_y)
        self.cvi.check_params(self.params, y.shape[-1])
        context = self._inference_context(y.shape[0])
        _, _, m, V = self._infer_state(self.params, y, valid_y, context)
        return m, V

    def transform(self, y: Array, valid_y: Array):
        """Infer latent trajectories for new data.

        Parameters
        ----------
        y : Array
            Observations to transform.
        valid_y : Array
            Observation mask aligned with `y`.

        Raises
        ------
        NotImplementedError
            Raised until an out-of-sample transform implementation is provided.
        """
        raise NotImplementedError

    def fit_transform(self, y: Array, valid_y: Array) -> Array:
        """Fit the model and return the posterior mean in one call.

        Parameters
        ----------
        y : Array
            Observations to fit.
        valid_y : Array
            Observation mask aligned with `y`.

        Returns
        -------
        Array
            Posterior mean of the latent trajectories.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from cvhmax.cvhm import CVHM
        >>> from cvhmax.hm import HidaMatern
        >>> y = jnp.asarray(...)
        >>> valid_y = jnp.ones_like(y, dtype=bool)
        >>> kernels = [HidaMatern(order=0) for _ in range(2)]
        >>> model = CVHM(n_components=2, dt=1.0, kernels=kernels)
        >>> m = model.fit_transform(y, valid_y)
        """
        self.fit(y, valid_y)
        return self.posterior[0]


def lift(j_latent: Array, J_latent: Array, M: Array) -> tuple[Array, Array]:
    """Lift latent-space information to state-space.

    Parameters
    ----------
    j_latent : Array
        Information vectors in latent space, shape ``(..., latent_dim (K))``.
    J_latent : Array
        Information matrices in latent space,
        shape ``(..., latent_dim (K), latent_dim (K))``.
    M : Array
        Selection mask shaped ``(latent_dim (K), state_dim (L))``.

    Returns
    -------
    tuple[Array, Array]
        Information vectors and matrices in state space with trailing
        dimensions ``(state_dim (L),)`` and ``(state_dim (L), state_dim (L))``.
    """
    j = j_latent @ M
    J = M.T @ J_latent @ M
    return j, J


def project(z: Array, Z: Array, M: Array) -> tuple[Array, Array]:
    """Project state-space information posterior to latent-space moments.

    Converts information-form state ``(z, Z)`` to moment-form latent
    ``(m, V)`` by selecting the components indicated by ``M``.

    Parameters
    ----------
    z : Array
        Information vectors shaped ``(sequences, time, state_dim (L))``.
    Z : Array
        Information matrices shaped
        ``(sequences, time, state_dim (L), state_dim (L))``.
    M : Array
        Selection mask shaped ``(latent_dim (K), state_dim (L))``.

    Returns
    -------
    tuple[Array, Array]
        Posterior means ``(sequences, time, latent_dim (K))`` and covariances
        ``(sequences, time, latent_dim (K), latent_dim (K))`` in latent space.
    """
    return sde2gp(z, Z, M)


def sde2gp(z: Array, Z: Array, M: Array) -> tuple[Array, Array]:
    """Convert information-form SDE state into GP mean and covariance.

    Parameters
    ----------
    z : Array
        Information vectors shaped `(sequences, time, state_dim (L))`.
    Z : Array
        Information matrices shaped `(sequences, time, state_dim (L), state_dim (L))`.
    M : Array
        Selection mask shaped `(latent_dim (K), state_dim (L))` mapping
        SDE state coordinates to GP components.

    Returns
    -------
    tuple[Array, Array]
        Posterior means and covariances induced by the mask `M`.
    """
    m = vmap(lambda zk, Zk: jnp.linalg.solve(Zk, zk[..., None])[..., 0] @ M.T)(z, Z)
    V = vmap(vmap(lambda Zk: M @ cho_inv(Zk) @ M.T))(Z)
    return m, V
