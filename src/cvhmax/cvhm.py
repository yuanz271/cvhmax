from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
import secrets

import jax
from jax import Array, NamedSharding, numpy as jnp, vmap
from jax.sharding import PartitionSpec as P
from jax.scipy.linalg import block_diag
import chex

from .cvi import CVI, Gaussian
from .utils import real_repr, symm, cho_inv, training_progress, to_device
from .filtering import bifilter, information_filter


@dataclass
class CVHM:
    """Variational CVHM model wrapper for latent state inference and smoothing.

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
    lr : float, default=0.1
        Learning rate for pseudo-observation updates.
    max_iter : int, default=10
        Maximum number of outer EM iterations.
    cvi_iter : int, default=5
        Number of inner CVI smoothing iterations per EM step.

    Attributes
    ----------
    posterior : tuple[Array, Array]
        Posterior mean and covariance. Shapes are
        `(trials, time, latent_dim (K))` and `(trials, time, latent_dim (K), latent_dim (K))`
        after calling :meth:`fit`.
    """

    n_components: int
    dt: float
    kernels: Sequence[Any]
    params: Any = None
    observation: str = "Gaussian"
    lr: float = 0.1
    cvi: type[CVI] = field(init=False, default=Gaussian)
    max_iter: int = 10
    cvi_iter: int = 5
    posterior: tuple[Array, Array] = field(init=False)

    def __post_init__(self):
        """Resolve the CVI subclass for the requested observation model."""
        self.cvi = CVI.registry.get(self.observation, Gaussian)

    def Af(self):
        """Forward transition matrix for the latent SSM.

        Returns
        -------
        Array
            Block-diagonal real-valued transition matrix.
        """
        C = block_diag(*[kernel.Af(self.dt) for kernel in self.kernels])
        return real_repr(C)

    def Qf(self):
        """Forward process noise covariance for the latent SSM.

        Returns
        -------
        Array
            Block-diagonal process noise covariance.
        """
        C = block_diag(*[kernel.Qf(self.dt) for kernel in self.kernels])
        return symm(real_repr(C))

    def Ab(self):
        """Backward transition matrix for the latent SSM.

        Returns
        -------
        Array
            Block-diagonal real-valued transition matrix.
        """
        C = block_diag(*[kernel.Ab(self.dt) for kernel in self.kernels])
        return real_repr(C)

    def Qb(self):
        """Backward process noise covariance for the latent SSM.

        Returns
        -------
        Array
            Block-diagonal process noise covariance.
        """
        C = block_diag(*[kernel.Qb(self.dt) for kernel in self.kernels])
        return symm(real_repr(C))

    def Q0(self):
        """Stationary prior covariance of the latent process.

        Returns
        -------
        Array
            Block-diagonal stationary covariance.
        """
        C = block_diag(*[kernel.K(0.0) for kernel in self.kernels])
        return symm(real_repr(C))

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
            M = M.at[i, offset].set(1.0)
            offset += kernel.nple

        return M

    def fit(self, y: Array, valid_y: Array | None = None, *, random_state=None):
        """Fit the CVHM model to observations using CVI-EM.

        Parameters
        ----------
        y : Array
            Observations shaped `(trials, time, features)` or `(time, features)`.
        valid_y : Array, optional
            Binary mask matching `y` that flags observed entries. Missing values
            default to all ones when omitted.
        random_state : int | None, optional
            Seed used for initialization. Drawn from `secrets` when absent.

        Returns
        -------
        CVHM
            Fitted instance for chaining.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from cvhmax.cvhm import CVHM
        >>> from cvhmax.hm import HidaMatern
        >>> y = jnp.asarray(...)  # (trials, time, features)
        >>> valid_y = jnp.ones_like(y[..., 0], dtype=jnp.uint8)
        >>> kernels = [HidaMatern(order=0) for _ in range(2)]
        >>> model = CVHM(n_components=2, dt=1.0, kernels=kernels, observation="Gaussian")
        >>> model.fit(y, valid_y=valid_y, random_state=0)
        """
        if valid_y is None:
            valid_y = jnp.ones(y.shape[:-1], dtype=jnp.uint)

        if y.ndim == 2:
            y = jnp.expand_dims(y, 0)
            valid_y = jnp.expand_dims(valid_y, 0)

        chex.assert_equal_shape_prefix((y, valid_y), 2)

        if random_state is None:
            random_state = secrets.randbits(32)

        params = self.params = self.cvi.initialize_params(
            y,
            valid_y,
            self.n_components,
            random_state=random_state,
            params=self.params,
        )

        Af = self.Af()
        Qf = self.Qf()
        Ab = self.Ab()
        Qb = self.Qb()
        Q0 = self.Q0()

        Pf = cho_inv(Qf)
        Pb = cho_inv(Qb)

        # >>> Make stationary distribution
        n_trials = jnp.size(y, 0)
        n_bins = jnp.size(y, 1)
        L = Af.shape[0]
        z0 = jnp.zeros(L)
        Z0 = cho_inv(Q0)
        z0 = jnp.tile(z0, (n_trials, 1))
        Z0 = jnp.tile(Z0, (n_trials, 1, 1))
        # <<<

        # >>> Make dummy variables
        z = jnp.zeros((n_trials, n_bins, L))
        Z = jnp.zeros((n_trials, n_bins, L, L))
        m = jnp.zeros((n_trials, n_bins, self.n_components))
        V = jnp.zeros((n_trials, n_bins, self.n_components, self.n_components))
        # <<<

        n_devices = len(jax.devices())
        mesh = jax.make_mesh((n_devices,), ("batch",))
        sharding = NamedSharding(mesh, P("batch"))
        y, valid_y, z, Z, m, V = to_device((y, valid_y, z, Z, m, V), sharding)

        M = self.latent_mask()

        # Initialize pseudo-observations in latent space
        jl, Jl = vmap(
            self.cvi.initialize_info, in_axes=(None, 0, 0)
        )(params, y, valid_y)

        smooth_batch = vmap(
            lambda jk, Jk, zk0, Zk0: bifilter(jk, Jk, zk0, Zk0, Af, Pf, Ab, Pb)
        )
        fwd_batch = vmap(
            lambda jk, Jk, zk0, Zk0: information_filter(
                (zk0, Zk0), (jk, Jk), Af, Pf
            )
        )

        def em_step(iter, carry):
            params, _, _, jl, Jl, *_ = carry

            # Refresh pseudo-obs from current params.  For conjugate
            # (Gaussian) readouts the pseudo-observations are a
            # deterministic function of the readout parameters.  For
            # non-conjugate (Poisson) readouts this provides a warm
            # restart.
            jl, Jl = vmap(
                self.cvi.initialize_info, in_axes=(None, 0, 0)
            )(params, y, valid_y)

            # Forward-filter warm-up: lift the per-bin pseudo-obs to
            # state space, run a forward information filter, project
            # the *predicted* moments back to latent space, and refine
            # the pseudo-obs.  This provides a sequentially coherent
            # initialisation (owned by CVHM, not CVI) that replaces
            # the forward-filter pass previously inside
            # Poisson.initialize_info.  For conjugate readouts the
            # update is idempotent.
            j_w, J_w = lift(jl, Jl, M)
            zp, Zp, _, _ = fwd_batch(j_w, J_w, z0, Z0)
            m_w, V_w = project(zp, Zp, M)
            jl, Jl = self.cvi.update_pseudo(
                params, y, valid_y, m_w, V_w, jl, Jl, 1.0
            )

            # CVI iterations: CVI ↔ filtering via CVHM bridge
            def cvi_step(i, carry_cvi):
                jl, Jl = carry_cvi
                # Lift latent → state
                j, J = lift(jl, Jl, M)
                # Filter in state space
                z, Z = smooth_batch(j, J, z0, Z0)
                # Project state → latent
                m, V = project(z, Z, M)
                # CVI update in latent space
                jl, Jl = self.cvi.update_pseudo(
                    params, y, valid_y, m, V, jl, Jl, self.lr
                )
                return jl, Jl

            jl, Jl = jax.lax.fori_loop(0, self.cvi_iter, cvi_step, (jl, Jl))

            # Final smooth after CVI converges
            j, J = lift(jl, Jl, M)
            z, Z = smooth_batch(j, J, z0, Z0)
            m, V = project(z, Z, M)

            # M-step: update observation model
            params, nell = self.cvi.update_readout(params, y, valid_y, m, V)

            return params, z, Z, jl, Jl, m, V, nell

        with training_progress() as pbar:
            task_id = pbar.add_task("Training", total=self.max_iter, nell=jnp.nan)

            def step(i, carry):
                carry = em_step(i, carry)
                *_, nell = carry
                jax.debug.callback(
                    lambda step_i, x: pbar.update(
                        task_id,
                        completed=int(step_i) + 1,
                        nell=float(x),
                    ),
                    i,
                    nell,
                    ordered=True,
                )
                return carry

            carry = (params, z, Z, jl, Jl, m, V, jnp.nan)

            carry = jax.lax.fori_loop(0, self.max_iter, step, carry)

        params, z, Z, jl, Jl, m, V, _ = carry
        self.params = params
        self.latent = (z, Z)
        self.posterior = (m, V)
        return self

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
        >>> valid_y = jnp.ones_like(y[..., 0], dtype=jnp.uint8)
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
        Information vectors shaped ``(trials, time, state_dim (L))``.
    Z : Array
        Information matrices shaped
        ``(trials, time, state_dim (L), state_dim (L))``.
    M : Array
        Selection mask shaped ``(latent_dim (K), state_dim (L))``.

    Returns
    -------
    tuple[Array, Array]
        Posterior means ``(trials, time, latent_dim (K))`` and covariances
        ``(trials, time, latent_dim (K), latent_dim (K))`` in latent space.
    """
    return sde2gp(z, Z, M)


def sde2gp(z: Array, Z: Array, M: Array) -> tuple[Array, Array]:
    """Convert information-form SDE state into GP mean and covariance.

    Parameters
    ----------
    z : Array
        Information vectors shaped `(trials, time, state_dim (L))`.
    Z : Array
        Information matrices shaped `(trials, time, state_dim (L), state_dim (L))`.
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
