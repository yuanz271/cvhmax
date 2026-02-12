from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
import secrets

import jax
from jax import Array, NamedSharding, numpy as jnp, vmap
from jax.sharding import PartitionSpec as P
from jax.scipy.linalg import block_diag
import chex

from .cvi import CVI, Gaussian, Params
from .utils import real_repr, symm, training_progress, to_device
from .filtering import bifilter


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
    params : Params | None, optional
        Initial CVI parameter state. Defaults to `None`.
    likelihood : str, default="Gaussian"
        Name of the CVI likelihood registered in `CVI.registry`.
    lr : float, default=0.1
        Learning rate for pseudo-observation updates.
    max_iter : int, default=10
        Maximum number of outer EM iterations.
    cvi_iter : int, default=3
        Number of inner CVI smoothing iterations per EM step.

    Attributes
    ----------
    posterior : tuple[Array, Array]
        Posterior mean and covariance. Shapes are
        `(trials, time, latent_dim)` and `(trials, time, latent_dim, latent_dim)`
        after calling :meth:`fit`.
    """

    n_components: int
    dt: float
    kernels: Sequence[Any]
    params: Params | None = None
    likelihood: str = "Gaussian"
    lr: float = 0.1
    cvi: type[CVI] = field(init=False, default=Gaussian)
    max_iter: int = 10
    cvi_iter: int = 5
    posterior: tuple[Array, Array] = field(init=False)

    def __post_init__(self):
        """Resolve the CVI subclass for the requested likelihood."""
        self.cvi = CVI.registry.get(self.likelihood, Gaussian)

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
        """Construct the block-diagonal latent-to-SSM selection matrix.

        Returns
        -------
        Array
            Mask mapping latent components to real-valued SSM coordinates.
        """
        ssm_dim = sum([kernel.nple for kernel in self.kernels])
        M = jnp.zeros((self.n_components, 2 * ssm_dim))
        for i in range(self.n_components):
            M = M.at[i, i * 2].set(1.0)

        return M

    def fit(self, y: Array, ymask: Array | None = None, *, random_state=None):
        """Fit the CVHM model to observations using CVI-EM.

        Parameters
        ----------
        y : Array
            Observations shaped `(trials, time, features)` or `(time, features)`.
        ymask : Array, optional
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
        >>> ymask = jnp.ones_like(y[..., 0], dtype=jnp.uint8)
        >>> kernels = [HidaMatern(order=0) for _ in range(2)]
        >>> model = CVHM(n_components=2, dt=1.0, kernels=kernels, likelihood="Gaussian")
        >>> model.fit(y, ymask=ymask, random_state=0)
        """
        if ymask is None:
            ymask = jnp.ones(y.shape[:-1], dtype=jnp.uint)

        if y.ndim == 2:
            y = jnp.expand_dims(y, 0)
            ymask = jnp.expand_dims(ymask, 0)

        chex.assert_equal_shape_prefix((y, ymask), 2)

        if random_state is None:
            random_state = secrets.randbits(32)

        params = self.params = self.cvi.initialize_params(
            y, ymask, self.n_components, self.latent_mask(), random_state=random_state
        )

        Af = self.Af()
        Qf = self.Qf()
        Ab = self.Ab()
        Qb = self.Qb()
        Q0 = self.Q0()

        Pf = jnp.linalg.inv(Qf)
        Pb = jnp.linalg.inv(Qb)

        # >>> Make stationary distribution
        n_trials = jnp.size(y, 0)
        n_bins = jnp.size(y, 1)
        L = Af.shape[0]
        z0 = jnp.zeros(L)
        Z0 = jnp.linalg.inv(Q0)
        z0 = jnp.tile(z0, (n_trials, 1))
        Z0 = jnp.tile(Z0, (n_trials, 1, 1))
        # <<<

        # >>> Make dummpy variables
        z = jnp.zeros((n_trials, n_bins, L))
        Z = jnp.zeros((n_trials, n_bins, L, L))
        m = jnp.zeros((n_trials, n_bins, self.n_components))
        V = jnp.zeros((n_trials, n_bins, self.n_components, self.n_components))
        # <<<

        n_devices = len(jax.devices())
        mesh = jax.make_mesh((n_devices,), ("batch",))
        sharding = NamedSharding(mesh, P("batch"))
        y, ymask, z, Z, m, V = to_device((y, ymask, z, Z, m, V), sharding)

        # Initialize information update
        j, J = vmap(self.cvi.initialize_info, in_axes=(None, 0, 0, None, None))(
            params, y, ymask, Af, Qf
        )

        def em_step(iter, carry):
            params, j, J, *_ = carry
            M = params.lmask()

            (z, Z), (j, J) = self.cvi.infer(
                params,
                j,
                J,
                y,
                ymask,
                z0,
                Z0,
                smooth_fun=bifilter,
                smooth_args=(Af, Pf, Ab, Pb),
                cvi_iter=self.cvi_iter,
                lr=self.lr,
            )

            # to canonical form FutureWarning: jnp.linalg.solve: batched 1D solves with b.ndim > 1 are deprecated, and in the future will be treated as a batched 2D solve. Use solve(a, b[..., None])[..., 0] to avoid this warning.
            m, V = sde2gp(z, Z, M)

            params, nell = self.cvi.update_readout(params, y, ymask, m, V)

            return params, z, Z, j, J, m, V, nell

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

            carry = (params, z, Z, j, J, m, V, jnp.nan)

            # for em_it in range(self.max_iter):
            # carry = step(em_it, carry)
            carry = jax.lax.fori_loop(0, self.max_iter, step, carry)

        params, z, Z, j, J, m, V, _ = carry  # type: ignore
        self.params = params
        self.latent = (z, Z)
        self.posterior = (m, V)  # type: ignore
        return self

    def transform(self, y: Array, ymask: Array):
        """Infer latent trajectories for new data.

        Parameters
        ----------
        y : Array
            Observations to transform.
        ymask : Array
            Observation mask aligned with `y`.

        Raises
        ------
        NotImplementedError
            Raised until an out-of-sample transform implementation is provided.
        """
        raise NotImplementedError

    def fit_transform(self, y: Array, ymask: Array) -> Array:
        """Fit the model and return the posterior mean in one call.

        Parameters
        ----------
        y : Array
            Observations to fit.
        ymask : Array
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
        >>> ymask = jnp.ones_like(y[..., 0], dtype=jnp.uint8)
        >>> kernels = [HidaMatern(order=0) for _ in range(2)]
        >>> model = CVHM(n_components=2, dt=1.0, kernels=kernels)
        >>> m = model.fit_transform(y, ymask)
        """
        self.fit(y, ymask)
        return self.posterior[0]


def sde2gp(z: Array, Z: Array, M: Array) -> tuple[Array, Array]:
    """Convert information-form latents into GP mean and covariance.

    Parameters
    ----------
    z : Array
        Information vectors shaped `(trials, time, state_dim)`.
    Z : Array
        Information matrices shaped `(trials, time, state_dim, state_dim)`.
    M : Array
        Latent-to-output mask applied to recover GP marginals.

    Returns
    -------
    tuple[Array, Array]
        Posterior means and covariances induced by the mask `M`.
    """
    m = vmap(lambda zk, Zk: jnp.linalg.solve(Zk, zk[..., None])[..., 0] @ M.T)(z, Z)
    V = vmap(lambda Zk: M @ jnp.linalg.inv(Zk) @ M.T)(Z)
    return m, V
