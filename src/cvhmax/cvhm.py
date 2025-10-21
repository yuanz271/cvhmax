from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
import secrets

import jax
from jax import Array, numpy as jnp, vmap
from jax.scipy.linalg import block_diag
import chex

from .cvi import CVI, Gaussian, Params
from .utils import real_repr, symm, training_progress
from .filtering import bifilter


@dataclass
class CVHM:
    n_components: int
    dt: float
    kernels: Sequence[Any]
    params: Params | None = None
    likelihood: str = "Gaussian"
    lr: float = 0.1
    cvi: type[CVI] = field(init=False, default=Gaussian)
    max_iter: int = 10
    cvi_iter: int = 3
    posterior: tuple[Array, Array] = field(init=False)

    def __post_init__(self):
        self.cvi = CVI.registry.get(self.likelihood, Gaussian)

    def Af(self):
        C = block_diag(*[kernel.Af(self.dt) for kernel in self.kernels])
        return real_repr(C)

    def Qf(self):
        C = block_diag(*[kernel.Qf(self.dt) for kernel in self.kernels])
        return symm(real_repr(C))

    def Ab(self):
        C = block_diag(*[kernel.Ab(self.dt) for kernel in self.kernels])
        return real_repr(C)

    def Qb(self):
        C = block_diag(*[kernel.Qb(self.dt) for kernel in self.kernels])
        return symm(real_repr(C))

    def Q0(self):
        C = block_diag(*[kernel.K(0.0) for kernel in self.kernels])
        return symm(real_repr(C))

    def latent_mask(self):
        ssm_dim = sum([kernel.nple for kernel in self.kernels])
        M = jnp.zeros((self.n_components, 2 * ssm_dim))
        for i in range(self.n_components):
            M = M.at[i, i * 2].set(1.0)

        return M

    def fit(self, y: Array, ymask: Array | None = None, *, random_state=None):
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
                    lambda x: pbar.update(task_id, advance=1, nell=x), nell
                )
                return carry

            carry = (params, z, Z, j, J, m, V, jnp.nan)

            for em_it in range(self.max_iter):
                carry = step(em_it, carry)
            # carry = jax.lax.fori_loop(0, self.max_iter, step, carry)

        params, z, Z, j, J, m, V, _ = carry  # type: ignore
        self.params = params
        self.latent = (z, Z)
        self.posterior = (m, V)  # type: ignore
        return self

    def transform(self, y: Array, ymask: Array):
        raise NotImplementedError

    def fit_transform(self, y: Array, ymask: Array) -> Array:
        self.fit(y, ymask)
        return self.posterior[0]


def sde2gp(z: Array, Z: Array, M: Array) -> tuple[Array, Array]:
    m = vmap(lambda zk, Zk: jnp.linalg.solve(Zk, zk[..., None])[..., 0] @ M.T)(z, Z)
    V = vmap(lambda Zk: M @ jnp.linalg.inv(Zk) @ M.T)(Z)
    return m, V
