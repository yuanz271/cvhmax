from collections.abc import Iterable
from dataclasses import dataclass, field
import secrets
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jaxtyping import Float, Array
from tqdm import trange

from . import cvi
from .cvi import CVI, Params
from .utils import real_repr, symm
from .filtering import bifilter


@dataclass
class CVHM:
    n_components: int
    dt: float
    kernels: None
    params: Params = field(default=None)
    likelihood: str = field(default="Gaussian")
    lr: float = field(default=0.1)
    # nat_step: Callable  # natural param
    # m_step: Callable  # M step
    observation: ClassVar[CVI] = field(init=False)
    max_iter: int = field(default=10)
    cvi_iter: int = field(default=3)
    posterior: tuple[list[Float[Array, " time latent"]], list[Float[Array, " time latent latent"]]] = field(init=False)

    def __post_init__(self):
        self.observation = getattr(cvi, self.likelihood)

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

    def mask(self):
        # ssm_dim = sum([kernel.ssm_dim for kernel in self.kernels])
        ssm_dim = sum([kernel.nple for kernel in self.kernels])
        M = jnp.zeros((self.n_components, 2 * ssm_dim))
        for i in range(self.n_components):
            M = M.at[i, i * 2].set(1.0)

        return M

    def fit(self, y: list[Float[Array, " time obs"]], *, random_state=None):
        # check y
        if not isinstance(y, list):
            y = [y]
        # self.params.populate(y, self.n_factors)
        
        if random_state is None:
            random_state = secrets.randbits(32)

        self.params = self.observation.initialize_params(y, self.n_components, self.mask(), random_state=random_state)

        params = self.params

        Af = self.Af()
        Qf = self.Qf()
        Ab = self.Ab()
        Qb = self.Qb()
        Q0 = self.Q0()

        Pf = jnp.linalg.inv(Qf)
        Pb = jnp.linalg.inv(Qb)

        z0 = jnp.zeros(Af.shape[0])
        Z0 = jnp.linalg.inv(Q0)

        zZ0s = [(z0, Z0) for _ in y]  # stationary distribution
        # Af = Af + 1e-3 * jnp.eye(Af.shape[0])

        jJ = [self.observation.init_info(params, yk, Af, Qf) for yk in y]

        def em_step(iter, carry):
            params, jJ, *_ = carry
            M = params.M

            zZ, jJ = self.observation.cvi(params, jJ, y, zZ0s, smooth_fun=bifilter, smooth_args=(Af, Pf, Ab, Pb), cvi_iter=self.cvi_iter, lr=self.lr)

            # to canonical form FutureWarning: jnp.linalg.solve: batched 1D solves with b.ndim > 1 are deprecated, and in the future will be treated as a batched 2D solve. Use solve(a, b[..., None])[..., 0] to avoid this warning.
            mV = [
                (jnp.linalg.solve(Zk, zk[..., None])[..., 0] @ M.T, M @ jnp.linalg.inv(Zk) @ M.T)
                for zk, Zk in zZ
            ]
            m, V = zip(*mV)  # zip(*iterable) is its own inverse
            
            params = self.observation.update_readout(params, y, m, V)

            return params, jJ, m, V
        
        init_val = em_step(0, (params, jJ, None, None))
        params, jJ, m, V = jax.lax.fori_loop(1, self.max_iter, em_step, init_val)
        
        # for em_it in trange(self.max_iter):

        #     # bidirectional filtering
        #     # info = [
        #     #     info_repr(yk, H, d, R) + (z0, Z0) for yk in y
        #     # ]  # Here's the place that optimize the natural parameters
        #     # TODO: nat_step for CVI per likelihood
            
        #     zZ, jJ = self.observation.cvi(params, jJ, y, zZ0s, smooth_fun=bifilter, smooth_args=(Af, Pf, Ab, Pb), cvi_iter=self.cvi_iter, lr=self.lr)
        #     # zZ0s = [(z[0], Z[0]) for z, Z in zZ]

        #     # to canonical form FutureWarning: jnp.linalg.solve: batched 1D solves with b.ndim > 1 are deprecated, and in the future will be treated as a batched 2D solve. Use solve(a, b[..., None])[..., 0] to avoid this warning.
        #     mV = [
        #         (jnp.linalg.solve(Zk, zk[..., None])[..., 0] @ M.T, M @ jnp.linalg.inv(Zk) @ M.T)
        #         for zk, Zk in zZ
        #     ]
        #     m, V = zip(*mV)  # zip(*iterable) is its own inverse
            
        #     params = self.observation.update_readout(params, y, m, V)

        self.params = params
        self.posterior = (m, V)
        return self
    
    def transform(self, y: list[Float[Array, " time obs"]]):
        raise NotImplementedError
    
    def fit_transform(self, y: list[Float[Array, " time obs"]]):
        self.fit(y)
        return self.posterior[0]
