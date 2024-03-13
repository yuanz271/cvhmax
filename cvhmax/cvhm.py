from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jaxtyping import Float, Array

from .cvi import Params, observation_estimate
from .utils import info_repr, real_representation, symm
from .filtering import bifilter


@dataclass
class CVHM:
    n_components: int
    dt: float
    kernels: None
    params: Params = None
    likelihood: str = field(default="Gaussian")
    # nat_step: Callable  # natural param
    # m_step: Callable  # M step
    max_iter: int = field(default=10)
    components_: tuple[list[Float[Array, " time latent"]], list[Float[Array, " time latent latent"]]] = field(init=False)

    def __post_init__(self):
        # check
        pass

    def Af(self):
        C = block_diag(*[kernel.Af(self.dt) for kernel in self.kernels])
        return real_representation(C)

    def Qf(self):
        C = block_diag(*[kernel.Qf(self.dt) for kernel in self.kernels])
        return symm(real_representation(C))

    def Ab(self):
        C = block_diag(*[kernel.Ab(self.dt) for kernel in self.kernels])
        return real_representation(C)

    def Qb(self):
        C = block_diag(*[kernel.Qb(self.dt) for kernel in self.kernels])
        return symm(real_representation(C))

    def Q0(self):
        C = block_diag(*[kernel.K(0.0) for kernel in self.kernels])
        return symm(real_representation(C))

    def mask(self):
        # ssm_dim = sum([kernel.ssm_dim for kernel in self.kernels])
        ssm_dim = sum([kernel.nple for kernel in self.kernels])
        M = jnp.zeros((self.n_components, 2 * ssm_dim))
        for i in range(self.n_components):
            M = M.at[i, i * 2].set(1.0)

        return M

    def fit(self, y: list[Float[Array, " time obs"]]):
        # check y
        if not isinstance(y, list):
            y = [y]
        # self.params.populate(y, self.n_factors)
        C = self.params.C
        d = self.params.d
        R = self.params.R

        M = self.mask()
        
        for j in range(self.max_iter):
            Af = self.Af()
            Qf = self.Qf()
            Ab = self.Ab()
            Qb = self.Qb()
            Q0 = self.Q0()

            Pf = jnp.linalg.inv(Qf)
            Pb = jnp.linalg.inv(Qb)

            H = C @ M  # effective emission

            z0 = jnp.zeros(Af.shape[0])
            Z0 = P0 = jnp.linalg.inv(Q0)

            # bidirectional filtering
            info = [
                info_repr(yk, H, d, R) + (z0, Z0) for yk in y
            ]  # Here's the place that optimize the natural parameters
            # TODO: nat_step for CVI per likelihood

            zZ = [
                bifilter(ik, Ik, z0k, Z0k, Af, Pf, Ab, Pb) for ik, Ik, z0k, Z0k in info
            ]

            # to canonical form
            m_and_V = [
                (jnp.linalg.solve(Zk, zk) @ M.T, M @ jnp.linalg.inv(Zk) @ M.T)
                for zk, Zk in zZ
            ]
            m, V = zip(*m_and_V)  # zip(*iterable) is its own inverse

            # learn observation
            # outer loop
            #   inner loop
            #       e_step()
            #       m_step()
            #   h_step()

            C, d, R = observation_estimate(y, m, V)  # m_step

        self.params.C = C
        self.params.d = d
        self.params.R = R

        self.components_ = (m, V)
        return self
    
    def transform(self, y: list[Float[Array, " time obs"]]):
        raise NotImplementedError
    
    def fit_transform(self, y: list[Float[Array, " time obs"]]):
        self.fit(y)
        return self.components_

    def get_params(self):
        raise NotImplementedError

    def set_params(self, params: Params):
        raise NotImplementedError
