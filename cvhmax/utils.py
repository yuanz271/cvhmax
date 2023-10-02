from typing import Dict, List

import jax.numpy as jnp
import jax.scipy as jsp


def symm(x):
    return .5 * (x + x.T)


def real_representation(c):
    return jnp.block([[c.real, -c.imag], [c.imag, c.real]])


def info_repr(y, C, d, R):
    T = y.shape[0]
    I = C.T @ jnp.linalg.solve(R, C)
    i = C.T @ jnp.linalg.solve(R, y.T - d)
    i = i.T
    I = jnp.tile(I, (T, 1, 1))
    return i, I


def conjtrans(x):
    """
    Conjugate transpose
    """
    return jnp.conjugate(x.T)


def gamma(x):
    """JAX gamma function"""
    return jnp.exp(jsp.special.gammaln(x))


def kernel_mask(kernel_spec: Dict):
    """mask for a single kernel"""
    size = kernel_spec['order'] + 1
    M = jnp.zeros(size)  # vector
    M = M.at[0].set(1.)
    return M


def mixture_mask(mixture_spec: List[Dict]):
    return jnp.concatenate([kernel_mask(kernel_spec) for kernel_spec in mixture_spec])


def latent_mask(latent_spec):
    left = jsp.linalg.block_diag(*[mixture_mask(mixture_spec) for mixture_spec in latent_spec])
    right = jnp.zeros_like(left)  # the rest SSM dimensions are for the imaginary part.
    M = jnp.column_stack([left, right])
    return M
