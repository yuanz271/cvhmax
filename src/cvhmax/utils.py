from functools import partial
from typing import Callable, Dict, List, Optional

import jax
from jax import numpy as jnp, scipy as jsp
from jaxtyping import Array, Scalar
import optax
import optax.tree_utils as otu


def lbfgs_solve(init_params, fun, max_iter=15000, factr=1e12):
    # argument default values copied from scipy.optimize.fmin_l_bfgs_b
    tol = factr * jnp.finfo(float).eps

    # opt = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(max_backtracking_steps=15))
    opt = optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=15))
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return params, state
    
    def continuing_criterion(carry):
        _, state =  carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err > tol))
    
    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(continuing_criterion, step, init_carry)

    return final_params, final_params


def symm(x):
    return .5 * (x + x.T)


def real_repr(c):
    return jnp.block([[c.real, -c.imag], [c.imag, c.real]])


@jax.jit
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


@jax.jit
def norm_loading(w, axis=0):
    _norm = partial(
            _norm_except_axis,
            norm=partial(jnp.linalg.norm, keepdims=True),
            axis=axis,
        )
    return w / _norm(w)


def _norm_except_axis(v: Array, norm: Callable[[Array], Scalar], axis: Optional[int]):
    if axis is None:
        return norm(v)
    else:
        return jax.vmap(norm, in_axes=axis, out_axes=axis)(v)
