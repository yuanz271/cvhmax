from collections.abc import Callable
from functools import partial

import jax
from jax import Array, numpy as jnp, scipy as jsp
import optax
import optax.tree_utils as otu
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

EPS = 1e-6


def lbfgs_solve(init_params, fun, max_iter=100, factr=1e12):
    """Approximate bounded LBFGS solver mirroring SciPy defaults.

    Parameters
    ----------
    init_params : PyTree
        Initial parameter values.
    fun : Callable
        Objective function accepting the parameter PyTree.
    max_iter : int, default=100
        Maximum number of optimisation steps.
    factr : float, default=1e12
        Factor controlling the gradient tolerance `factr * eps`.

    Returns
    -------
    PyTree
        Optimised parameter values.
    """
    # argument default values copied from scipy.optimize.fmin_l_bfgs_b
    tol = factr * jnp.finfo(float).eps

    # opt = optax.lbfgs(linesearch=optax.scale_by_backtracking_linesearch(max_backtracking_steps=15))
    opt = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=15)
    )
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err > tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )

    return final_params


def symm(x):
    """Symmetrise a matrix.

    Parameters
    ----------
    x : Array
        Input matrix.

    Returns
    -------
    Array
        Symmetric part `(x + x.T) / 2`.
    """
    return 0.5 * (x + x.T)


def real_repr(c):
    """Convert a complex matrix to its real-valued block representation.

    Parameters
    ----------
    c : Array
        Complex-valued matrix.

    Returns
    -------
    Array
        Real block matrix embedding the complex values.
    """
    return jnp.block([[c.real, -c.imag], [c.imag, c.real]])


def trial_info_repr(
    y: Array, ymask: Array, C: Array, d: Array, R: Array
) -> tuple[Array, Array]:
    """Compute Gaussian observation information per time bin.

    Parameters
    ----------
    y : Array
        Observation tensor.
    ymask : Array
        Observation mask aligned with `y`.
    C : Array
        Observation matrix.
    d : Array
        Bias term.
    R : Array
        Observation covariance.

    Returns
    -------
    tuple[Array, Array]
        Observation information vectors and matrices.
    """
    T = y.shape[0]

    J = C.T @ jnp.linalg.solve(R, C)
    j = C.T @ jnp.linalg.solve(R, y.T - d)
    j = j.T
    J = jnp.tile(J, (T, 1, 1))

    j: Array = jnp.where(jnp.expand_dims(ymask, -1), j, 0)  # broadcastable mask
    # J: Array = jnp.where(jnp.expand_dims(ymask, (-2, -1)), J, 0)

    return j, J


def conjtrans(x):
    """Return the conjugate transpose of a matrix.

    Parameters
    ----------
    x : Array
        Complex-valued matrix.

    Returns
    -------
    Array
        Conjugate-transposed matrix.
    """
    return jnp.conjugate(x.T)


def gamma(x):
    """Evaluate the gamma function using JAX.

    Parameters
    ----------
    x : Array
        Input values.

    Returns
    -------
    Array
        Result of the gamma function applied element-wise.
    """
    return jnp.exp(jsp.special.gammaln(x))


def kernel_mask(kernel_spec: dict):
    """Mask for a single kernel.

    Parameters
    ----------
    kernel_spec : dict
        Kernel hyperparameters containing an `order` entry.

    Returns
    -------
    Array
        One-hot mask selecting the real component.
    """
    size = kernel_spec["order"] + 1
    M = jnp.zeros(size)  # vector
    M = M.at[0].set(1.0)
    return M


def mixture_mask(mixture_spec: list[dict]):
    """Concatenate kernel masks for a latent mixture.

    Parameters
    ----------
    mixture_spec : list[dict]
        Sequence of kernel specifications.

    Returns
    -------
    Array
        Concatenated mask covering all kernels in the mixture.
    """
    return jnp.concatenate([kernel_mask(kernel_spec) for kernel_spec in mixture_spec])


def latent_mask(latent_spec):
    """Construct a block-diagonal mask across latent mixtures.

    Parameters
    ----------
    latent_spec : list[list[dict]]
        Per-latent lists of kernel specifications.

    Returns
    -------
    Array
        Block-diagonal mask mapping latents to state-space coordinates.
    """
    left = jsp.linalg.block_diag(
        *[mixture_mask(mixture_spec) for mixture_spec in latent_spec]
    )
    right = jnp.zeros_like(left)  # the rest SSM dimensions are for the imaginary part.
    M = jnp.column_stack([left, right])
    return M


@jax.jit
def norm_loading(w, axis=0):
    """Normalise loadings along the requested axis.

    Parameters
    ----------
    w : Array
        Loading matrix.
    axis : int, default=0
        Axis along which to compute norms.

    Returns
    -------
    Array
        Normalised loadings.
    """
    _norm = partial(
        _norm_except_axis,
        norm=partial(jnp.linalg.norm, keepdims=True),
        axis=axis,
    )
    return w / (_norm(w) + EPS)


def _norm_except_axis(v: Array, norm: Callable[[Array], float], axis: int | None):
    """Apply a norm across all axes except the specified one.

    Parameters
    ----------
    v : Array
        Input tensor.
    norm : Callable[[Array], float]
        Norm function applied to slices.
    axis : int | None
        Axis exempt from the norm operation.

    Returns
    -------
    Array
        Tensor of norms broadcast across the exempt axis.
    """
    if axis is None:
        return norm(v)
    else:
        return jax.vmap(norm, in_axes=axis, out_axes=axis)(v)


def training_progress():
    """Create a rich progress bar configured for CVI training.

    Returns
    -------
    Progress
        Rich progress instance with CVI-specific columns.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TextColumn("•"),
        "Elapsed",
        TimeElapsedColumn(),
        TextColumn("•"),
        "Remainning",
        TimeRemainingColumn(),
        TextColumn("•"),
        "Negative ELL",
        TextColumn("{task.fields[nell]:.3f}"),
    )


def ridge_estimate(y, m, V, lam=0.1):
    """Solve a ridge regression for the observation model.

    Parameters
    ----------
    y : Array
        Observations.
    m : Array
        Posterior means.
    V : Array
        Posterior covariances (unused but kept for API parity).
    lam : float, default=0.1
        Ridge penalty scaling.

    Returns
    -------
    tuple[Array, Array, Array]
        Loading matrix, bias vector, and observation covariance.
    """
    y = jnp.vstack(y)
    m = jnp.vstack(m)

    T, y_dim = y.shape
    _, z_dim = m.shape

    m1 = jnp.column_stack([jnp.ones((T, 1)), m])

    assert m1.shape == (T, z_dim + 1)

    zy = m1.T @ y  # (z + 1, t) (t, y) -> (z + 1, y)
    zz = m1.T @ m1  # (z + 1, t) (t, z + 1) -> (z + 1, z + 1)
    eye = jnp.eye(zz.shape[0])
    w = jnp.linalg.solve(zz + lam * eye, zy)  # (z + 1, z + 1) (z + 1, y) -> (z + 1, y)

    r = y - m1 @ w  # (t, y)
    R = r.T @ r / T  # (y, y)

    d, C = jnp.split(w, [1], axis=0)  # (1, y), (z, y)

    d = jnp.squeeze(d)
    C = C.T
    assert d.shape == (y_dim,)
    assert C.shape == (y_dim, z_dim)

    return C, d, R


def filter_array(arr: Array, mask: Array) -> Array:
    """Filter array leading axes by a less-or-equal rank mask array.

    Parameters
    ----------
    arr : Array
        Input array.
    mask : Array
        Boolean or integer mask with compatible leading shape.

    Returns
    -------
    Array
        Filtered array containing only entries where the mask is positive.
    """
    return arr[mask > 0]
