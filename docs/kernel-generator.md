# Kernel Generator

The `kernel_generator` subpackage constructs Hida-Matern state-space
kernel matrices for **arbitrary smoothness orders** at runtime. It uses
SymPy for symbolic differentiation and `sympy2jax` to convert the
resulting expressions into JAX functions that are compatible with
`jax.jit`, `jax.vmap`, and `jax.grad`.

Before this subpackage existed, only Matern-1/2 (`order=0`) and
Matern-3/2 (`order=1`) were available via hand-coded expressions.
The kernel generator removes that limitation.

## Background

The Hida-Matern kernel is a complex-valued covariance function

```
k(tau) = sigma^2 * matern(nu, tau, rho) * exp(i * omega * tau)
```

where `sigma` is the amplitude, `rho` is the length scale, `omega` is
an oscillation frequency, and `nu = p + 1/2` controls the smoothness.

By expressing this kernel in state-space form, Kalman filtering reduces
GP inference from O(T^3) to O(T).  Each kernel's complex state dimension
equals the generator order `nple = M = p + 1`.  The total state dimension
across all `K` kernels is `L = 2 * sum(nple)` (factor of 2 from real-valued
representation).

### Order conventions

The generator order `M` is the SSM state dimension. The `HidaMatern`
dataclass in `hm.py` uses a smoothness index where `order = M - 1`.

| Generator order (M) | Matern smoothness | `HidaMatern.order` | Per-kernel dim (`nple`) |
|----------------------|-------------------|--------------------|------------------------|
| 1                    | 1/2               | 0                  | 1         |
| 2                    | 3/2               | 1                  | 2         |
| 3                    | 5/2               | 2                  | 3         |
| N                    | (2N-1)/2          | N-1                | N         |

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from cvhmax.kernel_generator import make_kernel

# Build a Matern-5/2 kernel (generator order M=3)
gen = make_kernel(3)

sigma = jnp.array(1.0)
rho = jnp.array(2.0)
omega = jnp.array(0.5)

# State-space covariance at lag tau
K = gen.create_K_hat(jnp.array(1.0), sigma, rho, omega)
print(K.shape)  # (3, 3)

# Stationary covariance K(0)
K0 = gen.create_K_hat(jnp.array(0.0), sigma, rho, omega)

# Spectral moments (length 2*M = 6)
moments = gen.get_moments(sigma, rho, omega)
print(moments.shape)  # (6,)

# Scalar base kernel
k_val = gen.get_base_kernel(jnp.array(1.0), sigma, rho, omega)
```

`make_kernel` returns a cached `HidaMaternKernelGenerator` instance.
Calling it again with the same order returns the same object with no
additional symbolic computation.

You can also construct a generator directly:

```python
from cvhmax.kernel_generator import HidaMaternKernelGenerator

gen = HidaMaternKernelGenerator(order=4)  # Matern-7/2
```

## Transparent integration with HidaMatern

The existing `HidaMatern` class and the free functions `Ks`, `Af`, `Qf`,
`Ab`, `Qb` in `hm.py` automatically dispatch to the kernel generator for
orders >= 2. No code changes are needed in downstream callers.

```python
from cvhmax.hm import HidaMatern

# Matern-5/2 kernel -- previously raised NotImplementedError
hm = HidaMatern(sigma=1.0, rho=2.0, omega=0.5, order=2, s=1e-6)

K0 = hm.K(0.0)      # (3, 3) complex
A  = hm.Af(1.0)     # (3, 3) complex
Q  = hm.Qf(1.0)     # (3, 3) complex
```

This means you can use higher-order kernels directly in the full CVHM
pipeline:

```python
from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern

kernels = [
    HidaMatern(sigma=1.0, rho=50.0, omega=0.0, order=2),  # Matern-5/2
    HidaMatern(sigma=1.0, rho=30.0, omega=2.0, order=3),  # Matern-7/2
]

model = CVHM(
    n_components=2,
    dt=1.0,
    kernels=kernels,
    observation="Gaussian",
    max_iter=5,
)
model.fit(y, valid_y=valid_y, random_state=0)
```

### Dispatch rules

| `HidaMatern.order` | Method         | Matrix size |
|---------------------|---------------|-------------|
| 0                   | Hand-coded `Ks0` | 1 x 1    |
| 1                   | Hand-coded `Ks1` | 2 x 2    |
| >= 2                | `kernel_generator.make_kernel(order + 1)` | (order+1) x (order+1) |

The dict-based `Ks(kernelparam, tau)` function follows the same rules.

## Using the generator directly

### `create_K_hat`

Returns the M x M complex state-space covariance matrix at a given lag.

```python
gen = make_kernel(2)  # Matern-3/2, M=2

tau   = jnp.array(0.5)
sigma = jnp.array(1.0)
rho   = jnp.array(1.0)
omega = jnp.array(0.0)

K = gen.create_K_hat(tau, sigma, rho, omega)
# shape: (2, 2), dtype: complex128
```

At `tau = 0` the matrix is the stationary covariance `K(0)`, which is
Hermitian positive definite:

```python
K0 = gen.create_K_hat(jnp.array(0.0), sigma, rho, omega)

# Verify Hermitian property
assert jnp.allclose(K0, jnp.conjugate(K0.T))

# Verify positive definiteness (via real representation)
from cvhmax.utils import real_repr
eigvals = jnp.linalg.eigvalsh(real_repr(K0))
assert jnp.all(eigvals > 0)
```

The SSM dynamics matrices are derived from `K_hat`:

```python
from cvhmax.utils import conjtrans

K0 = gen.create_K_hat(jnp.array(0.0), sigma, rho, omega)
Kt = gen.create_K_hat(jnp.array(1.0), sigma, rho, omega)

# Forward transition: A = K(t) @ K(0)^{-1}
A = conjtrans(jnp.linalg.solve(conjtrans(K0), conjtrans(Kt)))

# Process noise: Q = K(0) - K(t) @ K(0)^{-1} @ K(t)^H
Q = K0 - Kt @ jnp.linalg.solve(K0, conjtrans(Kt))

# Lyapunov equation holds: A @ K(0) @ A^H + Q == K(0)
assert jnp.allclose(A @ K0 @ conjtrans(A) + Q, K0, atol=1e-10)
```

### `get_moments`

Returns the 2M spectral moments of the kernel. Even-indexed entries are
the absolute values of the even-order derivatives of `k(tau)` evaluated
at `tau = 0`. Odd-indexed entries are always zero (by stationarity).

```python
gen = make_kernel(3)
moments = gen.get_moments(jnp.array(1.0), jnp.array(1.0), jnp.array(0.0))
# shape: (6,)
# moments[0] == sigma^2 == 1.0
# moments[1] == 0.0
# moments[2] > 0  (related to curvature)
# moments[3] == 0.0
# ...
```

These moments are useful for spectral density estimation and Whittle
hyperparameter fitting.

### `get_base_kernel`

Evaluates the scalar kernel `k(|tau|)` and returns its real part. Useful
for plotting the covariance profile or for diagnostics.

```python
gen = make_kernel(2)
sigma = jnp.array(1.0)
rho   = jnp.array(1.0)
omega = jnp.array(0.0)

# At tau=0, the base kernel equals sigma^2
k0 = gen.get_base_kernel(jnp.array(0.0), sigma, rho, omega)
assert jnp.isclose(k0, sigma**2)

# The kernel decays with increasing lag (when omega=0)
k1 = gen.get_base_kernel(jnp.array(1.0), sigma, rho, omega)
k5 = gen.get_base_kernel(jnp.array(5.0), sigma, rho, omega)
assert k0 >= k1 >= k5
```

## JIT, vmap, and grad compatibility

All three methods are fully compatible with JAX transformations.

### JIT

```python
@jax.jit
def compute_K(tau, sigma, rho, omega):
    gen = make_kernel(3)
    return gen.create_K_hat(tau, sigma, rho, omega)

K = compute_K(jnp.array(0.5), jnp.array(1.0), jnp.array(1.0), jnp.array(0.0))
```

Because `make_kernel` is cached, the `gen` object is the same across
calls and does not trigger re-tracing.

### grad

```python
def loss(sigma):
    gen = make_kernel(2)
    K0 = gen.create_K_hat(jnp.array(0.0), sigma, jnp.array(1.0), jnp.array(0.0))
    return jnp.sum(K0.real)

grad_fn = jax.grad(loss)
print(grad_fn(jnp.array(1.0)))
```

### vmap

```python
gen = make_kernel(2)

taus = jnp.linspace(0.0, 5.0, 100)
sigma = jnp.array(1.0)
rho   = jnp.array(1.0)
omega = jnp.array(0.0)

# Evaluate the base kernel over a batch of tau values
k_vals = jax.vmap(lambda t: gen.get_base_kernel(t, sigma, rho, omega))(taus)
```

## Performance notes

**First-call cost.** The first call to `make_kernel(M)` triggers symbolic
differentiation via SymPy, which involves computing `2M - 1` successive
derivatives and their limits. This is fast for low orders (< 1 second for
M <= 5) but grows for high orders (several seconds for M = 10+).

**Caching.** `make_kernel` uses `functools.lru_cache` with a capacity of
32 entries. After the first call for a given order, all subsequent calls
return the cached generator instantly. The cache persists for the
lifetime of the process.

**JIT compilation.** The JAX functions produced by `sympy2jax` are
compiled on their first evaluation with specific input shapes. Subsequent
evaluations with the same shapes hit the XLA cache.

**Numerical conditioning.** For very high orders (M > 8), the
Lyapunov equation `A @ K(0) @ A^H + Q = K(0)` may accumulate numerical
error (1e-9 at M=5, 1e-7 at M=8). This is inherent to the matrix
algebra, not the symbolic generation. For most practical applications,
orders up to 5 or 6 are sufficient.

## Module layout

```
src/cvhmax/kernel_generator/
    __init__.py       # exports HidaMaternKernelGenerator, make_kernel
    matern.py         # symbolic Matern polynomial + Hida-Matern kernel
    generator.py      # core class: symbolic diff -> sympy2jax -> JAX callables
```

## See also

- `api.md` -- API reference for all public symbols
- `algorithms.md` -- mathematical details of the K_hat construction
- `quickstart.md` -- end-to-end CVHM examples including higher-order kernels
