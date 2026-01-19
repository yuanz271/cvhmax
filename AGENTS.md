# cvhmax Knowledge Base

**Generated:** 2026-01-19  
**Commit:** 9a6489d  
**Branch:** main

## Overview

Variational latent-state inference with Hida-Matern kernels in JAX. Couples information-form Kalman filtering with CVI-EM for smooth latent trajectory recovery from high-dimensional Gaussian/Poisson observations.

## Structure

```
cvhmax/
├── src/cvhmax/       # Core library
│   ├── cvhm.py       # CVHM wrapper orchestrating CVI-EM loop
│   ├── cvi.py        # CVI base + Gaussian/Poisson readouts (635 lines, largest)
│   ├── filtering.py  # Forward/backward information filters (bifilter)
│   ├── hm.py         # HidaMatern kernel class, state-space covariance blocks
│   ├── hp.py         # Whittle spectral hyperparameter fitting
│   └── utils.py      # LBFGS solver, ridge regression, progress bars
├── tests/            # Mirrors src/ layout (test_<module>.py)
└── pyproject.toml    # hatchling build, ruff config, dev deps
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| Add new likelihood | `cvi.py` | Subclass `CVI`, register via `__init_subclass__` |
| Modify SSM dynamics | `hm.py` | `Af/Qf/Ab/Qb` methods on `HidaMatern` |
| Change filtering algo | `filtering.py` | `bifilter`, `information_filter_step` |
| Tune hyperparameters | `hp.py` | `whittle()` for spectral fitting |
| Adjust training loop | `cvhm.py` | `CVHM.fit()`, `em_step` inner function |
| Add utility functions | `utils.py` | Keep JAX-jittable, pure |

## Code Map

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `CVHM` | class | cvhm.py:17 | Main API entry point |
| `CVI` | class | cvi.py:76 | Base class with registry pattern |
| `Gaussian` | class | cvi.py:179 | Linear-Gaussian readout |
| `Poisson` | class | cvi.py:433 | Exponential-link Poisson readout |
| `Params` | class | cvi.py:47 | Equinox Module for readout params |
| `HidaMatern` | class | hm.py:148 | Kernel as linear Gaussian SSM |
| `bifilter` | func | filtering.py:140 | Forward+backward info filter merge |
| `whittle` | func | hp.py:159 | Spectral hyperparameter optimizer |
| `lbfgs_solve` | func | utils.py:20 | Optax LBFGS wrapper |

## Conventions

- **Type hints**: Use `jaxtyping` for array shapes
- **Docstrings**: NumPy-style (`Parameters`, `Returns`, `Raises`)
- **Naming**: Classes PascalCase, functions lower_snake_case
- **Constants**: Cluster near consumers (e.g., `EPS = 1e-6`, `TAU = 1e-6`, `MAX_LOGRATE = 7.0`)
- **Functions**: Keep pure and JAX-jittable
- **Exports**: Add public APIs to `src/cvhmax/__init__.py`
- **ruff**: Ignores `E501` (line length), `F722` (forward refs)

## Anti-Patterns

- **No `as any` / `# type: ignore`** except where unavoidable (2 instances in cvi.py for LBFGS return types)
- **No empty `__init__.py`**: Current state is empty - add exports when ready
- **Avoid breaking jit**: No Python side-effects in jittable functions
- **No `fori_loop` with callbacks**: Training loop uses Python for-loop to support `jax.debug.callback`

## Technical Debt (TODOs)

| Location | Issue |
|----------|-------|
| hp.py:56 | Convert trainable scalar to 0-dim JAX array |
| hp.py:151 | Add L2 regularization to Whittle loss |
| hm.py:23 | Evaluate sympy2jax/equinox for symbolic kernels |
| hm.py:280-282 | Composite kernel support (linear combinations, pytree params) |

## Commands

```bash
# Install (dev)
uv sync --group dev
# or: python -m pip install -e . -r requirements-dev.lock

# Test
pytest                          # full suite
pytest tests/test_hp.py         # single module

# Lint
ruff check src tests            # check
ruff check src tests --fix      # auto-fix

# JAX config
export JAX_ENABLE_X64=1         # 64-bit precision (required for tests)
export XLA_FLAGS=--xla_force_host_platform_device_count=1  # CPU debugging
```

## Notes

- Tests generate artifacts (e.g., `cvhm.pdf`) - clean up manually or use tmpdir
- Some tests commented out (`test_whittle`) - work in progress
- Seed stochastic tests with `np.random.seed` / `jax.random.PRNGKey` for reproducibility
- `chex` used for shape assertions in tests
- No CI/CD configured - run `pytest` and `ruff check` before PRs
