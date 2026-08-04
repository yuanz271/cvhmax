# Agent Guide (cvhmax)

This file is for coding agents working in this repo. Keep changes small, follow existing JAX patterns, and prefer pure/jittable code. For project context, design specifications, API details, and other agent-agnostic information, consult `README.md`, `docs/`, source docstrings, and repository metadata as needed; do not duplicate that information here.

## Hard Rules (Git + Safety)

- Commits are allowed autonomously at meaningful milestones, when a plan is
  complete, or when the task is being closed; do not commit every individual
  edit. Before each commit, inspect `git status` and the relevant diff, and
  commit only the scoped changes for the milestone.
- Never run `git push` without explicit user approval.
- Avoid destructive git commands (`reset --hard`, force-push, rewriting history)
  unless explicitly requested.
- Do not revert/overwrite unrelated local changes; work around them.

## Branching Model (GitHub Flow)

- `main` is the default branch and always deployable.
- Create feature branches from `main` (e.g., `feature/add-poisson-cvi`).
- Open a PR to merge feature branches back to `main`.
- No long-lived `develop` branch; delete feature branches after merge.



## Build / Lint / Test Commands

Assume Python >= 3.12 (`pyproject.toml`). JAX tests expect 64-bit.

```bash
# Install (dev)
uv sync --group dev
# or
python -m pip install -e .
python -m pip install -r requirements-dev.lock  # if present

# (Optional) build frontend tooling
python -m pip install build

# Lint
ruff check src tests
ruff check src tests --fix

# Format (optional; only run if the repo already uses it)
ruff format src tests

# Test (full)
JAX_ENABLE_X64=1 pytest

# Test (single file)
JAX_ENABLE_X64=1 pytest tests/test_hp.py

# Test (single test function)
JAX_ENABLE_X64=1 pytest tests/test_hp.py::test_whittle

# Test (by substring match)
JAX_ENABLE_X64=1 pytest -k whittle

# Fast iteration options
JAX_ENABLE_X64=1 pytest -q
JAX_ENABLE_X64=1 pytest -x --maxfail=1
JAX_ENABLE_X64=1 pytest --lf          # rerun last failures
JAX_ENABLE_X64=1 pytest --ff          # run last failures first

# Benchmark (slow, ~30 s; skipped by default)
JAX_ENABLE_X64=1 pytest tests/test_benchmark.py --run-slow

# Build (sdist/wheel)
python -m build
# (Alternative, if hatch is installed)
# hatch build
```

Useful debugging env vars:

```bash
export JAX_ENABLE_X64=1
export XLA_FLAGS=--xla_force_host_platform_device_count=1  # CPU-only / debugging
```

## Where To Change What

| Task | Location | Notes |
|------|----------|-------|
| Add new observation model | `src/cvhmax/cvi.py` | Subclass `CVI`; registry via `__init_subclass__` |
| Adjust EM loop / training | `src/cvhmax/cvhm.py` | `CVHM.fit()`, `em_step` |
| Filtering logic | `src/cvhmax/filtering.py` | `information_filter_step`, `bifilter` |
| Kernel / dynamics blocks | `src/cvhmax/hm.py` | `Af/Qf/Ab/Qb`, `K(tau)` |
| Hyperparameter fitting | `src/cvhmax/hp.py` | `whittle()` |
| Utilities | `src/cvhmax/utils.py` | Keep jittable + side-effect free |

## Common Workflows

### Adding a New Observation Model

1. Add a new `CVI` subclass in `src/cvhmax/cvi.py`.
2. Ensure it registers via `__init_subclass__`.
3. Add tests under `tests/` mirroring the module layout.
4. Update docstrings and `docs/api.md`.

## Benchmark Regression Guard

Any change that affects inference quality **must** pass the benchmark:

```bash
JAX_ENABLE_X64=1 pytest tests/test_benchmark.py --run-slow
```

The benchmark runs three Poisson-VdP scenarios (frozen readout, estimated
readout, variable-length trials) and asserts R² ≥ 0.96.  Baseline R² is
~0.972 on all three.  If a change drops below the floor, either fix the
regression or update the floor with justification.

## Docs Update Rule

**Every code change must include matching documentation updates.** This is a hard rule, not a suggestion. Check the touchpoints table below after every change and update all affected files before considering the task complete.

| Change Type | Update |
|-------------|--------|
| Public API surface | `docs/api.md` + docstrings + `__all__` in `__init__.py` |
| Shapes/masks/data model | `docs/data-model.md` + docstrings |
| Algorithms/logic changes | `docs/algorithms.md` + docstrings |
| Kernel generator | `docs/kernel-generator.md` + `docs/algorithms.md` |
| Examples / quickstart | `README.md` |
| Dev workflow/tooling | `AGENTS.md` + `README.md` (if user-facing) |

## Code Style Guidelines

The repo is linted with Ruff (ignores `E501` and `F722`). There is no required line length; readability wins.

### Imports

- Group imports: standard library, third-party, then local (`from .utils import ...`).
- Prefer `import jax.numpy as jnp` and `from jax import Array` for typing.
- Avoid `numpy` in core logic unless necessary; keep NumPy usage mainly in tests/IO.
- Keep `jax.numpy.linalg` ops (`solve`, `multi_dot`) explicitly imported when it improves clarity.

### Formatting

- Use f-strings; keep docstrings NumPy-style (`Parameters`, `Returns`, `Raises`).
- Use explicit names over single-letter variables unless it is standard math (`z`, `Z`, `J`, `Q`).
- Constants: `UPPER_SNAKE_CASE` near the code that uses them (e.g., `TAU`, `EPS`).
- Prefer small, composable helpers over long monolithic functions (especially around `lax.scan`).

### Types

- Arrays: annotate as `jax.Array` (or `Array` from `jax`).
- Scalars: use `float`, `int`, `bool` unless a 0-d JAX array is required.
- Public functions/methods should have type hints and stable return types.
- Pytree containers: use `equinox.Module` (e.g., `Params`) or `@dataclass` with JAX-compatible fields.

### Naming

- Classes: `PascalCase` (`CVHM`, `HidaMatern`, `Poisson`).
- Functions/vars: `lower_snake_case`.
- Private helpers: prefix `_`.

### Error Handling and Validation

- Validate shapes/dtypes at API boundaries (e.g., `CVHM.fit`) using `chex.assert_*` or explicit checks.
- Inside `jax.jit`/`lax.scan` code paths, avoid Python exceptions; do checks outside jit.
- Raise `ValueError` for invalid values, `TypeError` for wrong types; include shapes in messages.
- Prefer failing early with a helpful message over silently broadcasting/reshaping inputs.

### JAX + Numerical Conventions

- Prefer `jnp.linalg.solve(A, b)` over explicit inverses; keep `inv(...)` only when clearly justified.
- Keep core computations pure (no mutation except via `.at[...]`), and avoid Python side effects in jittable code.
- Randomness: use explicit seeds/keys; tests should be reproducible.
- Masking: use `jnp.where` with broadcastable masks; keep masks as integer/bool arrays.
- Be careful with dtype: tests assume 64-bit; avoid accidental `float32` constants (`1.0` is ok with x64 enabled).
- Prefer `vmap` over Python loops for per-trial/per-time computations; use `lax.scan` for recurrences.
