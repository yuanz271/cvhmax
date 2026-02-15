# Contributing

Thanks for helping improve cvhmax. See `AGENTS.md` for the full style guide.

## Setup

```bash
uv sync --group dev
```

Fallback:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.lock  # if present
```

Enable 64-bit precision for tests:

```bash
export JAX_ENABLE_X64=1
```

## Lint and Test

```bash
ruff check src tests
ruff check src tests --fix

JAX_ENABLE_X64=1 pytest
JAX_ENABLE_X64=1 pytest tests/test_hp.py
JAX_ENABLE_X64=1 pytest tests/test_hp.py::test_whittle
JAX_ENABLE_X64=1 pytest -k whittle
```

## Code Style

Follow `AGENTS.md` for JAX-specific conventions. Key points:

- Keep core logic pure and jittable.
- Validate shapes/dtypes at API boundaries (`chex.assert_*`).
- Prefer `jnp.linalg.solve` over explicit inverses.

## Adding a New Observation Model

1) Add a new `CVI` subclass in `src/cvhmax/cvi.py`.
2) Ensure it registers via `__init_subclass__`.
3) Add tests under `tests/` mirroring the module layout.
4) Update docstrings and `docs/api.md`.

## Updating Kernels or Filtering

- Kernel changes live in `src/cvhmax/hm.py`.
- Filtering logic lives in `src/cvhmax/filtering.py`.

Update docstrings and `docs/algorithms.md` if behavior changes.
