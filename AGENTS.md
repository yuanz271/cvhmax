# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/cvhmax`, with inference routines in `cvhm.py`, `cvi.py`, filtering helpers in `filtering.py`, kernel definitions in `hm.py` and `hp.py`, and shared utilities in `utils.py`. Add new public APIs under `src/cvhmax` so they can be exported from `__init__.py`. Tests mirror the module layout inside `tests/` (`test_cvhm.py`, `test_cvi.py`, etc.); place new suites beside the logic they cover. Dependency pins and tooling live in `pyproject.toml`, with `requirements*.lock` and `uv.lock` supporting reproducible environments.

## Build, Test, and Development Commands
- `uv sync --group dev` installs runtime and development dependencies; if `uv` is unavailable, run `python -m pip install -e . -r requirements-dev.lock`.
- `pytest` (or `pytest tests/test_hp.py` for a focused run) executes the regression tests; expect artifacts like `cvhm.pdf` in the working directory.
- `ruff check src tests` lints the project; add `--fix` to apply safe patches.

## Coding Style & Naming Conventions
Follow Python 3.12 style with four-space indentation, type hints, and `jaxtyping` annotations for array shapes. Export classes in PascalCase (`CVHM`, `HidaMatern`) and helper functions in lower_snake_case. Keep functions pure and JAX-jittable, and cluster shared constants (for example, `EPS`) near their consumers. Document public APIs with NumPy-style docstrings (`Parameters`, `Returns`, `Raises`) to align examples and autodoc output. `ruff` is configured to ignore `E501` and `F722`; lean on expressive naming and add docstrings where they clarify math.

## Testing Guidelines
Seed stochastic tests (`np.random.seed`, `jax.random.PRNGKey`) to keep runs reproducible. Name files `test_<module>.py`, assert on shapes or summary metrics, and document any long-running paths. Enable 64-bit precision when needed via `export JAX_ENABLE_X64=1`, matching the existing tests’ `config.update("jax_enable_x64", True)`. If a test writes artifacts, clean them up in a fixture or direct them to a temporary path.

## Commit & Pull Request Guidelines
Write short, imperative commit messages (for example, `replace E-step loop with fori_loop`) and group related work together. Pull requests should include a brief summary, linked issues, notes on numerical impacts, and confirmation that `pytest` and `ruff check` succeeded. Attach diagnostic plots when they help reviewers reason about latent factor behaviour.

## JAX Configuration Tips
Set `XLA_FLAGS=--xla_force_host_platform_device_count=1` when debugging on CPU-only machines to mirror CI. Enable 64-bit precision early via `jax.config.update("jax_enable_x64", True)` or the `JAX_ENABLE_X64` environment variable. Profile long optimisations with `jax.debug.print` or the shared `rich` progress utilities in `utils.py`.
