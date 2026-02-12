# Installation

This project targets Python 3.12+ and JAX. Use `uv` for a clean dev environment when possible.

## Quick Setup

```bash
uv sync --group dev
```

If `uv` is unavailable:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.lock  # if present
```

## JAX Notes

- JAX wheels are platform-specific. For GPU/TPU installs, follow the JAX install guide.
- Tests expect 64-bit precision.

```bash
export JAX_ENABLE_X64=1
```

Optional CPU-only debugging:

```bash
export XLA_FLAGS=--xla_force_host_platform_device_count=1
```
