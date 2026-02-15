# Contributing

The full contributor guide lives at `docs/contributing.md`.

Start here:

- `docs/contributing.md`
- `AGENTS.md`

Quick commands:

```bash
uv sync --group dev
ruff check src tests
JAX_ENABLE_X64=1 pytest tests/test_hp.py::test_whittle
```
