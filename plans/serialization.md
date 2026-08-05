# CVHM Serialization Implementation Plan

## Objective

Implement a paired model persistence API:

```python
model.save("model.cvhmax")
restored = CVHM.load("model.cvhmax")
```

The saved artifact must be one self-contained ZIP archive. It must restore a
semantically equivalent in-memory `CVHM` object from data alone, without
serializing the posterior, latent inference cache, or arbitrary executable
Python object graphs.

`save()` returns `None`. `load()` is a classmethod returning a reconstructed
`CVHM` object.

## Design decisions

- Do not serialize `CVHM` directly: it is a regular dataclass with mixed static
  configuration, kernel objects, and JAX arrays rather than a pure persistence
  PyTree.
- Use a versioned ZIP archive containing `manifest.json` and an optional
  Equinox parameter-leaf payload.
- Use Equinox for `Params` leaves, JSON for static configuration, and explicit
  reconstruction for `CVHM` and `HidaMatern`.
- Do not use `pickle` or `cloudpickle`.
- Save model state only. Do not save `posterior` or `latent`.
- Initially support `CVHM`, built-in `HidaMatern` kernels of orders 0, 1, and
  2, the built-in `cvhmax.cvi.Params` container, and observation models
  `Gaussian` and `Poisson`.
- Reject custom kernels, custom observation models, and custom CVI parameter
  PyTrees unless a future explicit, versioned serialization protocol is added.
- Do not silently fall back to `Gaussian` when a saved observation model is
  unknown.

## Implementation steps

### 1. Add serialization helpers

Create `src/cvhmax/serialization.py` with:

- archive format identifier and schema version;
- manifest construction and validation;
- `HidaMatern` encode/decode helpers;
- `Params` metadata and skeleton construction;
- ZIP archive read/write helpers;
- Equinox serialization to and from in-memory byte buffers;
- archive-member safety validation.

Use the standard library (`zipfile`, `json`, `io`, `os`, and related path/file
utilities) plus the existing Equinox dependency. Do not introduce a new
serialization dependency.

### 2. Define the manifest

The manifest must record:

- `format`: `cvhmax.CVHM`;
- integer schema `version`;
- all behavior-affecting `CVHM` configuration:
  `n_components`, `dt`, `observation`, `lr`, `max_iter`, and `cvi_iter`;
- kernel type and all `HidaMatern` parameters:
  `sigma`, `rho`, `omega`, `order`, and `s`;
- whether `params` is present;
- parameter type and structural metadata, including `R_is_none`;
- array shape and dtype metadata needed to build an Equinox restore
  skeleton;
- optional package and dependency version metadata for reproducibility.

Use the fully qualified kernel identifier `cvhmax.hm.HidaMatern` consistently.
The initial format supports only observation identifiers `Gaussian` and
`Poisson`; unknown or custom observation identifiers must be rejected during
save and load.

JSON values must be finite and representable without loss of model semantics.
Non-finite values, unknown model types, and malformed structural metadata must
cause a clear load error.

For schema version 1, the archive policy is:

- `manifest.json` is required exactly once;
- `params.eqx` is required exactly when `params.present` is true;
- unknown regular members are ignored;
- duplicate member names, unsafe member names, malformed required members, and
  unsupported schema versions are rejected.

### 3. Implement `CVHM.save`

Add:

```python
CVHM.save(path) -> None
```

The method must:

1. Validate the model and all supported types before writing anything.
2. Build the complete manifest.
3. Serialize `params` with Equinox when present.
4. Write one ZIP archive containing:
   - `manifest.json`;
   - `params.eqx` when fitted parameters exist.
5. Exclude `posterior`, `latent`, device buffers, and compilation caches.
6. Write atomically through a temporary file followed by `os.replace`.
7. Avoid leaving a partial archive at the destination after failure.

Support both unfitted models (`params is None`) and fitted models.

### 4. Implement `CVHM.load`

Add:

```python
@classmethod
CVHM.load(path) -> CVHM
```

The method must:

1. Open the archive and validate that it is a ZIP file.
2. Validate member names, duplicate members, member sizes, and required
   members before allocating restore arrays.
3. Parse and validate the manifest format identifier and schema version.
4. Reconstruct supported `HidaMatern` kernels from manifest data.
5. Validate the saved observation identifier against `CVI.registry` and the
   supported built-in set; never silently substitute the default observation.
6. Construct a `CVHM` skeleton from the saved configuration.
7. Construct a matching `Params` skeleton with recorded shapes/dtypes and
   `R is None` state.
8. Restore parameter leaves with Equinox.
9. Return the reconstructed model without `posterior` or `latent` attributes.

Restoration must not depend on the save-time CPU/GPU device. Arrays should be
normal JAX arrays on the current environment.

### 5. Add an inference-only reconstruction path

Add an inference-only API, such as:

```python
restored.infer(y, valid_y=None)
```

It must use restored kernels and `params` to compute the posterior without
performing the observation-model M-step and without changing fitted readout
parameters. It should return the posterior, or document precisely whether it
stores and returns it, but it must not silently refit the model.

This is needed because the serialization format intentionally excludes
`posterior` and `latent`. The fit loop should be refactored so that the common
filtering/smoothing path can be reused by `fit()` and `infer()` without
duplicating numerical logic.

### 6. Add validation errors

Cover at least:

- invalid archive or non-ZIP path;
- malformed JSON;
- wrong format identifier;
- unsupported schema version;
- missing `manifest.json`;
- missing `params.eqx` when parameters are declared present;
- unexpected parameter payload when parameters are absent;
- unknown kernel type;
- unsupported Hida–Matérn order;
- unsupported custom kernel;
- unsupported or unknown observation model;
- unsupported parameter container;
- malformed shape or dtype metadata;
- non-finite numeric metadata;
- inconsistent parameter structure;
- duplicate ZIP member names;
- absolute or traversal member paths;
- unexpected archive metadata or member types;
- archive/member size limits exceeded.

Errors should identify the invalid field or member and provide an actionable
message. Do not extract archive members to the filesystem; read required
members directly into bounded in-memory buffers.

### 7. Add tests

Create `tests/test_serialization.py` covering:

- unfitted model round trip;
- fitted Gaussian model round trip;
- fitted Poisson model round trip with `Params.R is None`;
- exact reconstruction of constructor configuration;
- exact reconstruction of kernel parameters and kernel dynamics;
- equality of restored parameter arrays;
- absence of `posterior` and `latent` after loading;
- single self-contained ZIP archive contents;
- no pickle/cloudpickle payload or object serialization;
- malformed manifest and unsupported-version errors;
- missing or inconsistent archive members;
- unsupported custom kernel failure;
- unsupported observation-model failure;
- unsafe archive member names and duplicate members;
- unknown regular-member handling according to the version-1 policy;
- overwrite behavior and no partial archive after a failed save;
- inference after loading without changing readout parameters.

Add a subprocess test for fresh-process portability:

1. Fit and save a model in one Python process.
2. Start a second Python process with only the archive and installed package
   available.
3. Load the archive, validate configuration and parameter values, and run the
   inference-only path.

Use temporary paths and deterministic arrays/seeds. Tests should verify
behavior and format invariants, not private implementation details.

### 8. Update public documentation

Update `docs/api.md` to document `CVHM.save`, `CVHM.load`, and the inference-only
API, and link to `docs/serialization.md` for the format contract. Clarify that
persistence saves reusable model state only and that posterior reconstruction
is a separate inference operation.

Update `docs/serialization.md` to match the final observation-model policy,
archive-member policy, `save`/`load` return types, and inference-only workflow.

### 9. Validate archive safety and compatibility

Before allocating restore arrays, validate:

- ZIP member names and paths;
- duplicate members;
- compressed and uncompressed sizes;
- allowed required members;
- JSON size and structure;
- required versus optional members;
- schema version and format identifier.

Use explicit bounded limits for manifest and parameter payload sizes where
practical. Do not extract archive members to the filesystem. A future
incompatible format requires a new schema version or an explicit migration
path.

## Validation commands

Run the smallest relevant checks during implementation:

```bash
uv run ruff check src tests
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest tests/test_serialization.py -q
```

Then run the regression suite:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest -q
```

If GPU access is available, also run:

```bash
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu \
  uv run pytest tests/test_gpu.py --run-gpu -q
```

Finally run `git diff --check` and inspect the complete diff.

## Definition of done

The task is complete when a fitted `CVHM` can be saved to one archive and
loaded in a fresh Python process using only that archive, with equivalent
configuration, kernels, and fitted readout parameters; no posterior or latent
cache is restored; the inference-only path can recompute posterior state
without refitting; no pickle mechanism is present; malformed or unsafe
archives fail clearly; and the relevant tests and checks pass.
