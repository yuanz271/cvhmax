# CVHM Serialization Implementation Plan

## Objective

Implement:

```python
model.save("model.cvhmax")
restored = CVHM.load("model.cvhmax")
posterior = restored.infer(y, valid_y)
```

The saved artifact is one self-contained ZIP archive containing model
configuration and fitted readout state, but not posterior or latent caches.
`save()` returns `None`; `load()` is a classmethod returning `CVHM`; `infer()`
returns posterior state without refitting readout parameters.

## Ownership and scope

`CVHM` is the composite GP-plus-GLM model and owns:

- model-level configuration: `n_components`, `dt`, `lr`, `max_iter`, and
  `cvi_iter`;
- the HidaMatern kernels;
- the stateless readout algorithm name (`observation`); and
- fitted readout parameters (`params`).

`Gaussian` and `Poisson` are pure `CVI` algorithms. Their methods consume
`Params` and return pseudo-observation information or updated `Params`; the
classes do not own fitted state.

The initial implementation assumes built-in `Gaussian` and `Poisson`
readouts. A CVI subclass name is stored as a string and resolved through the
built-in mapping; there is no readout configuration protocol or resolver in
this scope.

Kernels are assumed to be `HidaMatern`. A user may pass a subclass only when it
is semantically equivalent to the base class. The loader always reconstructs a
base `HidaMatern`; subclass identity, state, and behavior are not preserved.
The format does not attempt to support new kernel families.

## Configuration protocol

`CVHM` and `HidaMatern` use one common protocol:

```python
class Serializable(Protocol):
    def get_config(self) -> dict: ...

    @classmethod
    def from_config(cls, config: dict): ...
```

`get_config()` returns only JSON-compatible component configuration. The
archive owns the outer format/version metadata. `from_config()` translates the
stable configuration to the current implementation. Normal constructors
remain ordinary Python APIs and do not accept serialized dictionaries as a
second calling convention.

`CVHM.get_config()` stores its own fields, the observation class name, and the
list of kernel configurations. `CVHM.from_config()` delegates kernel
reconstruction and resolves the built-in observation name. Fitted `params` is
separate: Equinox restores its leaves into a runtime-created `like` PyTree
with the same structure and leaf types. The `like` tree values are not saved.

No cross-version compatibility is promised. The archive version exists only to
reject incompatible or malformed files clearly. Do not implement migration
machinery.

## Implementation steps

### 1. Add serialization helpers

Create or refactor `src/cvhmax/serialization.py` with:

- archive format/version validation;
- `CVHM` and HidaMatern configuration encoding/decoding;
- built-in observation-name resolution for `Gaussian` and `Poisson`;
- runtime reconstruction of the Equinox `like` PyTree for `Params`;
- ZIP archive read/write helpers;
- Equinox serialization to/from in-memory byte buffers; and
- archive-member safety validation.

Use the standard library plus existing Equinox. Do not add a serialization
dependency, pickle, cloudpickle, or archive-driven imports.

### 2. Define the manifest

Use a compact manifest with:

- `format` and integer `version`;
- `model.n_components`, `dt`, `lr`, `max_iter`, `cvi_iter`;
- `model.observation` as the `CVI` class name string;
- `model.kernels`, each containing HidaMatern fields `sigma`, `rho`, `omega`,
  `order`, and `s`;
- `params.present`;
- `params` type/structure metadata, including `R_is_none`; and
- parameter array shapes and dtypes needed to recreate the `like` PyTree.

Do not store a kernel type field: kernel entries are HidaMatern by contract.
Do not store readout type/config envelopes: the observation class name string
is sufficient for the built-in stateless algorithms.

Reject non-finite values, unknown observation names, unsupported HidaMatern
orders, malformed shapes/dtypes, and parameter metadata inconsistencies.

### 3. Implement `CVHM` configuration delegation

Add `CVHM.get_config()` and `CVHM.from_config(config)`.

`get_config()` must serialize only CVHM-owned fields and delegate each kernel's
configuration to its `get_config()` method. It must record `observation` as the
CVI class name, not serialize the CVI class or fitted parameters.

`from_config()` must:

1. validate model-level fields;
2. reconstruct each HidaMatern kernel via `from_config()`;
3. resolve only the built-in `Gaussian` and `Poisson` names;
4. construct the CVHM object; and
5. leave `posterior` and `latent` unset.

Unknown observations must raise an error; they must never silently fall back to
`Gaussian`.

### 4. Implement model save/load

`CVHM.save(path) -> None` must:

1. validate model configuration, kernels, built-in observation, and Params
   structure before writing;
2. create the manifest through `CVHM.get_config()`;
3. serialize CVHM-owned `params` with Equinox when present;
4. write `manifest.json` and optional `params.eqx` into one ZIP archive;
5. exclude posterior, latent, devices, and compilation caches; and
6. write atomically through a temporary file and `os.replace`.

`CVHM.load(path) -> CVHM` must:

1. validate ZIP structure, member paths, duplicates, sizes, and required
   members;
2. validate format/version and manifest fields;
3. reconstruct the CVHM/kernels/readout through `from_config()`;
4. construct the runtime Equinox `like` PyTree for CVHM-owned Params from the
   manifest structure metadata; and
5. restore `params.eqx` leaves with Equinox.

No sidecar files, pickle, dynamic imports, or save-time device assumptions are
allowed.

### 5. Implement inference-only reconstruction

Add:

```python
restored.infer(y, valid_y=None)
```

It must return `(m, V)`, use restored kernels and Params, not call
`update_readout`, not mutate `self.params`, and not assign `self.posterior` or
`self.latent`. Refactor shared filtering/smoothing logic if needed rather than
duplicating numerically sensitive code.

### 6. Add tests

Create `tests/test_serialization.py` covering:

- unfitted model round trip;
- fitted Gaussian round trip;
- fitted Poisson round trip with `Params.R is None`;
- exact CVHM-owned configuration and kernel configuration reconstruction;
- supported HidaMatern orders 0, 1, and 2;
- multiple components;
- parameter-array equality after Equinox restore;
- no posterior/latent after load;
- one self-contained ZIP with no pickle payload;
- malformed manifest, unsupported version, unknown observation, unsupported
  order, missing/inconsistent members, malformed metadata, unsafe names,
  duplicates, and size-limit errors;
- overwrite and failed-save cleanup behavior;
- inference after load without Params mutation;
- HidaMatern subclass inputs are restored as base HidaMatern instances; and
- fresh-process load plus inference.

Tests should use temporary paths and deterministic data and should validate
behavior and format invariants rather than private implementation details.

### 7. Update documentation

Update `docs/api.md` with `CVHM.get_config()`, `CVHM.from_config()`,
`CVHM.save()`, `CVHM.load()`, and `CVHM.infer()`.

Keep `docs/serialization.md` synchronized with the final manifest, ownership,
HidaMatern-subclass, built-in-readout, and no-version-compatibility policies.

### 8. Validate

```bash
uv run ruff check src tests
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest tests/test_serialization.py -q
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest -q
JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu \
  uv run pytest tests/test_gpu.py --run-gpu -q
git diff --check
```

## Definition of done

A fitted CVHM can be saved to one archive and loaded in a fresh process using
only that archive, with equivalent CVHM-owned configuration, HidaMatern kernel
semantics, built-in stateless readout, and fitted Params. No posterior or
latent cache is restored; `infer()` recomputes posterior without refitting;
no pickle or archive-driven import exists; malformed archives fail clearly;
and all relevant tests pass.
