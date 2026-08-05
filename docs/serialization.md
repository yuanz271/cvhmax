# Model Serialization

This document specifies persistence for `CVHM` models.

## Scope

The persistence API saves and restores a reusable model, not a completed
inference cache. A model consists of:

- `CVHM` configuration;
- `HidaMatern` kernel configurations;
- the stateless readout algorithm name; and
- fitted readout parameters owned by `CVHM`, when present.

The posterior and latent inference caches are transient outputs. They are not
saved because they can be reconstructed relatively quickly from a restored
model and may be substantially larger than the model itself.

The public API is:

```python
model.save("model.cvhmax")
restored = CVHM.load("model.cvhmax")
posterior = restored.infer(y, valid_y)
```

`save()` returns `None`. `load()` is a classmethod returning a reconstructed
`CVHM` object. `infer()` recomputes posterior state without performing a
readout-parameter update.

## Configuration and ownership

`CVHM` is the composite model. It owns model-level configuration, kernels, the
stateless readout algorithm identifier, and fitted readout parameters:

```text
CVHM
├── kernels
├── observation: stateless CVI algorithm
└── params: fitted readout PyTree
```

`Gaussian` and `Poisson` are pure algorithmic `CVI` classes. Their methods
consume `Params` and, where appropriate, return an updated `Params`; the
classes do not own fitted state.

`CVHM` and `HidaMatern` use one common configuration protocol:

```python
class Serializable(Protocol):
    def get_config(self) -> dict: ...

    @classmethod
    def from_config(cls, config: dict): ...
```

The configuration is JSON-compatible and sufficient to reproduce the
component's semantics. `from_config()` owns translation from the stable
configuration to the current implementation. Normal constructors remain
ordinary Python APIs and do not accept serialized dictionaries as a second
calling convention.

For this working-version format, no cross-version compatibility is promised.
The archive is expected to be loaded by the same working version of cvhmax
that wrote it. The format version is retained only to reject incompatible or
malformed files clearly; no migration framework is required.

## File format

A save produces one self-contained ZIP archive at the requested path. It must
not require sidecar files, the original Python process, or the original model
object to load successfully.

The archive contains at least:

```text
model.cvhmax
├── manifest.json
└── params.eqx
```

`params.eqx` is omitted when the model has no fitted readout parameters. The
manifest contains model and kernel configuration; Equinox stores the numerical
leaves of the CVHM-owned fitted parameter PyTree.

The format must not use `pickle`, `cloudpickle`, or any mechanism that
reconstructs arbitrary executable Python object graphs. The archive is a
data format, not a Python object snapshot.

## Manifest

`manifest.json` is UTF-8 JSON. It contains the model configuration, HidaMatern
configuration, and the readout class name. It does not encode current
constructor signatures as a separate persistence mechanism.

A representative manifest is:

```json
{
  "format": "cvhmax.CVHM",
  "version": 1,
  "model": {
    "n_components": 2,
    "dt": 1.0,
    "lr": 0.1,
    "max_iter": 10,
    "cvi_iter": 5,
    "observation": "Gaussian",
    "kernels": [
      {
        "sigma": 1.0,
        "rho": 50.0,
        "omega": 0.0,
        "order": 0,
        "s": 1e-5
      }
    ]
  },
  "params": {
    "present": true,
    "type": "cvhmax.cvi.Params",
    "R_is_none": false,
    "arrays": {
      "C": {"shape": [2, 1], "dtype": "float64"},
      "d": {"shape": [2], "dtype": "float64"},
      "R": {"shape": [2, 2], "dtype": "float64"}
    }
  }
}
```

The manifest must record:

- archive format and integer version;
- all `CVHM`-owned configuration: `n_components`, `dt`, `lr`, `max_iter`,
  `cvi_iter`, and `observation`;
- each HidaMatern configuration: `sigma`, `rho`, `omega`, `order`, and `s`;
- whether fitted parameters are present;
- parameter-tree metadata needed to create the runtime Equinox `like` PyTree,
  including whether `Params.R` is `None`; and
- array shape and dtype metadata required by the Equinox restore path.

JSON values must be finite and representable without loss of model semantics.
Malformed structural metadata and unknown observation names must cause a clear
load error rather than silently falling back to `Gaussian`.

The `observation` value is the `CVI` subclass name, for example `"Gaussian"`
or `"Poisson"`. The initial implementation supports the built-in names only.
The kernel entries are assumed to be `HidaMatern`; no kernel type field is
needed.

## Kernel subclasses

A kernel supplied to `CVHM` may be a subclass of `HidaMatern`, but the format
preserves only the base HidaMatern configuration. Loading always reconstructs
a base `HidaMatern` instance. Subclass-specific state and behavior are not
preserved; callers must use subclasses only when they are semantically
equivalent to the base class. The format does not attempt to support new
kernel families.

## Save procedure

`CVHM.save(path)` must:

1. Validate all `CVHM` configuration, HidaMatern kernels, the built-in
   observation name, and fitted parameter structure before writing anything.
2. Obtain kernel configuration through the HidaMatern serialization protocol.
3. Serialize fitted `CVHM.params` with Equinox when present.
4. Write one ZIP archive containing `manifest.json` and, when applicable,
   `params.eqx`.
5. Exclude `posterior`, `latent`, device buffers, compilation caches, and
   other transient runtime state.
6. Write atomically through a temporary file followed by `os.replace`.
7. Avoid leaving a partial archive at the destination after failure.

The save operation must support both unfitted models (`params is None`) and
fitted models. Unsupported kernel subclasses, observation names, or parameter
structures must fail with an actionable error unless their semantics are
explicitly covered by the working implementation.

## Load procedure

`CVHM.load(path)` must:

1. Open and validate that the archive is a ZIP file.
2. Validate member names, duplicate members, member sizes, and required
   members before allocating restore arrays.
3. Parse and validate the archive format/version; reject files from an
   incompatible working version.
4. Validate the observation class name against the built-in `CVI` mapping and
   construct the corresponding stateless algorithm. Never silently substitute
   `Gaussian`.
5. Reconstruct each `HidaMatern` kernel through its configuration protocol.
6. Construct `CVHM` with its own saved configuration, reconstructed kernels,
   and readout algorithm.
7. Reconstruct a runtime Equinox `like` PyTree for CVHM-owned `params` with the
   same structure and leaf types as the saved fitted state. The `like` tree is
   not saved as a separate artifact.
8. Restore parameter leaves from `params.eqx` with Equinox.
9. Return the reconstructed model without `posterior` or `latent` attributes.

Restoration must not depend on the save-time device. Arrays may be restored to
the current default JAX device and copied to another device by normal JAX
operations afterward.

A loaded model must not have a posterior or latent cache merely because the
saved model was previously fitted. `model.infer(y, valid_y=None)` recomputes a
posterior without updating fitted readout parameters.

## Supported types

The initial implementation supports:

- `CVHM`;
- `HidaMatern` kernels of orders 0, 1, and 2; subclasses are accepted only
  when they are semantically equivalent to the base class and are restored as
  base `HidaMatern` instances;
- the stateless `Gaussian` and `Poisson` `CVI` algorithms; and
- the Equinox `cvhmax.cvi.Params` parameter container.

User-defined readouts are not part of the initial format. User-defined kernel
subclasses are supported only under the HidaMatern-subclass restriction above.
Arbitrary Python classes remain rejected.

## Safety

Although the format contains no pickle payload, loading still consumes data
from disk and must validate archive paths, member names, sizes, JSON structure,
configuration values, and parameter metadata before allocation. Archives from
untrusted sources should not be accepted without the normal filesystem and
resource controls of the calling application.
