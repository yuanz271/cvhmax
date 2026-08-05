# Model Serialization

This document specifies persistence for fitted `CVHM` models.

## Scope

The persistence API saves and restores a `CVHM` object as a reusable model. A
model consists of:

- `CVHM` constructor configuration;
- GP kernel objects and their parameters; and
- fitted observation-model parameters (`CVHM.params`), when present.

The posterior and latent inference caches are transient outputs. They are not
saved because they can be reconstructed by running inference again and may be
substantially larger than the model itself.

The public API is:

```python
model.save("model.cvhmax")
restored = CVHM.load("model.cvhmax")
posterior = restored.infer(y, valid_y)
```

`save()` returns `None`. `load()` is a classmethod returning a reconstructed
`CVHM` object. `infer()` recomputes posterior state without performing a
readout-parameter update.

`CVHM.load()` must return an in-memory `CVHM` object that is semantically
equivalent to the saved object. Equivalence means that it has the same model
configuration, kernel behavior, and observation-model parameters, subject to
normal device placement and numerical representation differences.

## File format

A save produces one self-contained archive at the requested path. The archive
must not require sidecar files, the original Python process, or the original
model object to load successfully.

The archive is ZIP-based and contains at least:

```text
model.cvhmax
├── manifest.json
└── params.eqx
```

`params.eqx` is omitted when the model has no fitted readout parameters.
For schema version 1, `manifest.json` is required exactly once and
`params.eqx` is required exactly when `params.present` is true. Unknown regular
members are ignored; duplicate member names, unsafe member names, malformed
required members, and unsupported schema versions are rejected.

The format must not use `pickle`, `cloudpickle`, or any serialization mechanism
that reconstructs arbitrary executable Python object graphs. The archive is a
data format, not a Python object snapshot.

## Manifest

`manifest.json` is UTF-8 JSON and contains all non-array information required
to reconstruct the model. At minimum it includes:

```json
{
  "format": "cvhmax.CVHM",
  "version": 1,
  "model": {
    "n_components": 2,
    "dt": 1.0,
    "observation": "Gaussian",
    "lr": 0.1,
    "max_iter": 10,
    "cvi_iter": 5
  },
  "kernels": [
    {
      "class": "cvhmax.hm.HidaMatern",
      "sigma": 1.0,
      "rho": 50.0,
      "omega": 0.0,
      "order": 0,
      "s": 1e-5
    }
  ],
  "params": {
    "present": true,
    "type": "cvhmax.cvi.Params",
    "R_is_none": false
  }
}
```

The initial format supports only the built-in `Gaussian` and `Poisson`
observation identifiers. Unknown or custom observation models must be rejected
rather than silently falling back to `Gaussian`.

The manifest schema must record:

- a format identifier and integer schema version;
- every constructor argument that affects `CVHM` behavior;
- the fully qualified supported kernel type and all kernel parameters;
- whether fitted readout parameters are present;
- the parameter-tree type and structural metadata needed to build a
  deserialization skeleton, including whether `Params.R` is `None`;
- array shape and dtype metadata when required by the Equinox restore path;
- optional package and dependency version metadata for reproducibility.

JSON values must be finite and representable without loss of model semantics.
Non-finite values, unknown model types, and malformed structural metadata must
cause a clear load error.

## Save procedure

`CVHM.save(path)` must:

1. Validate that the destination is a single archive path.
2. Serialize model configuration and kernel descriptions into the manifest.
3. Serialize `CVHM.params` with Equinox leaf serialization when parameters are
   present.
4. Write the manifest and parameter payload into one archive atomically, or
   remove an incomplete archive before reporting failure.
5. Exclude `posterior`, `latent`, device buffers, compilation caches, and other
   transient runtime state.

The save operation must support both an unfitted model (`params is None`) and a
fitted model. Saving unsupported kernel or parameter types must fail before
writing a partial archive, with an actionable error message.

## Load procedure

`CVHM.load(path)` must:

1. Open and validate the archive and manifest format identifier.
2. Check the schema version and reject unsupported versions clearly.
3. Validate all required members before constructing the result.
4. Reconstruct each supported kernel from its manifest description.
5. Construct a `CVHM` skeleton from the saved configuration.
6. Construct a matching `Params` skeleton when fitted parameters are present.
7. Restore parameter leaves from `params.eqx` using Equinox.
8. Return the reconstructed `CVHM` with the restored `params`.

The loader must not depend on the device type used during saving. Arrays may be
restored to the current default JAX device and may be copied to another device
by normal JAX operations after loading.

A loaded model must not have a posterior or latent cache merely because the
saved model was previously fitted. Posterior reconstruction is a separate
inference operation and must not silently perform an observation-model update.
The inference-only API `model.infer(y, valid_y=None)` recomputes a posterior
from a loaded model without refitting its readout parameters.

## Supported types

The initial implementation supports:

- `CVHM`;
- built-in `HidaMatern` kernels with orders 0, 1, and 2; and
- the built-in `cvhmax.cvi.Params` parameter container.

Custom kernels and custom CVI parameter PyTrees are not implicitly serialized.
They must either be rejected with a clear error or gain an explicit,
versioned serialization protocol in a future extension. Serializing arbitrary
Python classes would violate the data-only format requirement.

## Compatibility and safety

The format is versioned independently of Python pickle. Loaders must validate
schema and structural metadata before restoring arrays. A future incompatible
format requires a new schema version or an explicit migration path.

The archive is intended to be portable across CPU and GPU environments, but
not necessarily across incompatible numerical or model-code changes. The
manifest should record package and dependency versions when available, and
load errors should identify incompatibilities rather than silently producing a
different model.

Although the format contains no pickle payload, loading still consumes data
from disk and must validate archive paths, member names, sizes, and JSON
structures before allocation. Archives from untrusted sources should not be
accepted without the normal filesystem and resource controls of the calling
application.
