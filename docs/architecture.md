# Architecture

This page describes the high-level design of cvhmax and the separation of
concerns between its core components.

## Three-Way Separation

The codebase enforces a strict boundary between three components.  Each
component works in its own space and communicates through CVHM, which acts
as a bridge.

```
CVI (observation model)     CVHM (bridge)     Filtering (dynamics)
    latent space (K)     ←——— M ———→          state space (L)
```

| Component | Space | Knows about | Responsibility |
|-----------|-------|-------------|----------------|
| CVI | latent `(K)` | observations only | information updates, readout params |
| Filtering | state `(L)` | dynamics only | information-form smoothing |
| CVHM | both via `M` | bridge | `lift`, `project`, loop orchestration |

### Data Flow

```
CVI                          CVHM                         Filtering
(latent space)               (bridge)                     (state space)

observations ──→ initialize_info ──→ j_latent, J_latent
                                           │
                                     lift (j @ M, M.T @ J @ M)
                                           │
                                           ▼                      ← warm-up
                              j, J ──→ forward_filter ──→ zp, Zp  (predicted)
                                           │
                                     project (sde2gp)
                                           │
                                           ▼
                              m_w, V_w ─→ update_pseudo ──→ j_latent, J_latent
                                           │
                                     lift (j @ M, M.T @ J @ M)
                                           │
                                           ▼                      ← CVI loop
                              j, J ──→ bifilter ──→ z, Z
                                           │
                                     project (sde2gp)
                                           │
                                           ▼
                              m, V ──→ update_pseudo ──→ j_latent, J_latent
                                           │
                                         (repeat)
                                           │
                                           ▼
                              m, V ──→ update_readout ──→ params
```

## Spaces and Dimensions

| Space | Dimension | Owner | Purpose |
|-------|-----------|-------|---------|
| state | `L = 2 * sum(nple)` | Filtering | SDE dynamics |
| latent | `K = n_components` | CVI | observation model |

The selection mask `M` (shape `(K, L)`) is a structural constant
determined by the kernel configuration.  It is owned by CVHM and never
passed to CVI.

## Conversion Formulas (owned by CVHM)

**lift** (latent → state information):
```
j_state = j_latent @ M              # (..., K) → (..., L)
J_state = M.T @ J_latent @ M        # (..., K, K) → (..., L, L)
```

**project** (state information → latent moments):
```
m, V = sde2gp(z, Z, M)
# m = solve(Z, z) @ M.T             # (..., L) → (..., K)
# V = M @ inv(Z) @ M.T              # (..., L, L) → (..., K, K)
```

## Forward-Filter Warm-Up

After `initialize_info` returns per-bin pseudo-observations (computed
independently at each time bin), CVHM runs a single forward information
filter pass before the CVI loop.  This propagates causal information
across bins so that each bin's pseudo-observations are seeded from a
sequentially coherent prediction rather than a zero-mean prior.

The warm-up is a state-space operation (it uses `Af`, `Pf`) and therefore
belongs in CVHM, not CVI.  It uses the **predicted** (one-step-ahead)
states — not the posteriors — to avoid double-counting the current bin's
observation.  The resulting latent moments are passed to `update_pseudo`
with `lr=1.0` to fully replace the initial pseudo-observations.

For conjugate readouts (Gaussian) the warm-up is idempotent because
`update_pseudo` returns the same pseudo-observations regardless of the
posterior.

## Observation Model Interface

CVI subclasses implement four stateless class methods, all operating in
latent space:

| Method | Input | Output |
|--------|-------|--------|
| `initialize_params` | observations | opaque params |
| `initialize_info` | params, observations | `(j, J)` in latent space |
| `update_pseudo` | params, observations, posterior `(m, V)`, current `(j, J)` | updated `(j, J)` in latent space |
| `update_readout` | params, observations, posterior `(m, V)` | updated params |

### Duck-typed params

The params structure is opaque to CVHM.  Each CVI subclass may use any
pytree-compatible container — a tuple, a dataclass, an equinox `Module`,
or anything else.  The built-in `Gaussian` and `Poisson` readouts use the
`Params` dataclass (with fields `C`, `d`, `R`), but this is not required.

## Benefits

1. **Clean separation**: CVI knows nothing about state space; filtering
   knows nothing about observations; CVHM bridges the two.
2. **Easy to extend**: new observation models only need to implement four
   stateless methods in latent space.
3. **Smaller arrays in CVI**: latent dimension `K ≤ L`, less computation.
4. **Testable**: each CVI method is a pure function of its inputs.
