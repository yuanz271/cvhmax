"""Automatic relevance determination: how many latents does the data need?

Simulates two Van der Pol latent trajectories, observes them through a
Gaussian readout, and fits a model with more latents than the data contains.
The same data, kernels, and initialisation are used with two loading kinds:

* ``"ridge"`` holds every loading precision at a fixed value, so the surplus
  latents are shrunk but keep a small nonzero loading.
* ``"ard"`` fits one precision per latent against the evidence, so the
  loadings of surplus latents are driven to numerical zero and their
  precisions separate from those of the latents the data needs.

The ranking of the fitted precisions, ``model.params.alpha``, is how the
effective dimensionality is read off. The absolute value of a precision depends
on the latent scale and on the guard prior's constants, so any threshold is a
choice made per fit.

ARD needs a conjugate loading representation, so it is available for the
Gaussian readout and not for Poisson.

Usage::

    JAX_ENABLE_X64=1 python examples/demo_ard.py
"""

import matplotlib.pyplot as plt
import numpy as np
from jax import config
from jax import numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.hm import HidaMatern

config.update("jax_enable_x64", True)

N_TRUE = 2  # latents that generate the data
N_FIT = 5  # latents in the fitted model


def simulate_vdp(n_sequences, n_time, dt, rng, mu=1.0, noise_std=0.05):
    """Euler-Maruyama integration of the Van der Pol oscillator.

    Returns
    -------
    ndarray, shape (n_sequences, n_time, 2)
        Standardised trajectories (position, velocity).
    """
    sqrt_dt = np.sqrt(dt)
    x = np.empty((n_sequences, n_time, 2))
    for sequence in range(n_sequences):
        state = np.array([2.0, 0.0]) + rng.normal(0.0, 0.1, size=2)
        for t in range(n_time):
            x[sequence, t] = state
            drift = np.array([state[1], mu * (1 - state[0] ** 2) * state[1] - state[0]])
            state = state + drift * dt + noise_std * sqrt_dt * rng.standard_normal(2)
    flat = x.reshape(-1, 2)
    return (x - flat.mean(axis=0)) / flat.std(axis=0)


def fit(y, loading, dt):
    """Fit an over-specified model with the given loading kind."""
    kernels = [
        HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1, s=1e-5) for _ in range(N_FIT)
    ]
    model = CVHM(
        n_components=N_FIT,
        dt=dt,
        kernels=kernels,
        observation="Gaussian",
        loading=loading,
        max_iter=500,
        tol=1e-3,
        min_iter=5,
        convergence_patience=3,
    )
    return model.fit(y, random_state=0)


def latent_recovery(model, truth):
    """R² of the true latents regressed on the two strongest fitted latents.

    A fitted latent is identified only up to a linear map, so the comparison
    regresses each true latent on the retained fitted means.
    """
    norms = np.linalg.norm(np.asarray(model.params.C), axis=0)
    keep = np.argsort(norms)[::-1][:N_TRUE]
    means = np.asarray(model.posterior[0])[..., keep].reshape(-1, N_TRUE)
    design = np.column_stack([means, np.ones(len(means))])
    target = truth.reshape(-1, N_TRUE)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return 1.0 - residual.var(axis=0) / target.var(axis=0), keep


def main():
    rng = np.random.default_rng(2024)
    n_sequences, n_time, dt = 4, 500, 0.02
    n_features, noise_std = 30, 0.4

    truth = simulate_vdp(n_sequences, n_time, dt, rng)
    loadings = rng.standard_normal((n_features, N_TRUE))
    loadings /= np.linalg.norm(loadings, axis=0, keepdims=True)
    y = truth @ loadings.T + noise_std * rng.standard_normal(
        (n_sequences, n_time, n_features)
    )
    y = jnp.asarray(y)
    print(
        f"{N_TRUE} true latents, {N_FIT} fitted latents; "
        f"data {tuple(y.shape)} (sequences, time, features)"
    )

    fits = {name: fit(y, name, dt) for name in ("ridge", "ard")}

    results = {}
    for name, model in fits.items():
        norms = np.linalg.norm(np.asarray(model.params.C), axis=0)
        r2, keep = latent_recovery(model, truth)
        results[name] = (norms, r2, keep)
        print(f"\n{name}: {model.n_iter_} iterations, converged={model.converged_}")
        print("  loading norm per latent :", np.array2string(norms, precision=3))
        print("  precision per latent    :", np.array2string(np.asarray(model.params.alpha), precision=3))
        print(f"  true-latent R² on the {N_TRUE} strongest fitted latents:",
              np.array2string(r2, precision=3))

    ard = fits["ard"]
    precision = np.asarray(ard.params.alpha)
    order = np.argsort(precision)
    print(
        "\nARD ranking (smallest precision first): "
        + ", ".join(f"latent {k}: {precision[k]:.3g}" for k in order)
    )
    print(
        f"Precision gap between the {N_TRUE}nd and {N_TRUE + 1}rd ranked latent: "
        f"{precision[order[N_TRUE]] / precision[order[N_TRUE - 1]]:.3g}x"
    )

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    index = np.arange(N_FIT)

    axes[0].bar(index, np.log10(precision), color="tab:blue")
    axes[0].set_xticks(index)
    axes[0].set_xlabel("fitted latent")
    axes[0].set_ylabel(r"$\log_{10}$ fitted precision $\alpha_k$")
    axes[0].set_title("ARD precisions")

    width = 0.38
    axes[1].bar(
        index - width / 2, results["ridge"][0], width, label="ridge", color="tab:gray"
    )
    axes[1].bar(
        index + width / 2, results["ard"][0], width, label="ARD", color="tab:blue"
    )
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_xticks(index)
    axes[1].set_xlabel("fitted latent")
    axes[1].set_ylabel(r"loading norm $\|C_{:,k}\|$ (symlog)")
    axes[1].set_title("Surplus loadings: shrunk vs. removed")
    axes[1].legend()

    keep = results["ard"][2]
    means = np.asarray(ard.posterior[0])[0][:, keep]
    design = np.column_stack([means, np.ones(len(means))])
    fitted, *_ = np.linalg.lstsq(design, truth[0], rcond=None)
    recovered = design @ fitted
    time = np.arange(n_time) * dt
    for k, label in enumerate(("position", "velocity")):
        axes[2].plot(time, truth[0][:, k], color=f"C{k}", lw=1.2, label=f"true {label}")
        axes[2].plot(time, recovered[:, k], color=f"C{k}", lw=1.2, ls="--", label=f"ARD {label}")
    axes[2].set_xlabel("time")
    axes[2].set_title("Latents recovered by ARD (sequence 0)")
    axes[2].legend(fontsize=8, ncol=2)

    fig.suptitle(f"ARD with {N_FIT} fitted latents, {N_TRUE} in the data")
    fig.tight_layout()
    fig.savefig("examples/demo_ard.pdf")
    print("\nSaved examples/demo_ard.pdf")


if __name__ == "__main__":
    main()
