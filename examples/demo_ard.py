"""Automatic relevance determination: how many latents does the data need?

Simulates Van der Pol latent trajectories and fits more latents than the data
contains, under two observation models:

* **Gaussian**, with a moderate noise level.
* **Poisson** counts, using the Laplace approximation ARD needs there.

Each is fitted with two loading kinds on the same data:

* ``"ridge"`` holds every loading precision at a fixed value, so the surplus
  latents are shrunk but keep a small nonzero loading.
* ``"ard"`` fits one precision per latent against the evidence, so the
  loadings of surplus latents are driven to numerical zero and their
  precisions separate from those of the latents the data needs.

The ranking of the fitted precisions, ``model.params.alpha``, is how the
effective dimensionality is read off. The absolute value of a precision depends
on the latent scale and on the guard prior's constants, so any threshold is a
choice made per fit.

The Poisson section uses a smaller dataset and fewer fixed-point iterations
than the Gaussian one: each Poisson M-step differentiates the objective per
feature and time step, which costs more per iteration and makes the default
``iterations=50`` needlessly expensive for a demonstration.

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


def fit(y, loading, dt, observation, *, loading_config=None, cvi_iter=20):
    """Fit an over-specified model with the given loading kind."""
    kernels = [
        HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1, s=1e-5) for _ in range(N_FIT)
    ]
    model = CVHM(
        n_components=N_FIT,
        dt=dt,
        kernels=kernels,
        observation=observation,
        loading=loading,
        loading_config=loading_config,
        cvi_iter=cvi_iter,
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


def run_case(y, truth, dt, observation, *, loading_config=None, cvi_iter=20):
    """Fit both loading kinds on one dataset and report what each retained."""
    print(
        f"\n{observation}: {N_TRUE} true latents, {N_FIT} fitted; "
        f"data {tuple(y.shape)} (sequences, time, features)"
    )
    results = {}
    for loading in ("ridge", "ard"):
        config_ = loading_config if loading == "ard" else None
        model = fit(y, loading, dt, observation, loading_config=config_, cvi_iter=cvi_iter)
        norms = np.linalg.norm(np.asarray(model.params.C), axis=0)
        precision = np.asarray(model.params.alpha)
        r2, keep = latent_recovery(model, truth)
        results[loading] = {
            "model": model,
            "norms": norms,
            "precision": precision,
            "r2": r2,
            "keep": keep,
        }
        print(f"  {loading}: {model.n_iter_} iterations, converged={model.converged_}")
        print("    loading norm per latent :", np.array2string(norms, precision=3))
        print("    precision per latent    :", np.array2string(precision, precision=3))
        print(
            f"    true-latent R² on the {N_TRUE} strongest fitted latents:",
            np.array2string(r2, precision=3),
        )

    ard = results["ard"]
    order = np.argsort(ard["precision"])
    ranked = ard["precision"][order]
    print(
        "  ARD ranking (smallest precision first): "
        + ", ".join(f"latent {k}: {ard['precision'][k]:.3g}" for k in order)
    )
    print(
        f"  precision gap between rank {N_TRUE} and {N_TRUE + 1}: "
        f"{ranked[N_TRUE] / ranked[N_TRUE - 1]:.3g}x"
    )
    return results


def plot_case(axes, results, truth, observation, dt):
    """One row: fitted precisions, surplus loadings, recovered latents."""
    ard = results["ard"]
    model = ard["model"]
    index = np.arange(N_FIT)

    axes[0].bar(index, np.log10(ard["precision"]), color="tab:blue")
    axes[0].set_xticks(index)
    axes[0].set_xlabel("fitted latent")
    axes[0].set_ylabel(r"$\log_{10}$ fitted precision $\alpha_k$")
    axes[0].set_title(f"{observation}: ARD precisions")

    width = 0.38
    axes[1].bar(index - width / 2, results["ridge"]["norms"], width,
                label="ridge", color="tab:gray")
    axes[1].bar(index + width / 2, ard["norms"], width, label="ARD", color="tab:blue")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_xticks(index)
    axes[1].set_xlabel("fitted latent")
    axes[1].set_ylabel(r"loading norm $\|C_{:,k}\|$ (symlog)")
    axes[1].set_title(f"{observation}: surplus loadings")
    axes[1].legend()

    keep = ard["keep"]
    means = np.asarray(model.posterior[0])[0][:, keep]
    design = np.column_stack([means, np.ones(len(means))])
    coefficients, *_ = np.linalg.lstsq(design, truth[0], rcond=None)
    recovered = design @ coefficients
    time = np.arange(len(recovered)) * dt
    for k, label in enumerate(("position", "velocity")):
        axes[2].plot(time, truth[0][:, k], color=f"C{k}", lw=1.2, label=f"true {label}")
        axes[2].plot(time, recovered[:, k], color=f"C{k}", lw=1.2, ls="--",
                     label=f"ARD {label}")
    axes[2].set_xlabel("time")
    axes[2].set_title(f"{observation}: latents recovered by ARD (sequence 0)")
    axes[2].legend(fontsize=8, ncol=2)


def main():
    rng = np.random.default_rng(2024)
    dt = 0.02

    # --- Gaussian: 2 true latents, 5 fitted ---
    n_sequences, n_time, n_features = 4, 500, 30
    truth = simulate_vdp(n_sequences, n_time, dt, rng)
    loadings = rng.standard_normal((n_features, N_TRUE))
    loadings /= np.linalg.norm(loadings, axis=0, keepdims=True)
    y_gaussian = jnp.asarray(
        truth @ loadings.T + 0.4 * rng.standard_normal((n_sequences, n_time, n_features))
    )
    gaussian = run_case(y_gaussian, truth, dt, "Gaussian")

    # --- Poisson: the same question on count data, with a smaller dataset ---
    n_sequences, n_time, n_features = 2, 200, 8
    truth_counts = simulate_vdp(n_sequences, n_time, dt, rng)
    loadings_counts = rng.standard_normal((n_features, N_TRUE))
    loadings_counts /= np.linalg.norm(loadings_counts, axis=0, keepdims=True)
    rates = np.exp(1.5 + 0.8 * truth_counts @ loadings_counts.T)
    y_poisson = jnp.asarray(rng.poisson(rates).astype(float))
    poisson = run_case(
        y_poisson, truth_counts, dt, "Poisson",
        loading_config={"iterations": 5}, cvi_iter=10,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plot_case(axes[0], gaussian, truth, "Gaussian", dt)
    plot_case(axes[1], poisson, truth_counts, "Poisson", dt)
    fig.suptitle(f"ARD with {N_FIT} fitted latents, {N_TRUE} in the data")
    fig.tight_layout()
    fig.savefig("examples/demo_ard.pdf")
    print("\nSaved examples/demo_ard.pdf")


if __name__ == "__main__":
    main()
