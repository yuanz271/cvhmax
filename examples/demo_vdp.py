"""Van der Pol oscillator demo: CVHM inference with frozen and estimated readout.

Synthesises 2-D latent trajectories from the Van der Pol oscillator, generates
Poisson spike counts through a known loading matrix, then runs three inference
cases:

1. **Frozen readout** — loading and bias are fixed at their true values,
   isolating filtering quality from readout estimation.
2. **Estimated readout** — loading and bias are initialised via factor
   analysis and refined by LBFGS at each EM M-step.
3. **Variable-length trials** — trials are truncated to different lengths,
   padded with :func:`~cvhmax.utils.pad_trials`, fitted as usual, and
   unpadded with :func:`~cvhmax.utils.unpad_trials`.

Usage::

    JAX_ENABLE_X64=1 python examples/demo_vdp.py
"""

import numpy as np
import matplotlib.pyplot as plt

from jax import config, numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.cvi import Params, Poisson
from cvhmax.hm import HidaMatern
from cvhmax.utils import pad_trials, unpad_trials

config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Frozen Poisson readout: keeps C and d at their true values
# ---------------------------------------------------------------------------


class FrozenPoisson(Poisson):
    """Poisson CVI that keeps readout parameters fixed.

    Pass pre-set ``Params`` via ``CVHM(params=...)`` and this subclass
    will preserve them throughout fitting — ``initialize_params`` returns
    them as-is (via the base class ``params`` argument) and
    ``update_readout`` is a no-op.
    """

    @classmethod
    def update_readout(cls, params, y, ymask, m, V):
        return params, jnp.nan


# ---------------------------------------------------------------------------
# Van der Pol oscillator
# ---------------------------------------------------------------------------


def vdp_rhs(t, state, mu):
    """Right-hand side of the Van der Pol ODE."""
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return [dxdt, dvdt]


def simulate_vdp(n_trials, T, dt, mu=1.0, noise_std=0.05, rng=None):
    """Integrate the Van der Pol SDE via Euler-Maruyama.

    At each time step, Gaussian state noise with standard deviation
    ``noise_std * sqrt(dt)`` is added to both state dimensions.

    Parameters
    ----------
    n_trials : int
        Number of independent trajectories.
    T : int
        Number of time bins per trial.
    dt : float
        Sampling interval.
    mu : float
        Nonlinearity parameter.
    noise_std : float
        Diffusion coefficient (state noise intensity).
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    x : ndarray, shape (n_trials, T, 2)
        Latent trajectories (position, velocity).
    """
    if rng is None:
        rng = np.random.default_rng()

    sqrt_dt = np.sqrt(dt)
    trajectories = np.empty((n_trials, T, 2))

    for trial in range(n_trials):
        state = np.array([2.0 + rng.normal(0, 0.1), 0.0 + rng.normal(0, 0.1)])
        for t in range(T):
            trajectories[trial, t] = state
            drift = np.array(vdp_rhs(None, state, mu))
            state = state + drift * dt + noise_std * sqrt_dt * rng.standard_normal(2)

    return trajectories


def standardize(x):
    """Zero-mean, unit-variance standardisation over trials and time.

    Parameters
    ----------
    x : ndarray, shape (n_trials, T, D)

    Returns
    -------
    x_std : ndarray, same shape
    mean : ndarray, shape (D,)
    std : ndarray, shape (D,)
    """
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    return (x - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Fit, evaluate, and plot
# ---------------------------------------------------------------------------


def run_case(
    y,
    x_std,
    kernels,
    dt,
    observation,
    title,
    out_path,
    *,
    params=None,
    max_iter=50,
    cvi_iter=5,
    random_state=42,
):
    """Fit a CVHM model and save a diagnostic figure.

    Parameters
    ----------
    y : Array
        Observations shaped ``(n_trials, T, n_obs)``.
    x_std : ndarray
        Standardised true latents shaped ``(n_trials, T, n_latents)``.
    kernels : list[HidaMatern]
        One kernel per latent component.
    dt : float
        Sampling interval.
    observation : str
        CVI subclass name (e.g. ``"FrozenPoisson"``, ``"Poisson"``).
    title : str
        Figure super-title.
    out_path : str
        Path where the PDF figure is saved.
    params : Params or None, optional
        Pre-set readout parameters.  When provided the model skips
        initialisation and uses these directly.
    max_iter : int
        Number of outer EM iterations.
    cvi_iter : int
        Number of inner CVI iterations per EM step.
    random_state : int
        Seed for initialisation.

    Returns
    -------
    float
        Pooled R-squared between aligned posterior and true latents.
    """
    n_trials, T, n_latents = x_std.shape

    model = CVHM(
        n_components=n_latents,
        dt=dt,
        kernels=kernels,
        observation=observation,
        max_iter=max_iter,
        cvi_iter=cvi_iter,
    )
    if params is not None:
        model.params = params

    model.fit(y, random_state=random_state)

    # --- Extract posterior and align ---
    m = np.asarray(model.posterior[0])  # (n_trials, T, n_latents)

    m_flat = m.reshape(-1, n_latents)
    x_flat = x_std.reshape(-1, n_latents)
    ones = np.ones((m_flat.shape[0], 1))
    A = np.hstack([m_flat, ones])
    W, _, _, _ = np.linalg.lstsq(A, x_flat, rcond=None)
    m_aligned = (A @ W).reshape(n_trials, T, n_latents)

    # --- R-squared ---
    ss_res = np.sum((m_aligned.reshape(-1, n_latents) - x_flat) ** 2)
    ss_tot = np.sum((x_flat - x_flat.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(f"R² (pooled): {r2:.4f}")

    for dim in range(n_latents):
        ss_res_d = np.sum((m_aligned[..., dim] - x_std[..., dim]) ** 2)
        ss_tot_d = np.sum((x_std[..., dim] - x_std[..., dim].mean()) ** 2)
        print(f"  Latent {dim}: R² = {1.0 - ss_res_d / ss_tot_d:.4f}")

    # --- Plots ---
    trial_idx = 0
    t_axis = np.arange(T) * dt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title)

    # Time courses
    for dim, ax in enumerate(axs[0]):
        ax.plot(t_axis, x_std[trial_idx, :, dim], "b", alpha=0.7, label="True")
        ax.plot(t_axis, m_aligned[trial_idx, :, dim], "r", alpha=0.7, label="Inferred")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Latent {dim + 1}")
        ax.set_title(f"Latent dim {dim + 1} — trial {trial_idx}")
        ax.legend(loc="upper right", fontsize="small")

    # 2D phase portrait — single trial
    ax = axs[1, 0]
    ax.plot(
        x_std[trial_idx, :, 0], x_std[trial_idx, :, 1], "b", alpha=0.5, label="True"
    )
    ax.plot(
        m_aligned[trial_idx, :, 0],
        m_aligned[trial_idx, :, 1],
        "r",
        alpha=0.5,
        label="Inferred",
    )
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title(f"Phase portrait — trial {trial_idx}")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    # 2D phase portrait — multiple trials
    ax = axs[1, 1]
    n_show = min(4, n_trials)
    for i in range(n_show):
        alpha = 0.4
        ax.plot(x_std[i, :, 0], x_std[i, :, 1], "b", alpha=alpha, linewidth=0.8)
        ax.plot(m_aligned[i, :, 0], m_aligned[i, :, 1], "r", alpha=alpha, linewidth=0.8)
    ax.plot([], [], "b", label="True")
    ax.plot([], [], "r", label="Inferred")
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title(f"Phase portrait — trials 0..{n_show - 1}")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

    return r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # --- Simulation parameters ---
    rng = np.random.default_rng(2024)
    n_trials = 10
    T = 500
    dt = 0.02
    mu = 1.0
    n_obs = 50
    n_latents = 2

    # --- Synthesise Van der Pol latents ---
    x_raw = simulate_vdp(n_trials, T, dt, mu=mu, rng=rng)
    x_std, _, _ = standardize(x_raw)

    # --- True readout ---
    C_true = rng.standard_normal((n_obs, n_latents))
    C_true /= np.linalg.norm(C_true, axis=0, keepdims=True)

    baseline_rate = 5.0
    d_true = np.full(n_obs, np.log(baseline_rate))

    # --- Generate Poisson observations ---
    log_rate = np.einsum("ntl,ol->nto", x_std, C_true) + d_true[None, None, :]
    y_np = rng.poisson(np.exp(log_rate))
    y = jnp.asarray(y_np, dtype=jnp.float64)

    print(f"Latents: {x_std.shape}  Observations: {y.shape}")

    # --- Shared kernels ---
    kernels = [
        HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1) for _ in range(n_latents)
    ]

    # --- Case 1: Frozen readout (true C and d) ---
    print("\n--- Frozen readout ---")
    true_params = Params(
        C=jnp.asarray(C_true),
        d=jnp.asarray(d_true),
        R=None,
        M=CVHM(
            n_components=n_latents, dt=dt, kernels=kernels, observation="FrozenPoisson"
        ).latent_mask(),
    )
    run_case(
        y,
        x_std,
        kernels,
        dt,
        observation="FrozenPoisson",
        title="CVHM on Van der Pol latents (frozen readout)",
        out_path="examples/demo_vdp_frozen.pdf",
        params=true_params,
        max_iter=50,
        cvi_iter=5,
    )

    # --- Case 2: Estimated readout (C and d learned from data) ---
    print("\n--- Estimated readout ---")
    run_case(
        y,
        x_std,
        kernels,
        dt,
        observation="Poisson",
        title="CVHM on Van der Pol latents (estimated readout)",
        out_path="examples/demo_vdp_estimated.pdf",
        max_iter=50,
        cvi_iter=5,
    )

    # --- Case 3: Variable-length trials (pad/unpad showcase) ---
    run_padded_case(y_np, x_std, dt, n_latents, n_trials, rng)


def run_padded_case(y_np, x_std, dt, n_latents, n_trials, rng):
    """Case 3: variable-length trials with pad/unpad workflow.

    Truncates existing equal-length trials to random lengths, pads them
    into a rectangular array, fits the model, and unpads the posterior.
    """
    print("\n--- Variable-length trials (padded) ---")

    # Simulate unequal trial lengths by truncating existing data
    trial_lengths_np = rng.integers(250, 501, size=n_trials)
    y_list = [
        jnp.asarray(y_np[i, :tl], dtype=jnp.float64)
        for i, tl in enumerate(trial_lengths_np)
    ]
    x_list = [x_std[i, :tl] for i, tl in enumerate(trial_lengths_np)]
    print(f"Trial lengths: {trial_lengths_np}")

    # --- Pad to rectangular arrays ---
    y_padded, ymask, trial_lengths = pad_trials(y_list)
    print(f"Padded shape: y={y_padded.shape}, ymask={ymask.shape}")

    # --- Fit (the filter skips padded bins automatically via ymask=0) ---
    kernels = [
        HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1) for _ in range(n_latents)
    ]
    model = CVHM(
        n_components=n_latents,
        dt=dt,
        kernels=kernels,
        observation="Poisson",
        max_iter=50,
        cvi_iter=5,
    )
    model.fit(y_padded, ymask=ymask, random_state=42)

    # --- Unpad posterior back to per-trial arrays ---
    m_list = unpad_trials(model.posterior[0], trial_lengths)
    V_list = unpad_trials(model.posterior[1], trial_lengths)
    for i in range(n_trials):
        print(
            f"  Trial {i}: T={trial_lengths_np[i]}, "
            f"m={m_list[i].shape}, V={V_list[i].shape}"
        )

    # --- R-squared on concatenated unpadded posteriors ---
    m_all = np.concatenate([np.asarray(mi) for mi in m_list])
    x_all = np.concatenate(x_list)
    ones = np.ones((m_all.shape[0], 1))
    A = np.hstack([m_all, ones])
    W, _, _, _ = np.linalg.lstsq(A, x_all, rcond=None)
    m_aligned_all = A @ W

    ss_res = np.sum((m_aligned_all - x_all) ** 2)
    ss_tot = np.sum((x_all - x_all.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    print(f"R² (pooled, variable-length): {r2:.4f}")

    # --- Plot multiple trials of various length ---
    n_show = min(4, n_trials)
    trial_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    # Pre-compute aligned posteriors for shown trials
    aligned = []
    for i in range(n_show):
        T_i = int(trial_lengths_np[i])
        m_i = np.asarray(m_list[i])
        A_i = np.hstack([m_i, np.ones((T_i, 1))])
        aligned.append(A_i @ W)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("CVHM on Van der Pol latents (variable-length trials, padded)")

    # Top row: time courses for each latent dimension
    for dim, ax in enumerate(axs[0]):
        for i in range(n_show):
            T_i = int(trial_lengths_np[i])
            t_ax = np.arange(T_i) * dt
            c = trial_colors[i]
            ax.plot(t_ax, x_list[i][:, dim], color=c, ls="-", alpha=0.6, lw=0.8)
            ax.plot(t_ax, aligned[i][:, dim], color=c, ls="--", alpha=0.9, lw=0.8)
        ax.plot([], [], "k-", alpha=0.6, label="True")
        ax.plot([], [], "k--", alpha=0.9, label="Inferred")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Latent {dim + 1}")
        ax.set_title(f"Latent dim {dim + 1}")
        ax.legend(loc="upper right", fontsize="small")

    # Bottom-left: phase portrait — single shortest trial
    shortest = int(np.argmin(trial_lengths_np[:n_show]))
    ax = axs[1, 0]
    c = trial_colors[shortest]
    T_s = int(trial_lengths_np[shortest])
    ax.plot(
        x_list[shortest][:, 0],
        x_list[shortest][:, 1],
        color=c,
        ls="-",
        alpha=0.6,
        label="True",
    )
    ax.plot(
        aligned[shortest][:, 0],
        aligned[shortest][:, 1],
        color=c,
        ls="--",
        alpha=0.9,
        label="Inferred",
    )
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title(f"Phase portrait — trial {shortest} (T={T_s})")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    # Bottom-right: phase portrait — multiple trials
    ax = axs[1, 1]
    for i in range(n_show):
        c = trial_colors[i]
        T_i = int(trial_lengths_np[i])
        ax.plot(
            x_list[i][:, 0],
            x_list[i][:, 1],
            color=c,
            ls="-",
            alpha=0.5,
            lw=0.8,
        )
        ax.plot(
            aligned[i][:, 0],
            aligned[i][:, 1],
            color=c,
            ls="--",
            alpha=0.8,
            lw=0.8,
        )
    ax.plot([], [], "k-", alpha=0.5, label="True")
    ax.plot([], [], "k--", alpha=0.8, label="Inferred")
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title(f"Phase portrait — trials 0..{n_show - 1}")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    fig.savefig("examples/demo_vdp_padded.pdf")
    plt.close(fig)
    print("Saved examples/demo_vdp_padded.pdf")

    return r2


if __name__ == "__main__":
    main()
