"""Van der Pol oscillator demo: CVHM inference with frozen and estimated readout.

Synthesises 2-D latent trajectories from the Van der Pol oscillator, generates
Poisson spike counts through a known loading matrix, then runs six inference
cases:

1. **Frozen readout, orders 0, 1, and 2** — loading and bias are fixed at
   their true values while only the Matérn kernel order changes. This isolates
   the effect of smoothness: Matérn-1/2, Matérn-3/2, and Matérn-5/2.
2. **Estimated readout** — loading and bias are initialised via factor
   analysis and refined by LBFGS at each EM M-step, using order 1.
3. **Variable-length trials** — trials are truncated to different lengths,
   padded with :func:`~cvhmax.utils.pad_trials`, fitted as usual, and
   unpadded with :func:`~cvhmax.utils.unpad_trials`, using order 1.
4. **Two sessions** — the same synthetic trajectory generates independent spike
   counts in two partially overlapping neuron populations.

Usage::

    JAX_ENABLE_X64=1 python examples/demo_vdp.py
"""

import matplotlib.pyplot as plt
import numpy as np
from jax import config
from jax import numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.cvi import MAX_LOGRATE, Params, Poisson
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
    def update_readout(cls, params, y, valid_y, m, V):
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


def make_kernels(n_latents, order, *, sigma=1.0, rho=1.0, omega=0.0, s=1e-5):
    """Construct identical latent kernels except for their Matérn order."""
    return [
        HidaMatern(
            sigma=sigma,
            rho=rho,
            omega=omega,
            order=order,
            s=s,
        )
        for _ in range(n_latents)
    ]


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

    # --- Shared kernel hyperparameters ---
    # The frozen-readout comparison changes only order; all other kernel and
    # optimization hyperparameters, data, readout, and random seed are fixed.
    kernel_hyperparams = {"sigma": 1.0, "rho": 1.0, "omega": 0.0, "s": 1e-5}

    # --- Cases 1-3: Frozen readout, varying kernel order ---
    print("\n--- Frozen readout: kernel-order comparison ---")
    true_params = Params(
        C=jnp.asarray(C_true),
        d=jnp.asarray(d_true),
        R=jnp.asarray(0.0),
    )
    frozen_r2 = {}
    for order in (0, 1, 2):
        print(f"\n  Matérn order={order}")
        frozen_r2[order] = run_case(
            y,
            x_std,
            make_kernels(n_latents, order, **kernel_hyperparams),
            dt,
            observation="FrozenPoisson",
            title=(
                "CVHM on Van der Pol latents "
                f"(frozen readout, Matérn order={order})"
            ),
            out_path=f"examples/demo_vdp_frozen_order{order}.pdf",
            params=true_params,
            max_iter=50,
            cvi_iter=5,
        )
    print(
        "Frozen-readout R² by order: "
        + ", ".join(f"{order}: {r2:.4f}" for order, r2 in frozen_r2.items())
    )

    # --- Case 4: Estimated readout (C and d learned from data) ---
    print("\n--- Estimated readout (order=1) ---")
    kernels = make_kernels(n_latents, order=1, **kernel_hyperparams)
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

    # --- Case 5: Variable-length trials (pad/unpad showcase) ---
    run_padded_case(y_np, x_std, dt, n_latents, n_trials, rng)

    # --- Case 6: Shared latents with partially overlapping observations ---
    run_two_session_case()


def run_padded_case(y_np, x_std, dt, n_latents, n_trials, rng):
    """Case 5: variable-length trials with pad/unpad workflow.

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
    y_padded, valid_y, trial_lengths = pad_trials(y_list)
    print(f"Padded shape: y={y_padded.shape}, valid_y={valid_y.shape}")

    # --- Fit (the filter skips padded bins automatically via valid_y=0) ---
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
    model.fit(y_padded, valid_y=valid_y, random_state=42)

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


def run_two_session_case(
    *, T=500, dt=0.02, max_iter=100, out_path="examples/demo_vdp_sessions.pdf"
):
    """Fit two sessions with a shared VdP trajectory and overlapping neurons.

    The sessions have independent spike counts generated with one shared
    readout. Twenty percent of recorded neuron-time entries are held out.
    After generation, the latent truth is used only for evaluating recovery;
    it is not supplied to the fit.

    Returns
    -------
    tuple[CVHM, dict]
        Fitted model and session-level recovery statistics.
    """
    print("\n--- Two sessions with partially overlapping neurons ---")
    rng = np.random.default_rng(2024)
    latent, _, _ = standardize(simulate_vdp(1, T, dt, rng=rng))
    latent = latent[0]
    n_sessions, n_neurons, n_latents = 2, 60, 2

    C_true = rng.standard_normal((n_neurons, n_latents)) * 0.3
    d_true = np.full(n_neurons, np.log(5.0))
    true_rate = np.exp(latent @ C_true.T + d_true)
    counts = rng.poisson(true_rate, size=(n_sessions, T, n_neurons))

    available = np.zeros((n_sessions, n_neurons), dtype=bool)
    available[0, :40] = True
    available[1, 20:] = True
    session, time, neuron = np.indices(counts.shape)
    held_out = available[:, None, :] & ((session + time + neuron) % 5 == 0)
    valid_y = available[:, None, :] & ~held_out
    y_train = np.where(valid_y, counts, 0)
    print(
        f"Sessions: {n_sessions}; recorded neurons: {available.sum(axis=1).tolist()}; "
        f"overlap: {np.sum(available[0] & available[1])}; "
        f"held-out recorded entries: {np.count_nonzero(held_out)}"
    )

    model = CVHM(
        n_components=n_latents,
        dt=dt,
        kernels=make_kernels(n_latents, order=1),
        observation="Poisson",
        max_iter=max_iter,
        tol=1e-4,
        cvi_iter=5,
    )
    model.fit(
        jnp.asarray(y_train, dtype=jnp.float64), jnp.asarray(valid_y), random_state=42
    )
    means, covariances = map(np.asarray, model.posterior)
    C, d = np.asarray(model.params.C), np.asarray(model.params.d)

    # One truth-based rotation is shared by both sessions; it is not used to fit.
    truth = np.broadcast_to(latent, means.shape)
    U, _, Vt = np.linalg.svd(means.reshape(-1, n_latents).T @ truth.reshape(-1, n_latents))
    aligned = means @ (U @ Vt)
    latent_r2 = [
        1 - np.sum((aligned[s] - latent) ** 2) / np.sum((latent - latent.mean(axis=0)) ** 2)
        for s in range(n_sessions)
    ]

    linear = np.einsum("btk,nk->btn", means, C) + d
    variance = np.einsum("ni,btij,nj->btn", C, covariances, C)
    predicted_rate = np.exp(np.minimum(linear + 0.5 * variance, MAX_LOGRATE))
    heldout_r2 = []
    for s in range(n_sessions):
        n_observed = valid_y[s].sum(axis=0)
        baseline = np.divide(
            y_train[s].sum(axis=0), n_observed,
            out=np.zeros(n_neurons), where=n_observed > 0,
        )
        target = counts[s][held_out[s]]
        predicted = predicted_rate[s][held_out[s]]
        reference = np.broadcast_to(baseline, (T, n_neurons))[held_out[s]]
        heldout_r2.append(
            1 - np.sum((target - predicted) ** 2) / np.sum((target - reference) ** 2)
        )

    print(f"Fit: iterations={model.n_iter_}, converged={model.converged_}")
    print(f"Held-out count R² by session: {np.round(heldout_r2, 4).tolist()}")
    print(f"Latent R² by session (one truth-based rotation): {np.round(latent_r2, 4).tolist()}")

    times = np.arange(T) * dt
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(
        "Two sessions: shared VdP latents, partially overlapping neurons\n"
        f"Iterations: {model.n_iter_}; converged: {model.converged_}; "
        f"held-out count R²: {heldout_r2[0]:.3f}, {heldout_r2[1]:.3f}"
    )
    colors = ("tab:blue", "tab:orange")
    for dim in range(n_latents):
        ax = axes[0, dim]
        ax.plot(times, latent[:, dim], "k", label="True")
        for s, color in enumerate(colors):
            ax.plot(times, aligned[s, :, dim], color=color, alpha=0.75, label=f"Session {s + 1}")
        ax.set(title=f"Latent {dim + 1}: truth-aligned", xlabel="Time (s)")
        ax.legend(fontsize="small")

    ax = axes[1, 0]
    ax.plot(latent[:, 0], latent[:, 1], "k", alpha=0.6, label="True")
    for s, color in enumerate(colors):
        ax.plot(aligned[s, :, 0], aligned[s, :, 1], color=color, alpha=0.7, label=f"Session {s + 1}")
    ax.set(title="Shared latent coordinates", xlabel="Latent 1", ylabel="Latent 2")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    ax.imshow(available, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax.set(
        title="Neuron availability (black = recorded)", xlabel="Neuron index",
        yticks=[0, 1], yticklabels=["Session 1", "Session 2"],
    )

    shared_neuron = 30
    for s, color in enumerate(colors):
        ax = axes[2, s]
        mask = held_out[s, :, shared_neuron]
        ax.plot(times, true_rate[:, shared_neuron], "k--", alpha=0.6, label="True rate")
        ax.plot(times, predicted_rate[s, :, shared_neuron], color=color, label="Predicted rate")
        ax.scatter(
            times[mask], counts[s, mask, shared_neuron], color=color, s=9,
            alpha=0.45, label="Held-out counts",
        )
        ax.set(
            title=f"Session {s + 1}: shared neuron {shared_neuron}",
            xlabel="Time (s)", ylabel="Count / bin",
        )
        ax.legend(fontsize="small")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return model, {
        "heldout_r2": heldout_r2,
        "latent_r2": latent_r2,
        "recorded_neurons": available.sum(axis=1).tolist(),
        "overlap_neurons": int(np.sum(available[0] & available[1])),
    }


if __name__ == "__main__":
    main()
