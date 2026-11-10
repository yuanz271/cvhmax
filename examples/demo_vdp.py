"""Van der Pol oscillator demo: CVHM inference with frozen and estimated readout.

Synthesises 2-D latent trajectories from the Van der Pol oscillator, generates
Poisson counts through a known loading matrix, then runs six inference
cases:

1. **Frozen readout, orders 0, 1, and 2** — loading and bias are fixed at
   their true values while only the Matérn kernel order changes. This isolates
   the effect of smoothness: Matérn-1/2, Matérn-3/2, and Matérn-5/2.
2. **Estimated readout** — loading and bias are initialised via factor
   analysis and refined by LBFGS at each EM M-step, using order 1.
3. **Variable-length sequences** — sequences are truncated to different lengths,
   padded with :func:`~cvhmax.utils.pad_sequences`, fitted as usual, and
   unpadded with :func:`~cvhmax.utils.unpad_sequences`, using order 1.
4. **Two sequences** — the same synthetic trajectory generates independent
   counts with partially overlapping features.

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
from cvhmax.utils import pad_sequences, unpad_sequences

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


def simulate_vdp(n_sequences, T, dt, mu=1.0, noise_std=0.05, rng=None):
    """Integrate the Van der Pol SDE via Euler-Maruyama.

    At each time step, Gaussian state noise with standard deviation
    ``noise_std * sqrt(dt)`` is added to both state dimensions.

    Parameters
    ----------
    n_sequences : int
        Number of independent trajectories.
    T : int
        Number of time steps per sequence.
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
    x : ndarray, shape (n_sequences, T, 2)
        Latent trajectories (position, velocity).
    """
    if rng is None:
        rng = np.random.default_rng()

    sqrt_dt = np.sqrt(dt)
    trajectories = np.empty((n_sequences, T, 2))

    for sequence in range(n_sequences):
        state = np.array([2.0 + rng.normal(0, 0.1), 0.0 + rng.normal(0, 0.1)])
        for t in range(T):
            trajectories[sequence, t] = state
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
    """Zero-mean, unit-variance standardisation over sequences and time.

    Parameters
    ----------
    x : ndarray, shape (n_sequences, T, D)

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
        Observations shaped ``(n_sequences, T, n_obs)``.
    x_std : ndarray
        Standardised true latents shaped ``(n_sequences, T, n_latents)``.
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
    n_sequences, T, n_latents = x_std.shape

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
    m = np.asarray(model.posterior[0])  # (n_sequences, T, n_latents)

    m_flat = m.reshape(-1, n_latents)
    x_flat = x_std.reshape(-1, n_latents)
    ones = np.ones((m_flat.shape[0], 1))
    A = np.hstack([m_flat, ones])
    W, _, _, _ = np.linalg.lstsq(A, x_flat, rcond=None)
    m_aligned = (A @ W).reshape(n_sequences, T, n_latents)

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
    sequence_idx = 0
    t_axis = np.arange(T) * dt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(title)

    # Time courses
    for dim, ax in enumerate(axs[0]):
        ax.plot(t_axis, x_std[sequence_idx, :, dim], "b", alpha=0.7, label="True")
        ax.plot(t_axis, m_aligned[sequence_idx, :, dim], "r", alpha=0.7, label="Inferred")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Latent {dim + 1}")
        ax.set_title(f"Latent dim {dim + 1} — sequence {sequence_idx}")
        ax.legend(loc="upper right", fontsize="small")

    # 2D phase portrait — single sequence
    ax = axs[1, 0]
    ax.plot(
        x_std[sequence_idx, :, 0], x_std[sequence_idx, :, 1], "b", alpha=0.5, label="True"
    )
    ax.plot(
        m_aligned[sequence_idx, :, 0],
        m_aligned[sequence_idx, :, 1],
        "r",
        alpha=0.5,
        label="Inferred",
    )
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title(f"Phase portrait — sequence {sequence_idx}")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    # 2D phase portrait — multiple sequences
    ax = axs[1, 1]
    n_show = min(4, n_sequences)
    for i in range(n_show):
        alpha = 0.4
        ax.plot(x_std[i, :, 0], x_std[i, :, 1], "b", alpha=alpha, linewidth=0.8)
        ax.plot(m_aligned[i, :, 0], m_aligned[i, :, 1], "r", alpha=alpha, linewidth=0.8)
    ax.plot([], [], "b", label="True")
    ax.plot([], [], "r", label="Inferred")
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title(f"Phase portrait — sequences 0..{n_show - 1}")
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
    n_sequences = 10
    T = 500
    dt = 0.02
    mu = 1.0
    n_obs = 50
    n_latents = 2

    # --- Synthesise Van der Pol latents ---
    x_raw = simulate_vdp(n_sequences, T, dt, mu=mu, rng=rng)
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

    # --- Case 5: Variable-length sequences (pad/unpad showcase) ---
    run_padded_case(y_np, x_std, dt, n_latents, n_sequences, rng)

    # --- Case 6: Shared latents with partially overlapping observations ---
    run_overlapping_features_case()


def run_padded_case(y_np, x_std, dt, n_latents, n_sequences, rng):
    """Case 5: variable-length sequences with pad/unpad workflow.

    Truncates existing equal-length sequences to random lengths, pads them
    into a rectangular array, fits the model, and unpads the posterior.
    """
    print("\n--- Variable-length sequences (padded) ---")

    # Simulate unequal sequence lengths by truncating existing data
    sequence_lengths_np = rng.integers(250, 501, size=n_sequences)
    y_list = [
        jnp.asarray(y_np[i, :tl], dtype=jnp.float64)
        for i, tl in enumerate(sequence_lengths_np)
    ]
    x_list = [x_std[i, :tl] for i, tl in enumerate(sequence_lengths_np)]
    print(f"Sequence lengths: {sequence_lengths_np}")

    # --- Pad to rectangular arrays ---
    y_padded, valid_y, sequence_lengths = pad_sequences(y_list)
    print(f"Padded shape: y={y_padded.shape}, valid_y={valid_y.shape}")

    # --- Fit (the filter skips padded time steps automatically via valid_y=0) ---
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

    # --- Unpad posterior back to per-sequence arrays ---
    m_list = unpad_sequences(model.posterior[0], sequence_lengths)
    V_list = unpad_sequences(model.posterior[1], sequence_lengths)
    for i in range(n_sequences):
        print(
            f"  Sequence {i}: T={sequence_lengths_np[i]}, "
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

    # --- Plot multiple sequences of various length ---
    n_show = min(4, n_sequences)
    sequence_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]

    # Pre-compute aligned posteriors for shown sequences
    aligned = []
    for i in range(n_show):
        T_i = int(sequence_lengths_np[i])
        m_i = np.asarray(m_list[i])
        A_i = np.hstack([m_i, np.ones((T_i, 1))])
        aligned.append(A_i @ W)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("CVHM on Van der Pol latents (variable-length sequences, padded)")

    # Top row: time courses for each latent dimension
    for dim, ax in enumerate(axs[0]):
        for i in range(n_show):
            T_i = int(sequence_lengths_np[i])
            t_ax = np.arange(T_i) * dt
            c = sequence_colors[i]
            ax.plot(t_ax, x_list[i][:, dim], color=c, ls="-", alpha=0.6, lw=0.8)
            ax.plot(t_ax, aligned[i][:, dim], color=c, ls="--", alpha=0.9, lw=0.8)
        ax.plot([], [], "k-", alpha=0.6, label="True")
        ax.plot([], [], "k--", alpha=0.9, label="Inferred")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Latent {dim + 1}")
        ax.set_title(f"Latent dim {dim + 1}")
        ax.legend(loc="upper right", fontsize="small")

    # Bottom-left: phase portrait — single shortest sequence
    shortest = int(np.argmin(sequence_lengths_np[:n_show]))
    ax = axs[1, 0]
    c = sequence_colors[shortest]
    T_s = int(sequence_lengths_np[shortest])
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
    ax.set_title(f"Phase portrait — sequence {shortest} (T={T_s})")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    # Bottom-right: phase portrait — multiple sequences
    ax = axs[1, 1]
    for i in range(n_show):
        c = sequence_colors[i]
        T_i = int(sequence_lengths_np[i])
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
    ax.set_title(f"Phase portrait — sequences 0..{n_show - 1}")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    fig.savefig("examples/demo_vdp_padded.pdf")
    plt.close(fig)
    print("Saved examples/demo_vdp_padded.pdf")

    return r2


def run_overlapping_features_case(
    *, T=500, dt=0.02, max_iter=100, out_path="examples/demo_vdp_overlap.pdf"
):
    """Fit two sequences with overlapping features and a shared VdP truth.

    The sequences have independent counts generated from one trajectory and
    readout. Twenty percent of observed feature-time entries are held out.
    The fit treats the sequences as independent; the latent truth is used only
    for evaluating recovery.

    Returns
    -------
    tuple[CVHM, dict]
        Fitted model and sequence-level recovery statistics.
    """
    print("\n--- Two sequences with partially overlapping features ---")
    rng = np.random.default_rng(2024)
    latent, _, _ = standardize(simulate_vdp(1, T, dt, rng=rng))
    latent = latent[0]
    n_sequences, n_features, n_latents = 2, 60, 2

    C_true = rng.standard_normal((n_features, n_latents)) * 0.3
    d_true = np.full(n_features, np.log(5.0))
    true_rate = np.exp(latent @ C_true.T + d_true)
    counts = rng.poisson(true_rate, size=(n_sequences, T, n_features))

    available = np.zeros((n_sequences, n_features), dtype=bool)
    available[0, :40] = True
    available[1, 20:] = True
    sequence, time, feature = np.indices(counts.shape)
    held_out = available[:, None, :] & ((sequence + time + feature) % 5 == 0)
    valid_y = available[:, None, :] & ~held_out
    y_train = np.where(valid_y, counts, 0)
    print(
        f"Sequences: {n_sequences}; recorded features: {available.sum(axis=1).tolist()}; "
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

    # One truth-based rotation is shared by both sequences; it is not used to fit.
    truth = np.broadcast_to(latent, means.shape)
    U, _, Vt = np.linalg.svd(means.reshape(-1, n_latents).T @ truth.reshape(-1, n_latents))
    aligned = means @ (U @ Vt)
    latent_r2 = [
        1 - np.sum((aligned[s] - latent) ** 2) / np.sum((latent - latent.mean(axis=0)) ** 2)
        for s in range(n_sequences)
    ]

    linear = np.einsum("btk,nk->btn", means, C) + d
    variance = np.einsum("ni,btij,nj->btn", C, covariances, C)
    predicted_rate = np.exp(np.minimum(linear + 0.5 * variance, MAX_LOGRATE))
    heldout_r2 = []
    for s in range(n_sequences):
        n_observed = valid_y[s].sum(axis=0)
        baseline = np.divide(
            y_train[s].sum(axis=0), n_observed,
            out=np.zeros(n_features), where=n_observed > 0,
        )
        target = counts[s][held_out[s]]
        predicted = predicted_rate[s][held_out[s]]
        reference = np.broadcast_to(baseline, (T, n_features))[held_out[s]]
        heldout_r2.append(
            1 - np.sum((target - predicted) ** 2) / np.sum((target - reference) ** 2)
        )

    print(f"Fit: iterations={model.n_iter_}, converged={model.converged_}")
    print(f"Held-out count R² by sequence: {np.round(heldout_r2, 4).tolist()}")
    print(f"Latent R² by sequence (one truth-based rotation): {np.round(latent_r2, 4).tolist()}")

    times = np.arange(T) * dt
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle(
        "Two sequences: shared VdP latents, partially overlapping features\n"
        f"Iterations: {model.n_iter_}; converged: {model.converged_}; "
        f"held-out count R²: {heldout_r2[0]:.3f}, {heldout_r2[1]:.3f}"
    )
    colors = ("tab:blue", "tab:orange")
    for dim in range(n_latents):
        ax = axes[0, dim]
        ax.plot(times, latent[:, dim], "k", label="True")
        for s, color in enumerate(colors):
            ax.plot(times, aligned[s, :, dim], color=color, alpha=0.75, label=f"Sequence {s + 1}")
        ax.set(title=f"Latent {dim + 1}: truth-aligned", xlabel="Time (s)")
        ax.legend(fontsize="small")

    ax = axes[1, 0]
    ax.plot(latent[:, 0], latent[:, 1], "k", alpha=0.6, label="True")
    for s, color in enumerate(colors):
        ax.plot(aligned[s, :, 0], aligned[s, :, 1], color=color, alpha=0.7, label=f"Sequence {s + 1}")
    ax.set(title="Shared latent coordinates", xlabel="Latent 1", ylabel="Latent 2")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1, 1]
    ax.imshow(available, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax.set(
        title="Feature availability (black = recorded)", xlabel="Feature index",
        yticks=[0, 1], yticklabels=["Sequence 1", "Sequence 2"],
    )

    shared_feature = 30
    for s, color in enumerate(colors):
        ax = axes[2, s]
        mask = held_out[s, :, shared_feature]
        ax.plot(times, true_rate[:, shared_feature], "k--", alpha=0.6, label="True rate")
        ax.plot(times, predicted_rate[s, :, shared_feature], color=color, label="Predicted rate")
        ax.scatter(
            times[mask], counts[s, mask, shared_feature], color=color, s=9,
            alpha=0.45, label="Held-out counts",
        )
        ax.set(
            title=f"Sequence {s + 1}: shared feature {shared_feature}",
            xlabel="Time (s)", ylabel="Count / time step",
        )
        ax.legend(fontsize="small")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return model, {
        "heldout_r2": heldout_r2,
        "latent_r2": latent_r2,
        "recorded_features": available.sum(axis=1).tolist(),
        "overlap_features": int(np.sum(available[0] & available[1])),
    }


if __name__ == "__main__":
    main()
