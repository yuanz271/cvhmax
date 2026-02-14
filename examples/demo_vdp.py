"""Van der Pol oscillator demo: evaluate CVHM filtering with known readout.

Synthesises 2-D latent trajectories from the Van der Pol oscillator, generates
Poisson spike counts through a known loading matrix, then runs CVHM inference
with the loading and bias fixed at their true values.  This isolates the
filtering quality from readout estimation.

Usage::

    JAX_ENABLE_X64=1 python examples/demo_vdp.py
"""

import numpy as np
import matplotlib.pyplot as plt

from jax import config, numpy as jnp

from cvhmax.cvhm import CVHM
from cvhmax.cvi import Params, Poisson
from cvhmax.hm import HidaMatern

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
    # Column-normalise so norm_loading(C_true) == C_true
    C_true /= np.linalg.norm(C_true, axis=0, keepdims=True)

    baseline_rate = 5.0
    d_true = np.full(n_obs, np.log(baseline_rate))

    # --- Generate Poisson observations ---
    log_rate = np.einsum("ntl,ol->nto", x_std, C_true) + d_true[None, None, :]
    y_np = rng.poisson(np.exp(log_rate))
    y = jnp.asarray(y_np, dtype=jnp.float64)

    print(f"Latents: {x_std.shape}  Observations: {y.shape}")

    # --- Build and fit model with frozen readout ---
    kernels = [
        HidaMatern(sigma=1.0, rho=1.0, omega=0.0, order=1) for _ in range(n_latents)
    ]
    model = CVHM(
        n_components=n_latents,
        dt=dt,
        kernels=kernels,
        observation="FrozenPoisson",
        max_iter=50,
        cvi_iter=5,
    )
    # Pre-set params so initialize_params returns them as-is
    model.params = Params(
        C=jnp.asarray(C_true),
        d=jnp.asarray(d_true),
        R=None,
        M=model.latent_mask(),
    )
    model.fit(y, random_state=42)

    # --- Extract posterior ---
    m, V = model.posterior
    m = np.asarray(m)  # (n_trials, T, n_latents)

    # --- Align inferred latents to true via least-squares ---
    # The GP prior controls latent scale independently of C, so a linear
    # transform is needed to map inferred coordinates back to true ones.
    m_flat = m.reshape(-1, n_latents)
    x_flat = x_std.reshape(-1, n_latents)
    # Solve x_flat ≈ [m_flat, 1] @ W  (affine map)
    ones = np.ones((m_flat.shape[0], 1))
    A = np.hstack([m_flat, ones])
    W, _, _, _ = np.linalg.lstsq(A, x_flat, rcond=None)
    m_aligned = (A @ W).reshape(n_trials, T, n_latents)

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
    fig.suptitle("CVHM inference on Van der Pol latents (frozen readout)")

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
    # Manual legend entries
    ax.plot([], [], "b", label="True")
    ax.plot([], [], "r", label="Inferred")
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title(f"Phase portrait — trials 0..{n_show - 1}")
    ax.legend(fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")

    out_path = "examples/demo_vdp.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
