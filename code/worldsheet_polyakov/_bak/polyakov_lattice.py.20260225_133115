\
"""
polyakov_lattice.py
-------------------
Lattice Polyakov (Euclidean, conformal gauge) baseline sampler.

Implements a Langevin sampler for the free (Gaussian) lattice action:
S = (T/2) * sum_{a,b} [ (Δσ X)^2/Δσ^2 + (Δτ X)^2/Δτ^2 ] ΔσΔτ

BCs:
- sigma periodic (default)
- tau non-periodic (default): Neumann-style boundary implemented by "clamped" neighbor
  (no wraparound). This keeps tau boundaries honest for S(δ).

This is intentionally minimalist and reviewer-safe:
- no Virasoro constraints enforced (baseline embedding fields only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass
class GridSpec:
    N_sigma: int
    N_tau: int
    delta_sigma: float = 1.0
    delta_tau: float = 1.0
    sigma_periodic: bool = True
    tau_periodic: bool = False


def _laplacian_2d(X: np.ndarray, grid: GridSpec) -> np.ndarray:
    """
    Discrete Laplacian per component, acting on X with shape (D, N_sigma, N_tau).

    sigma direction:
      periodic wrap if sigma_periodic else clamp.
    tau direction:
      periodic wrap if tau_periodic else clamp (Neumann-ish boundary).
    """
    D, Ns, Nt = X.shape
    ds2 = grid.delta_sigma ** 2
    dt2 = grid.delta_tau ** 2

    # sigma neighbors
    if grid.sigma_periodic:
        X_sp = np.roll(X, shift=-1, axis=1)
        X_sm = np.roll(X, shift=+1, axis=1)
    else:
        X_sp = X.copy()
        X_sm = X.copy()
        X_sp[:, :-1, :] = X[:, 1:, :]
        X_sp[:, -1, :] = X[:, -1, :]  # clamp
        X_sm[:, 1:, :] = X[:, :-1, :]
        X_sm[:, 0, :] = X[:, 0, :]    # clamp

    # tau neighbors
    if grid.tau_periodic:
        X_tp = np.roll(X, shift=-1, axis=2)
        X_tm = np.roll(X, shift=+1, axis=2)
    else:
        X_tp = X.copy()
        X_tm = X.copy()
        X_tp[:, :, :-1] = X[:, :, 1:]
        X_tp[:, :, -1] = X[:, :, -1]  # clamp
        X_tm[:, :, 1:] = X[:, :, :-1]
        X_tm[:, :, 0] = X[:, :, 0]    # clamp

    lap = (X_sp - 2.0 * X + X_sm) / ds2 + (X_tp - 2.0 * X + X_tm) / dt2
    return lap


def langevin_sample(
    grid: GridSpec,
    D: int,
    T: float,
    eta: float,
    n_steps_total: int,
    burn_in_steps: int,
    sample_every: int,
    n_saved_samples: int,
    rng_seed: int = 0,
    zero_mode_handling: str = "global_mean_subtract_each_step",
    noise_scale: float = 1.0,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Langevin sampler for exp(-S).
    Returns samples with shape (M, D, N_sigma, N_tau), where M = n_saved_samples.

    Notes:
    - For the free Euclidean action, ∇S is (T * ΔσΔτ) times the negative Laplacian (up to factors).
      Here we keep a simple, stable convention:
        X <- X + eta * laplacian(X) * (T * ΔσΔτ) + sqrt(2*eta)*noise
      with laplacian defined above.
    - Global mean subtraction is used to avoid drift in the zero mode.
    """
    rng = np.random.default_rng(rng_seed)
    Ns, Nt = grid.N_sigma, grid.N_tau
    if x0 is None:
        X = rng.normal(0.0, 0.1, size=(D, Ns, Nt)).astype(np.float64)
    else:
        X = np.array(x0, dtype=np.float64, copy=True)
        assert X.shape == (D, Ns, Nt)

    vol = grid.delta_sigma * grid.delta_tau
    saved = []
    save_count = 0

    for step in range(n_steps_total):
        lap = _laplacian_2d(X, grid)
        drift = (T * vol) * lap

        noise = rng.normal(0.0, 1.0, size=X.shape) * noise_scale
        X = X + eta * drift + np.sqrt(2.0 * eta) * noise

        if zero_mode_handling == "global_mean_subtract_each_step":
            X = X - X.mean(axis=(1, 2), keepdims=True)
        elif zero_mode_handling in ("none", None, ""):
            pass
        else:
            raise ValueError(f"Unknown zero_mode_handling: {zero_mode_handling}")

        if step >= burn_in_steps and ((step - burn_in_steps) % sample_every == 0):
            saved.append(X.copy())
            save_count += 1
            if save_count >= n_saved_samples:
                break

    if save_count < n_saved_samples:
        raise RuntimeError(
            f"Not enough samples saved. Wanted {n_saved_samples}, got {save_count}. "
            f"Increase n_steps_total or decrease n_saved_samples."
        )

    return np.stack(saved, axis=0)
