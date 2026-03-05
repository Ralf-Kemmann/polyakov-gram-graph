"""
polyakov_lattice.py
-------------------
Lattice Polyakov (Euclidean, conformal gauge) baseline sampler.

NEW (streaming):
- langevin_iter(...) yields samples one-by-one, so pipelines can avoid a huge samples array in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterator
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
    D, Ns, Nt = X.shape
    ds2 = grid.delta_sigma ** 2
    dt2 = grid.delta_tau ** 2

    if grid.sigma_periodic:
        X_sp = np.roll(X, shift=-1, axis=1)
        X_sm = np.roll(X, shift=+1, axis=1)
    else:
        X_sp = X.copy()
        X_sm = X.copy()
        X_sp[:, :-1, :] = X[:, 1:, :]
        X_sp[:, -1, :] = X[:, -1, :]
        X_sm[:, 1:, :] = X[:, :-1, :]
        X_sm[:, 0, :] = X[:, 0, :]

    if grid.tau_periodic:
        X_tp = np.roll(X, shift=-1, axis=2)
        X_tm = np.roll(X, shift=+1, axis=2)
    else:
        X_tp = X.copy()
        X_tm = X.copy()
        X_tp[:, :, :-1] = X[:, :, 1:]
        X_tp[:, :, -1] = X[:, :, -1]
        X_tm[:, :, 1:] = X[:, :, :-1]
        X_tm[:, :, 0] = X[:, :, 0]

    return (X_sp - 2.0 * X + X_sm) / ds2 + (X_tp - 2.0 * X + X_tm) / dt2


def langevin_iter(
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
) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    Ns, Nt = grid.N_sigma, grid.N_tau
    if x0 is None:
        X = rng.normal(0.0, 0.1, size=(D, Ns, Nt)).astype(np.float64)
    else:
        X = np.array(x0, dtype=np.float64, copy=True)
        assert X.shape == (D, Ns, Nt)

    vol = grid.delta_sigma * grid.delta_tau
    saved = 0

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
            yield X.copy()
            saved += 1
            if saved >= n_saved_samples:
                break

    if saved < n_saved_samples:
        raise RuntimeError(
            f"Not enough samples saved. Wanted {n_saved_samples}, got {saved}. "
            f"Increase n_steps_total or decrease n_saved_samples."
        )


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
    saved = list(
        langevin_iter(
            grid=grid, D=D, T=T, eta=eta,
            n_steps_total=n_steps_total,
            burn_in_steps=burn_in_steps,
            sample_every=sample_every,
            n_saved_samples=n_saved_samples,
            rng_seed=rng_seed,
            zero_mode_handling=zero_mode_handling,
            noise_scale=noise_scale,
            x0=x0,
        )
    )
    return np.stack(saved, axis=0)
