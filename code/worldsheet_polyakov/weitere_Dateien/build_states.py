\
"""
build_states.py
---------------
Builds the τ-window / patch-based state family { |psi_i> } from worldsheet fields X^μ(σ, τ).

Core idea:
- choose window length W_tau, stride s_tau
- i corresponds to window start b_i
- |psi_i> = vec( w_{a,b} * X_{a, b_i+b}^μ )  for b=0..W_tau-1
- |Phi_i> = ΔσΔτ |psi_i> (measure consistency)

Fairness via S(δ):
- For two windows i, j with δ = b_j - b_i, the common support is the overlap slice.
- Inner products and norms are computed on that overlap, then normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np


@dataclass
class StateSpec:
    W_tau: int
    stride_s_tau: int
    delta_sigma: float = 1.0
    delta_tau: float = 1.0
    weights_type: str = "flat"  # placeholder for future weighting (tapering etc.)


def window_starts(N_tau: int, W_tau: int, stride: int) -> np.ndarray:
    """All valid non-wrapping window starts b_i such that [b_i, b_i+W_tau) within [0, N_tau)."""
    if W_tau > N_tau:
        return np.array([], dtype=int)
    return np.arange(0, N_tau - W_tau + 1, stride, dtype=int)


def extract_window(X: np.ndarray, b0: int, W_tau: int) -> np.ndarray:
    """
    Extract window patch from X (shape D x N_sigma x N_tau).
    Returns patch (D x N_sigma x W_tau).
    """
    return X[:, :, b0:b0 + W_tau]


def weights(W_tau: int, N_sigma: int, kind: str = "flat") -> np.ndarray:
    """Weights w_{a,b}. For now: flat weights = 1."""
    if kind == "flat":
        return np.ones((N_sigma, W_tau), dtype=np.float64)
    raise ValueError(f"Unknown weights_type: {kind}")


def fairness_overlap_slices(W_tau: int, delta: int) -> Tuple[slice, slice]:
    """
    Given δ = b_j - b_i, return (slice_i, slice_j) into the window axis (0..W_tau-1)
    that represent the common support S(δ).
    """
    if delta >= W_tau or delta <= -W_tau:
        return slice(0, 0), slice(0, 0)  # empty overlap

    if delta >= 0:
        # i uses [delta:W), j uses [0:W-delta)
        return slice(delta, W_tau), slice(0, W_tau - delta)
    else:
        # delta < 0
        # i uses [0:W+delta), j uses [-delta:W)
        return slice(0, W_tau + delta), slice(-delta, W_tau)


def inner_product_fair(
    Xi: np.ndarray,
    Xj: np.ndarray,
    w: np.ndarray,
    delta_sigma: float,
    delta_tau: float,
    delta: int,
) -> Tuple[float, float, float]:
    """
    Fair inner product <Phi_i|Phi_j> restricted to S(δ), and corresponding self-norms:
      returns (ip, ni, nj) with:
        ip = <Phi_i|Phi_j>_S
        ni = sqrt(<Phi_i|Phi_i>_S)
        nj = sqrt(<Phi_j|Phi_j>_S)

    Xi, Xj are window patches with shape (D, N_sigma, W_tau).
    w has shape (N_sigma, W_tau).
    """
    assert Xi.shape == Xj.shape
    D, Ns, W = Xi.shape
    assert w.shape == (Ns, W)

    si, sj = fairness_overlap_slices(W, delta)
    if (si.stop - si.start) <= 0:
        return 0.0, 0.0, 0.0

    # select overlap
    Xi_o = Xi[:, :, si]  # D x Ns x L
    Xj_o = Xj[:, :, sj]  # D x Ns x L
    w_o = w[:, si]       # Ns x L  (same length L)

    # apply weights; broadcast across D
    Xi_w = Xi_o * w_o[None, :, :]
    Xj_w = Xj_o * w_o[None, :, :]

    # measure factor (ΔσΔτ)
    m = delta_sigma * delta_tau

    ip = m * float(np.sum(Xi_w * Xj_w))
    nii = m * float(np.sum(Xi_w * Xi_w))
    njj = m * float(np.sum(Xj_w * Xj_w))

    ni = float(np.sqrt(max(nii, 0.0)))
    nj = float(np.sqrt(max(njj, 0.0)))
    return ip, ni, nj
