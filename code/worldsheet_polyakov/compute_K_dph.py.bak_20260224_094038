\
"""
compute_K_dph.py
----------------
End-to-end baseline pipeline:
Polyakov lattice sampling -> build τ-window states -> compute fair K -> distances -> kNN graph -> shortest paths -> DPH plot + sweep summary.

Usage:
  python compute_K_dph.py outputs/worldsheet_polyakov/run0001/params.json

Outputs (in run dir):
  K.npy, K_abs.npy
  dph_plot.png
  sweeps_summary.json
  (optional) samples.npy if enabled

This is a "first real engine run" skeleton: readable, reproducible, conservative.
"""

from __future__ import annotations

import os
import json
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

from polyakov_lattice import GridSpec, langevin_sample
from build_states import StateSpec, window_starts, extract_window, weights, inner_product_fair


def _sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def compute_K_fair(
    X: np.ndarray,
    W_tau: int,
    stride: int,
    delta_sigma: float,
    delta_tau: float,
    weights_type: str = "flat",
    store_abs: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute fair-normalized K for a single configuration X (D x Ns x Nt).
    Returns:
      K (Nw x Nw), K_abs (Nw x Nw), starts (Nw,)
    """
    D, Ns, Nt = X.shape
    starts = window_starts(Nt, W_tau, stride)
    Nw = len(starts)
    if Nw == 0:
        raise ValueError("No windows available. Check W_tau, stride, N_tau.")

    w = weights(W_tau, Ns, kind=weights_type)

    # cache window patches
    patches = [extract_window(X, int(b0), W_tau) for b0 in starts]

    K = np.zeros((Nw, Nw), dtype=np.float64)

    # compute fair overlap + normalization per pair
    for i in range(Nw):
        Xi = patches[i]
        bi = int(starts[i])
        for j in range(i, Nw):
            Xj = patches[j]
            bj = int(starts[j])
            delta = bj - bi
            ip, ni, nj = inner_product_fair(
                Xi, Xj, w, delta_sigma=delta_sigma, delta_tau=delta_tau, delta=delta
            )
            if ni > 0.0 and nj > 0.0:
                kij = ip / (ni * nj)
            else:
                kij = 0.0
            K[i, j] = kij
            K[j, i] = kij

    if store_abs:
        K_abs = np.abs(K)
    else:
        K_abs = K.copy()

    return K, K_abs, starts


def aggregate_K_over_ensemble(
    samples: np.ndarray,
    W_tau: int,
    stride: int,
    delta_sigma: float,
    delta_tau: float,
    weights_type: str,
    store_abs: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute K for each sample and average.
    samples shape: (M, D, Ns, Nt)
    """
    Ks = []
    starts_ref = None
    for m in range(samples.shape[0]):
        K, K_abs, starts = compute_K_fair(
            samples[m], W_tau=W_tau, stride=stride,
            delta_sigma=delta_sigma, delta_tau=delta_tau,
            weights_type=weights_type, store_abs=store_abs
        )
        if starts_ref is None:
            starts_ref = starts
        else:
            if not np.array_equal(starts_ref, starts):
                raise RuntimeError("Window starts mismatch across samples (should not happen).")
        Ks.append(K)
    K_mean = np.mean(np.stack(Ks, axis=0), axis=0)
    if store_abs:
        K_abs_mean = np.abs(K_mean)
    else:
        K_abs_mean = K_mean.copy()
    return K_mean, K_abs_mean, starts_ref


def knn_graph_from_dist(d: np.ndarray, k: int, symmetrize: bool = True) -> List[List[Tuple[int, float]]]:
    """
    Build a kNN graph adjacency list from full distance matrix d (N x N).
    Edge weights are d_ij.
    """
    N = d.shape[0]
    adj = [[] for _ in range(N)]
    for i in range(N):
        # exclude self, pick k smallest
        idx = np.argsort(d[i])
        neighbors = [j for j in idx if j != i][:k]
        for j in neighbors:
            w = float(d[i, j])
            if math.isfinite(w):
                adj[i].append((j, w))

    if symmetrize:
        # keep min weight if multiple
        adj2 = [dict() for _ in range(N)]
        for i in range(N):
            for j, w in adj[i]:
                prev = adj2[i].get(j, float("inf"))
                if w < prev:
                    adj2[i][j] = w
                prev2 = adj2[j].get(i, float("inf"))
                if w < prev2:
                    adj2[j][i] = w
        return [[(j, w) for j, w in adj2[i].items()] for i in range(N)]
    return adj


def dijkstra_all_pairs(adj: List[List[Tuple[int, float]]]) -> np.ndarray:
    """
    Compute all-pairs shortest paths using Dijkstra from each source.
    For N~O(10^2) windows this is fine.
    Returns D matrix (N x N) with inf for disconnected pairs.
    """
    N = len(adj)
    Dmat = np.full((N, N), np.inf, dtype=np.float64)

    for src in range(N):
        dist = np.full(N, np.inf, dtype=np.float64)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            dcur, u = heappop(pq)
            if dcur != dist[u]:
                continue
            for v, w in adj[u]:
                nd = dcur + w
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(pq, (nd, v))
        Dmat[src] = dist
    return Dmat


def linear_fit_metrics(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Simple linear regression y ~ a x + b on finite points.
    Reports slope, intercept, R^2, n.
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(x.size)
    if n < 3:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    X = np.vstack([x, np.ones_like(x)]).T
    # least squares
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": n, "slope": slope, "intercept": intercept, "r2": r2}


def macro_window_slice(x: np.ndarray, q_lo: float = 0.2, q_hi: float = 0.8) -> np.ndarray:
    """Mask selecting a 'macro' window by x quantiles (simple, robust)."""
    m = np.isfinite(x)
    xf = x[m]
    if xf.size < 10:
        return m
    lo = float(np.quantile(xf, q_lo))
    hi = float(np.quantile(xf, q_hi))
    return m & (x >= lo) & (x <= hi)


def main(params_json_path: str) -> None:
    with open(params_json_path, "r", encoding="utf-8") as f:
        P = json.load(f)

    run_dir = os.path.dirname(params_json_path)
    _ensure_dir(run_dir)

    # --- unpack params ---
    D = int(P["target_dim_D"])
    T = float(P["tension_T"])
    gridP = P["grid"]
    Ns = int(gridP["N_sigma"])
    Nt = int(gridP["N_tau"])
    ds = float(gridP["delta_sigma"])
    dt = float(gridP["delta_tau"])
    bc = gridP["bc"]
    grid = GridSpec(
        N_sigma=Ns, N_tau=Nt,
        delta_sigma=ds, delta_tau=dt,
        sigma_periodic=bool(bc["sigma_periodic"]),
        tau_periodic=bool(bc["tau_periodic"]),
    )

    samp = P["sampling"]
    eta = float(samp["step_size_eta"])
    n_steps_total = int(samp["n_steps_total"])
    burn_in = int(samp["burn_in_steps"])
    sample_every = int(samp["sample_every"])
    n_saved = int(samp["n_saved_samples"])
    rng_seed = int(samp["rng_seed"])
    zm = samp.get("zero_mode_handling", "global_mean_subtract_each_step")
    noise_scale = float(samp.get("noise", {}).get("scale", 1.0))

    prag = P.get("pragmatics", {})
    max_samples_for_K = int(prag.get("max_samples_for_K", n_saved))
    save_samples = bool(prag.get("save_samples_npy", False))

    statesP = P["states"]
    stride = int(statesP["stride_s_tau"])
    W_sweep = list(statesP["W_tau_sweep"])
    weights_type = statesP.get("weights", {}).get("type", "flat")

    Kp = P["kernel_K"]
    store_abs = bool(Kp.get("store_abs", True))
    eps_sweep = list(Kp["epsilon_sweep"])

    graphP = P["distance_graph"]
    ell0 = float(graphP["ell0"])
    k_sweep = list(graphP["knn_k_sweep"])
    symm = bool(graphP.get("symmetrize", True))

    # --- sampling ---
    print(f"[run] sampling Langevin: Ns={Ns} Nt={Nt} D={D} n_saved={n_saved} (use {min(n_saved, max_samples_for_K)} for K)")
    samples = langevin_sample(
        grid=grid, D=D, T=T, eta=eta,
        n_steps_total=n_steps_total,
        burn_in_steps=burn_in,
        sample_every=sample_every,
        n_saved_samples=n_saved,
        rng_seed=rng_seed,
        zero_mode_handling=zm,
        noise_scale=noise_scale,
    )

    if save_samples:
        np.save(os.path.join(run_dir, "samples.npy"), samples)

    # optionally limit for K aggregation (pragmatic)
    M_use = min(samples.shape[0], max_samples_for_K)
    samples_use = samples[:M_use]

    sweep_results: Dict[str, Any] = {"run_id": P.get("run_id", "run"), "results": []}

    # For plotting: we’ll put one panel per W_tau (stacked) but in ONE figure to keep it simple.
    nW = len(W_sweep)
    fig, axes = plt.subplots(nW, 1, figsize=(8, 3.2 * nW))
    if nW == 1:
        axes = [axes]

    for wi, W_tau in enumerate(W_sweep):
        print(f"[run] computing K for W_tau={W_tau} ...")
        K_mean, K_abs_mean, starts = aggregate_K_over_ensemble(
            samples_use, W_tau=W_tau, stride=stride,
            delta_sigma=ds, delta_tau=dt,
            weights_type=weights_type, store_abs=store_abs
        )

        # save one K per W_tau (so we don't overwrite)
        np.save(os.path.join(run_dir, f"K_W{W_tau}.npy"), K_mean)
        if store_abs:
            np.save(os.path.join(run_dir, f"K_abs_W{W_tau}.npy"), K_abs_mean)

        Nw = K_mean.shape[0]
        # We'll generate x,y for the "best" (middle) sweep as representative, and store metrics for all sweeps.
        rep_xy = None

        for eps in eps_sweep:
            eps = float(eps)
            y_full = -np.log(np.abs(K_mean) + eps)
            d_full = ell0 * y_full  # since d = -ell0 log(|K|+eps)
            np.fill_diagonal(d_full, 0.0)

            for k in k_sweep:
                k = int(k)
                adj = knn_graph_from_dist(d_full, k=k, symmetrize=symm)
                Dmat = dijkstra_all_pairs(adj)
                x_full = Dmat / max(ell0, 1e-12)

                # build pair vectors (i<j) for analysis/plot
                iu, ju = np.triu_indices(Nw, k=1)
                x = x_full[iu, ju]
                y = y_full[iu, ju]

                # macro window
                m_macro = macro_window_slice(x, 0.2, 0.8)
                metrics_all = linear_fit_metrics(x, y)
                metrics_macro = linear_fit_metrics(x[m_macro], y[m_macro])

                sweep_results["results"].append({
                    "W_tau": W_tau,
                    "stride_s_tau": stride,
                    "epsilon": eps,
                    "knn_k": k,
                    "N_windows": int(Nw),
                    "fit_all": metrics_all,
                    "fit_macro_q20_q80": metrics_macro
                })

                # representative plot choice: middle eps and middle k
                if rep_xy is None:
                    # choose first as placeholder
                    rep_xy = (x, y, eps, k, m_macro)
                else:
                    # prefer eps=middle, k=middle
                    eps_mid = eps_sweep[len(eps_sweep)//2]
                    k_mid = k_sweep[len(k_sweep)//2]
                    if math.isclose(eps, float(eps_mid)) and k == int(k_mid):
                        rep_xy = (x, y, eps, k, m_macro)

        # plot representative scatter
        ax = axes[wi]
        if rep_xy is not None:
            x, y, eps_rep, k_rep, m_macro = rep_xy
            ax.scatter(x, y, s=6, alpha=0.5)
            # add macro fit line
            mm = np.isfinite(x) & np.isfinite(y) & m_macro
            if np.sum(mm) >= 3:
                met = linear_fit_metrics(x[mm], y[mm])
                xs = np.linspace(float(np.nanmin(x[mm])), float(np.nanmax(x[mm])), 50)
                ys = met["slope"] * xs + met["intercept"]
                ax.plot(xs, ys, linewidth=2)
                ax.set_title(f"W_tau={W_tau} (rep: eps={eps_rep:g}, k={k_rep})  macro R^2={met['r2']:.3f}")
            else:
                ax.set_title(f"W_tau={W_tau} (rep: eps={eps_rep:g}, k={k_rep})")
        else:
            ax.set_title(f"W_tau={W_tau} (no data)")
        ax.set_xlabel("x = D(i,j)/ell0")
        ax.set_ylabel("y = -log(|K_ij|+eps)")

    fig.tight_layout()
    plot_file = P["dph"]["outputs"]["plot_file"]
    fig.savefig(os.path.join(run_dir, plot_file), dpi=160)
    plt.close(fig)

    # save summary
    summary_file = P["dph"]["outputs"]["summary_file"]
    with open(os.path.join(run_dir, summary_file), "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    # append project log
    log_rel = P["io"].get("append_project_log", None)
    if log_rel:
        # interpret relative to base folder above run_dir
        base_dir = os.path.abspath(os.path.join(run_dir, "..", "..", ".."))
        log_path = os.path.join(base_dir, log_rel)
        _ensure_dir(os.path.dirname(log_path))
        stamp = __import__("datetime").datetime.now().isoformat(timespec="seconds")
        sha1 = _sha1_of_file(params_json_path)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"- {stamp} | {P.get('run_id','run')} | params_sha1={sha1} | W_tau={P['states']['W_tau_sweep']} | k={P['distance_graph']['knn_k_sweep']} | eps={P['kernel_K']['epsilon_sweep']}\n")

    print(f"[done] outputs in: {run_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python compute_K_dph.py outputs/worldsheet_polyakov/run0001/params.json")
    main(sys.argv[1])
