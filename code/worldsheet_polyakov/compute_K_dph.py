#!/usr/bin/env python3
"""
compute_K_dph.py (golden runner)
-------------------------------
End-to-end pipeline (with defensive engineering):

Polyakov lattice sampling -> window states -> fair Gram K -> distance D -> kNN graph -> all-pairs shortest paths -> DPH plot + summary.

Key properties:
- Streaming sampling (no samples array in RAM) via langevin_iter
- Streaming K accumulation across samples (no Ks list / stack)
- Band-limited K: skip pairs with delta_tau >= W_tau (no overlap -> K_ij = 0), exact + huge speedup
- Distance mapping uses val = clip(|K|, eps, 1.0) to ensure y>=0, d>=0 (Dijkstra safe)
- Macro mask "B": informative points via |K| > K_informative_min
- Atomic write for sweeps_summary.json (tmp + os.replace)
- Optional reuse_K_if_exists: skip sampling/K build and just recompute DPH/summary/plot from existing K files

Usage:
  python3 -u code/worldsheet_polyakov/compute_K_dph.py outputs/worldsheet_polyakov/run0003/params.json
"""

from __future__ import annotations

import os
import json
import math
import hashlib
from typing import Dict, Any, Tuple, List

import numpy as np

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heapq import heappush, heappop

from polyakov_lattice import GridSpec, langevin_iter
from build_states import window_starts, extract_window, weights, inner_product_fair
from sanity_checks import preflight_windows, post_k


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compute_K_fair_single(
    X: np.ndarray,
    W_tau: int,
    stride: int,
    delta_sigma: float,
    delta_tau: float,
    weights_type: str = "flat",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fair-normalized K for one configuration X of shape (D, Ns, Nt).
    Returns (K, starts).

    Band-limited: for fixed i, break when delta >= W_tau (no overlap => K_ij=0).
    """
    D, Ns, Nt = X.shape
    starts = window_starts(Nt, W_tau, stride)
    Nw = int(len(starts))
    if Nw <= 0:
        raise RuntimeError("No windows available (check W_tau/stride/N_tau).")

    w = weights(W_tau, Ns, kind=weights_type)
    patches = [extract_window(X, int(b0), W_tau) for b0 in starts]

    K = np.zeros((Nw, Nw), dtype=np.float64)

    for i in range(Nw):
        Xi = patches[i]
        bi = int(starts[i])
        for j in range(i, Nw):
            bj = int(starts[j])
            delta = bj - bi

            # No overlap => exactly zero; starts are increasing so we can break.
            if delta >= W_tau:
                break

            Xj = patches[j]
            ip, ni, nj = inner_product_fair(
                Xi, Xj, w,
                delta_sigma=delta_sigma,
                delta_tau=delta_tau,
                delta=delta,
            )
            kij = ip / (ni * nj) if (ni > 0.0 and nj > 0.0) else 0.0
            K[i, j] = kij
            K[j, i] = kij

    return K, starts


def knn_graph_from_dist(d: np.ndarray, k: int, symmetrize: bool = True) -> List[List[Tuple[int, float]]]:
    N = int(d.shape[0])
    k = int(min(max(k, 1), max(N - 1, 1)))
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(N)]

    for i in range(N):
        idx = np.argsort(d[i])
        neigh = [j for j in idx if j != i][:k]
        for j in neigh:
            w = float(d[i, j])
            if math.isfinite(w):
                adj[i].append((j, w))

    if not symmetrize:
        return adj

    adj2 = [dict() for _ in range(N)]
    for i in range(N):
        for j, w in adj[i]:
            adj2[i][j] = min(adj2[i].get(j, float("inf")), w)
            adj2[j][i] = min(adj2[j].get(i, float("inf")), w)
    return [[(j, w) for j, w in adj2[i].items()] for i in range(N)]


def dijkstra_all_pairs(adj: List[List[Tuple[int, float]]]) -> np.ndarray:
    N = len(adj)
    Dmat = np.full((N, N), np.inf, dtype=np.float64)

    for src in range(N):
        dist = np.full(N, np.inf, dtype=np.float64)
        dist[src] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src)]
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
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(x.size)
    if n < 3 or float(np.var(x)) == 0.0 or float(np.var(y)) == 0.0:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}

    X = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": n, "slope": slope, "intercept": intercept, "r2": r2}


def atomic_json_write(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main(params_json_path: str) -> None:
    with open(params_json_path, "r", encoding="utf-8") as f:
        P = json.load(f)

    run_dir = os.path.dirname(params_json_path)
    _ensure_dir(run_dir)

    # ---- params ----
    D = int(P["target_dim_D"])
    T = float(P["tension_T"])

    gridP = P["grid"]
    Ns = int(gridP["N_sigma"])
    Nt = int(gridP["N_tau"])
    ds = float(gridP["delta_sigma"])
    dt = float(gridP["delta_tau"])
    bc = gridP["bc"]
    grid = GridSpec(
        N_sigma=Ns,
        N_tau=Nt,
        delta_sigma=ds,
        delta_tau=dt,
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
    strict_sanity = bool(prag.get("strict_sanity", True))
    reuse_K = bool(prag.get("reuse_K_if_exists", False))
    progress_every = int(prag.get("progress_every", 5))
    max_samples_for_K = int(prag.get("max_samples_for_K", n_saved))
    K_min = float(prag.get("K_informative_min", 1e-4))

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

    plot_file = P["dph"]["outputs"]["plot_file"]
    summary_file = P["dph"]["outputs"]["summary_file"]

    # ---- preflight windows sanity ----
    sanity_windows: Dict[str, Any] = {}
    starts_by_W: Dict[int, np.ndarray] = {}
    for W_tau in W_sweep:
        W = int(W_tau)
        sanity_windows[str(W)] = preflight_windows(Nt=Nt, W_tau=W, stride=stride, strict=strict_sanity)
        starts_by_W[W] = window_starts(Nt, W, stride)

    # ---- load K if requested ----
    K_loaded: Dict[int, np.ndarray] = {}
    if reuse_K:
        for W_tau in W_sweep:
            W = int(W_tau)
            kpath = os.path.join(run_dir, f"K_W{W}.npy")
            if not os.path.exists(kpath):
                raise RuntimeError(f"[run] reuse_K_if_exists=true but missing: {kpath}")
            K_loaded[W] = np.load(kpath)
        print(f"[run] reuse_K_if_exists: loaded K for W={sorted(K_loaded.keys())}; skipping sampling/K build.", flush=True)

    # ---- sample + accumulate K (streaming) ----
    n_saved_effective = min(n_saved, max_samples_for_K)
    K_sum: Dict[int, np.ndarray] = {}
    starts_ref: Dict[int, np.ndarray] = {}

    if not reuse_K:
        print(f"[run] sampling Langevin: Ns={Ns} Nt={Nt} D={D} n_saved={n_saved_effective} (requested {n_saved}, cap {max_samples_for_K})", flush=True)

        for W_tau in W_sweep:
            W = int(W_tau)
            Nw = int(len(starts_by_W[W]))
            K_sum[W] = np.zeros((Nw, Nw), dtype=np.float64)
            starts_ref[W] = starts_by_W[W]

        m_count = 0
        for X in langevin_iter(
            grid=grid,
            D=D,
            T=T,
            eta=eta,
            n_steps_total=n_steps_total,
            burn_in_steps=burn_in,
            sample_every=sample_every,
            n_saved_samples=n_saved_effective,
            rng_seed=rng_seed,
            zero_mode_handling=zm,
            noise_scale=noise_scale,
        ):
            for W_tau in W_sweep:
                W = int(W_tau)
                K, starts = compute_K_fair_single(
                    X,
                    W_tau=W,
                    stride=stride,
                    delta_sigma=ds,
                    delta_tau=dt,
                    weights_type=weights_type,
                )
                if not np.array_equal(starts_ref[W], starts):
                    raise RuntimeError("[run] Window starts mismatch across samples (should not happen).")
                K_sum[W] += K

            m_count += 1
            if progress_every > 0 and (m_count % progress_every == 0 or m_count == n_saved_effective):
                print(f"[run] saved samples: {m_count}/{n_saved_effective}", flush=True)

        if m_count != n_saved_effective:
            raise RuntimeError(f"[run] internal: expected {n_saved_effective} samples, got {m_count}")

    # ---- sweep per W_tau ----
    sweep_results: Dict[str, Any] = {"run_id": P.get("run_id", "run"), "results": []}

    nW = len(W_sweep)
    fig, axes = plt.subplots(nW, 1, figsize=(8, 3.2 * nW))
    if nW == 1:
        axes = [axes]

    for wi, W_tau in enumerate(W_sweep):
        W = int(W_tau)
        print(f"[run] computing DPH for W_tau={W} ...", flush=True)

        if reuse_K:
            K_mean = K_loaded[W]
        else:
            # average
            K_mean = K_sum[W] / max(n_saved_effective, 1)

        K_abs_mean = np.abs(K_mean) if store_abs else K_mean.copy()

        # Sanity post-K + write sanity json
        sanity_k = post_k(K_mean, strict=strict_sanity)
        sanity_out = {"windows": sanity_windows[str(W)], "K": sanity_k, "reused_K": bool(reuse_K)}
        if not reuse_K:
            sanity_out["n_samples_used"] = int(n_saved_effective)
        atomic_json_write(os.path.join(run_dir, f"sanity_W{W}.json"), sanity_out)

        # Save K (optional: you can skip in reuse_K mode if you want)
        np.save(os.path.join(run_dir, f"K_W{W}.npy"), K_mean)
        if store_abs:
            np.save(os.path.join(run_dir, f"K_abs_W{W}.npy"), K_abs_mean)

        Nw = int(K_mean.shape[0])

        rep_xy = None

        for eps in eps_sweep:
            eps = float(eps)
            # Dijkstra-safe mapping: y>=0, d>=0
            absK = np.abs(K_mean)
            val = np.clip(absK, eps, 1.0)
            y_full = -np.log(val)
            d_full = ell0 * y_full
            np.fill_diagonal(d_full, 0.0)

            for k in k_sweep:
                k_eff = min(int(k), max(Nw - 1, 1))
                if k_eff < 1:
                    continue

                adj = knn_graph_from_dist(d_full, k=k_eff, symmetrize=symm)
                Dmat = dijkstra_all_pairs(adj)
                x_full = Dmat / max(ell0, 1e-12)

                iu, ju = np.triu_indices(Nw, k=1)
                x = x_full[iu, ju]
                y = y_full[iu, ju]

                # Macro mask B: |K| > K_min
                absK_pairs = absK[iu, ju]
                m_macro = np.isfinite(x) & np.isfinite(y) & (absK_pairs > K_min)

                fit_all = linear_fit_metrics(x, y)
                fit_macro = linear_fit_metrics(x[m_macro], y[m_macro])

                row = {
                    "W_tau": W,
                    "stride_s_tau": stride,
                    "epsilon": eps,
                    "knn_k": int(k_eff),
                    "N_windows": int(Nw),
                    "macro_def": f"|K|>{K_min:g}",
                    "n_macro": int(np.sum(m_macro)),
                    "macro_frac": float(np.sum(m_macro)) / float(len(m_macro)),
                    "fit_all": fit_all,
                    "fit_macro_q20_q80": fit_macro,
                }
                sweep_results["results"].append(row)

                # choose representative combo (single-combo case: first)
                if rep_xy is None:
                    rep_xy = (x, y, m_macro, row)

        # Plot representative for this W
        ax = axes[wi]
        if rep_xy is not None:
            x, y, m_macro, row = rep_xy
            ax.scatter(x, y, s=8, alpha=0.25)
            ax.scatter(x[m_macro], y[m_macro], s=10, alpha=0.9)
            r2 = (row["fit_macro_q20_q80"] or {}).get("r2")
            ax.set_title(f"W_tau={W} macro({row['macro_def']}) R^2={r2:.3f} n={row['n_macro']}")
        else:
            ax.set_title(f"W_tau={W} (no data)")
        ax.set_xlabel("x = D(i,j)/ell0")
        ax.set_ylabel("y = -log(clip(|K_ij|, eps, 1))")

    fig.tight_layout()

    # ensure dirs exist
    if os.path.dirname(plot_file):
        _ensure_dir(os.path.join(run_dir, os.path.dirname(plot_file)))
    if os.path.dirname(summary_file):
        _ensure_dir(os.path.join(run_dir, os.path.dirname(summary_file)))

    fig.savefig(os.path.join(run_dir, plot_file), dpi=160)
    plt.close(fig)

    # Atomic summary write + guard
    if not sweep_results.get("results"):
        raise RuntimeError("[run] no sweep results collected; check eps_sweep/k_sweep and indentation upstream.")
    atomic_json_write(os.path.join(run_dir, summary_file), sweep_results)

    # optional project log
    log_rel = P["io"].get("append_project_log", None)
    if log_rel:
        base_dir = os.path.abspath(os.path.join(run_dir, "..", "..", ".."))
        log_path = os.path.join(base_dir, log_rel)
        _ensure_dir(os.path.dirname(log_path))
        stamp = __import__("datetime").datetime.now().isoformat(timespec="seconds")
        sha1 = _sha1_of_file(params_json_path)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"- {stamp} | {P.get('run_id','run')} | params_sha1={sha1} | W_tau={W_sweep} | k={k_sweep} | eps={eps_sweep}\n")

    print(f"[done] outputs in: {run_dir}", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 compute_K_dph.py outputs/.../params.json")
    main(sys.argv[1])
