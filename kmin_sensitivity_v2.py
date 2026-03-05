#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def knn_graph_from_dist(d: np.ndarray, k: int, symmetrize: bool = True) -> List[List[Tuple[int, float]]]:
    """Build kNN adjacency list from a full distance matrix d (N x N)."""
    N = int(d.shape[0])
    k = int(min(max(k, 1), max(N - 1, 1)))

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    for i in range(N):
        idx = np.argsort(d[i])
        neighbors = [j for j in idx if j != i][:k]
        for j in neighbors:
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
    """All-pairs shortest paths on a non-negative weighted graph (Dijkstra from each source)."""
    N = len(adj)
    Dmat = np.full((N, N), np.inf, dtype=float)
    for s in range(N):
        dist = np.full(N, np.inf, dtype=float)
        dist[s] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, s)]
        while pq:
            dcur, u = heappop(pq)
            if dcur != dist[u]:
                continue
            for v, w in adj[u]:
                nd = dcur + w
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(pq, (nd, v))
        Dmat[s] = dist
    return Dmat


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Least squares fit y ~ a*x + b; returns R^2 (nan if ill-posed)."""
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return float("nan")
    if float(np.var(x)) == 0.0 or float(np.var(y)) == 0.0:
        return float("nan")

    X = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="K_min sensitivity sweep: macro_frac and R2_macro vs K_min.")
    ap.add_argument(
        "--params",
        default="outputs/worldsheet_polyakov/run0004/params.json",
        help="Path to params.json (default: run0004 baseline).",
    )
    ap.add_argument(
        "--W",
        type=int,
        default=None,
        help="W_tau to analyze (default: first element of states.W_tau_sweep).",
    )
    ap.add_argument("--kmin-min", type=float, default=1e-6, help="Minimum K_min (default: 1e-6).")
    ap.add_argument("--kmin-max", type=float, default=1e-3, help="Maximum K_min (default: 1e-3).")
    ap.add_argument("--n", type=int, default=13, help="Number of log-spaced K_min points (default: 13).")
    ap.add_argument(
        "--out",
        default="figures/kmin_sensitivity.png",
        help="Output PNG path (default: figures/kmin_sensitivity.png).",
    )
    args = ap.parse_args()

    params_path = Path(args.params)
    if not params_path.exists():
        raise SystemExit(f"params.json not found: {params_path}")

    P = load_json(params_path)
    run_dir = params_path.parent

    W_list = P.get("states", {}).get("W_tau_sweep", [24])
    W = int(args.W if args.W is not None else W_list[0])

    eps_sweep = P.get("kernel_K", {}).get("epsilon_sweep", [1e-8])
    eps = float(eps_sweep[0])

    k_sweep = P.get("distance_graph", {}).get("knn_k_sweep", [7])
    k = int(k_sweep[0])

    ell0 = float(P.get("distance_graph", {}).get("ell0", 1.0))
    symm = bool(P.get("distance_graph", {}).get("symmetrize", True))

    K_path = run_dir / f"K_W{W}.npy"
    if not K_path.exists():
        raise SystemExit(f"Missing K file: {K_path} (run ./run.sh runXXXX first)")

    K = np.load(K_path)
    Nw = int(K.shape[0])

    absK = np.abs(K)
    y_full = -np.log(np.clip(absK, eps, 1.0))  # guarantees y>=0
    d_full = ell0 * y_full
    np.fill_diagonal(d_full, 0.0)

    k_eff = min(k, max(Nw - 1, 1))
    adj = knn_graph_from_dist(d_full, k=k_eff, symmetrize=symm)
    Dmat = dijkstra_all_pairs(adj)
    x_full = Dmat / max(ell0, 1e-12)

    iu, ju = np.triu_indices(Nw, k=1)
    x = x_full[iu, ju]
    y = y_full[iu, ju]
    absK_pairs = absK[iu, ju]
    M = int(len(absK_pairs))

    kmins = np.logspace(math.log10(args.kmin_min), math.log10(args.kmin_max), int(args.n))
    macro_frac: List[float] = []
    r2_macro: List[float] = []
    n_macro: List[int] = []

    for K_min in kmins:
        mask = np.isfinite(x) & np.isfinite(y) & (absK_pairs > K_min)
        nm = int(np.sum(mask))
        n_macro.append(nm)
        macro_frac.append(nm / M if M > 0 else float("nan"))
        r2_macro.append(linear_fit_r2(x[mask], y[mask]))

    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_png.with_suffix(".json")
    out_json.write_text(
        json.dumps(
            {
                "params": str(params_path),
                "run_dir": str(run_dir),
                "W_tau": W,
                "eps": eps,
                "k": k_eff,
                "ell0": ell0,
                "Kmin_min": args.kmin_min,
                "Kmin_max": args.kmin_max,
                "n_points": int(args.n),
                "kmins": [float(v) for v in kmins],
                "n_macro": n_macro,
                "macro_frac": [float(v) for v in macro_frac],
                "r2_macro": [float(v) for v in r2_macro],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)

    ax1.plot(kmins, macro_frac, marker="o", linewidth=1.5, label="macro_frac")
    ax1.set_xscale("log")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel(r"$K_{\min}$")
    ax1.set_ylabel("macro_frac")

    ax2 = ax1.twinx()
    ax2.plot(kmins, r2_macro, marker="s", linewidth=1.5, label=r"$R^2_{\mathrm{macro}}$")
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel(r"$R^2_{\mathrm{macro}}$")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Place legend outside the plotting area to avoid overlap with curves
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    ax1.set_title(f"K_min sensitivity (W_tau={W}, k={k_eff}, eps={eps:g}, Nw={Nw})")
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)

    print("[ok] wrote", out_png)
    print("[ok] wrote", out_json)
    print("[info] macro counts range:", min(n_macro), "to", max(n_macro))


if __name__ == "__main__":
    main()
