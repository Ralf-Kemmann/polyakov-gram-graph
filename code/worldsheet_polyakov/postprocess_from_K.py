#!/usr/bin/env python3
from __future__ import annotations
import os, json, math
import numpy as np
from heapq import heappush, heappop

def knn_graph_from_dist(d: np.ndarray, k: int, symmetrize: bool = True):
    N = d.shape[0]
    k = int(min(max(k, 1), max(N - 1, 1)))
    adj = [[] for _ in range(N)]
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

def dijkstra_all_pairs(adj):
    N = len(adj)
    Dmat = np.full((N, N), np.inf, dtype=float)
    for s in range(N):
        dist = np.full(N, np.inf, dtype=float)
        dist[s] = 0.0
        pq = [(0.0, s)]
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

def linear_fit_metrics(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
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

def main(params_path: str):
    with open(params_path, "r", encoding="utf-8") as f:
        P = json.load(f)
    run_dir = os.path.dirname(params_path)

    W = int(P["states"]["W_tau_sweep"][0])
    K_path = os.path.join(run_dir, f"K_W{W}.npy")
    if not os.path.exists(K_path):
        raise SystemExit(f"Missing K file: {K_path}")
    K = np.load(K_path)
    Nw = int(K.shape[0])

    eps_sweep = list(P["kernel_K"]["epsilon_sweep"])
    k_sweep = list(P["distance_graph"]["knn_k_sweep"])
    ell0 = float(P["distance_graph"]["ell0"])
    symm = bool(P["distance_graph"].get("symmetrize", True))

    plot_file = P["dph"]["outputs"]["plot_file"]
    summary_file = P["dph"]["outputs"]["summary_file"]

    K_min = float(P.get("pragmatics", {}).get("K_informative_min", 1e-4))

    absK = np.abs(K)

    results = {"run_id": P.get("run_id", "run"), "results": []}
    rep = None

    for eps in eps_sweep:
        eps = float(eps)
        val = np.clip(absK, eps, 1.0)   # eps as floor only; ensures y>=0
        y_full = -np.log(val)
        d_full = ell0 * y_full
        np.fill_diagonal(d_full, 0.0)

        for k in k_sweep:
            k_eff = min(int(k), max(Nw - 1, 1))
            if k_eff < 1:
                continue
            adj = knn_graph_from_dist(d_full, k_eff, symmetrize=symm)
            Dmat = dijkstra_all_pairs(adj)
            x_full = Dmat / max(ell0, 1e-12)

            iu, ju = np.triu_indices(Nw, k=1)
            x = x_full[iu, ju]
            y = y_full[iu, ju]
            absK_pairs = absK[iu, ju]

            m_macro = np.isfinite(x) & np.isfinite(y) & (absK_pairs > K_min)

            fit_all = linear_fit_metrics(x, y)
            fit_macro = linear_fit_metrics(x[m_macro], y[m_macro])

            row = {
                "W_tau": W,
                "epsilon": eps,
                "knn_k": int(k_eff),
                "N_windows": int(Nw),
                "macro_def": f"|K|>{K_min:g}",
                "n_macro": int(np.sum(m_macro)),
                "macro_frac": float(np.sum(m_macro)) / float(len(m_macro)),
                "fit_all": fit_all,
                "fit_macro": fit_macro,
            }
            results["results"].append(row)

            # representative plot: pick first combo
            if rep is None:
                rep = (x, y, m_macro, row)

    # ensure dirs for outputs
    if os.path.dirname(plot_file):
        os.makedirs(os.path.join(run_dir, os.path.dirname(plot_file)), exist_ok=True)
    if os.path.dirname(summary_file):
        os.makedirs(os.path.join(run_dir, os.path.dirname(summary_file)), exist_ok=True)

    # atomic summary write
    out_sum = os.path.join(run_dir, summary_file)
    tmp = out_sum + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, out_sum)

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_plot = os.path.join(run_dir, plot_file)
    plt.figure(figsize=(8, 3.6))
    if rep is not None:
        x, y, m_macro, row = rep
        plt.scatter(x, y, s=8, alpha=0.25)
        plt.scatter(x[m_macro], y[m_macro], s=10, alpha=0.9)
        r2 = (row["fit_macro"] or {}).get("r2")
        plt.title(f"W_tau={row['W_tau']} eps={row['epsilon']:g} k={row['knn_k']}  macro({row['macro_def']}) R^2={r2:.3f}  n={row['n_macro']}")
    plt.xlabel("x = D(i,j)/ell0")
    plt.ylabel("y = -log(clip(|K_ij|, eps, 1))")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=160)
    plt.close()

    print("[ok] wrote", out_sum)
    print("[ok] wrote", out_plot)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: postprocess_from_K.py outputs/.../params.json")
    main(sys.argv[1])
