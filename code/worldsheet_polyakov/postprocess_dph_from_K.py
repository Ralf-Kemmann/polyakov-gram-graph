#!/usr/bin/env python3
from __future__ import annotations
import os, json, math
import numpy as np
import matplotlib.pyplot as plt
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
    if symmetrize:
        adj2 = [dict() for _ in range(N)]
        for i in range(N):
            for j, w in adj[i]:
                adj2[i][j] = min(adj2[i].get(j, float("inf")), w)
                adj2[j][i] = min(adj2[j].get(i, float("inf")), w)
        return [[(j, w) for j, w in adj2[i].items()] for i in range(N)]
    return adj

def dijkstra_all_pairs(adj):
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

def linear_fit_metrics(x: np.ndarray, y: np.ndarray):
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

def macro_window_slice(x: np.ndarray, q_lo: float = 0.2, q_hi: float = 0.8):
    m = np.isfinite(x)
    xf = x[m]
    if xf.size < 10:
        return m
    lo = float(np.quantile(xf, q_lo))
    hi = float(np.quantile(xf, q_hi))
    return m & (x >= lo) & (x <= hi)

def main(params_json: str):
    with open(params_json, "r", encoding="utf-8") as f:
        P = json.load(f)
    run_dir = os.path.dirname(params_json)

    W = int(P["states"]["W_tau_sweep"][0])
    K_path = os.path.join(run_dir, f"K_W{W}.npy")
    if not os.path.exists(K_path):
        raise SystemExit(f"Missing K file: {K_path}")

    K = np.load(K_path)
    Nw = K.shape[0]

    eps_sweep = list(P["kernel_K"]["epsilon_sweep"])
    k_sweep = list(P["distance_graph"]["knn_k_sweep"])
    ell0 = float(P["distance_graph"]["ell0"])
    symm = bool(P["distance_graph"].get("symmetrize", True))

    plot_file = P["dph"]["outputs"]["plot_file"]
    summary_file = P["dph"]["outputs"]["summary_file"]

    results = {"run_id": P.get("run_id","run"), "postprocess_only": True, "results": []}

    rep = None
    for eps in eps_sweep:
        eps = float(eps)
        y_full = -np.log(np.abs(K) + eps)
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
            m_macro = macro_window_slice(x, 0.2, 0.8)

            fit_all = linear_fit_metrics(x, y)
            fit_macro = linear_fit_metrics(x[m_macro], y[m_macro])

            results["results"].append({
                "W_tau": W, "epsilon": eps, "knn_k": int(k_eff), "N_windows": int(Nw),
                "fit_all": fit_all, "fit_macro_q20_q80": fit_macro
            })

            if rep is None:
                rep = (x, y, eps, k_eff, m_macro)

    # Plot representative
    plt.figure(figsize=(8,3.6))
    if rep is not None:
        x, y, eps_rep, k_rep, m_macro = rep
        plt.scatter(x, y, s=8, alpha=0.5)
        mm = np.isfinite(x) & np.isfinite(y) & m_macro
        if int(np.sum(mm)) >= 3:
            met = linear_fit_metrics(x[mm], y[mm])
            xs = np.linspace(float(np.nanmin(x[mm])), float(np.nanmax(x[mm])), 50)
            ys = met["slope"] * xs + met["intercept"]
            plt.plot(xs, ys, linewidth=2)
            plt.title(f"W_tau={W} (rep: eps={eps_rep:g}, k={k_rep})  macro R^2={met['r2']:.3f}")
        else:
            plt.title(f"W_tau={W} (rep: eps={eps_rep:g}, k={k_rep})  macro R^2=nan")
    plt.xlabel("x = D(i,j)/ell0")
    plt.ylabel("y = -log(|K_ij|+eps)")
    plt.tight_layout()

    os.makedirs(os.path.join(run_dir, os.path.dirname(plot_file)), exist_ok=True) if os.path.dirname(plot_file) else None
    os.makedirs(os.path.join(run_dir, os.path.dirname(summary_file)), exist_ok=True) if os.path.dirname(summary_file) else None

    plt.savefig(os.path.join(run_dir, plot_file), dpi=160)
    plt.close()

    with open(os.path.join(run_dir, summary_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[ok] wrote", os.path.join(run_dir, plot_file))
    print("[ok] wrote", os.path.join(run_dir, summary_file))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: postprocess_dph_from_K.py outputs/.../params.json")
    main(sys.argv[1])
