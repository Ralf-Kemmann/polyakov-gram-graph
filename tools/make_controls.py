#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, os, shutil, subprocess
from pathlib import Path
from heapq import heappush, heappop
from typing import Any, Dict, List, Tuple

import numpy as np


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, d: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


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
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 3 or float(np.var(x)) == 0.0 or float(np.var(y)) == 0.0:
        return float("nan")
    X = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def run_compute(params_json: Path) -> Tuple[int, str]:
    cmd = ["python3", "-u", "code/worldsheet_polyakov/compute_K_dph.py", str(params_json)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def make_control_abort(base_params: Path, out_dir: Path, W: int, stride: int) -> None:
    d = load_json(base_params)
    d["run_id"] = f"control_stride_ge_W_W{W}_S{stride}"
    d.setdefault("io", {})
    d["io"]["output_dir"] = str(out_dir)
    d.setdefault("states", {})
    d["states"]["W_tau_sweep"] = [int(W)]
    d["states"]["stride_s_tau"] = int(stride)
    d.setdefault("pragmatics", {})
    d["pragmatics"]["strict_sanity"] = True
    d["pragmatics"]["reuse_K_if_exists"] = False

    params_out = out_dir / "params.json"
    save_json(params_out, d)

    rc, out = run_compute(params_out)

    # Expect non-zero (abort)
    result = {
        "control": "stride_ge_W_abort",
        "expected": "abort",
        "returncode": rc,
        "W_tau": W,
        "stride": stride,
        "ok": (rc != 0),
        "note": "Strict sanity preflight should abort when max_overlap=0 (degenerate windows).",
    }
    # Keep last ~60 lines for sanity
    lines = out.splitlines()
    result["log_tail"] = "\n".join(lines[-60:])

    save_json(out_dir / "control_result.json", result)


def make_control_shuffle_edges(base_params: Path, out_dir: Path, seed: int = 0) -> None:
    """
    Shuffle-control: keeps |K| distribution but destroys geometry by shuffling off-diagonal y entries
    before graph distances. Produces sweeps_summary.json + dph_plot.png.
    """
    P = load_json(base_params)
    run_dir = base_params.parent

    W = int(P.get("states", {}).get("W_tau_sweep", [24])[0])
    eps = float(P.get("kernel_K", {}).get("epsilon_sweep", [1e-8])[0])
    k = int(P.get("distance_graph", {}).get("knn_k_sweep", [7])[0])
    ell0 = float(P.get("distance_graph", {}).get("ell0", 1.0))
    symm = bool(P.get("distance_graph", {}).get("symmetrize", True))
    K_min = float(P.get("pragmatics", {}).get("K_informative_min", 1e-4))

    K_path = run_dir / f"K_W{W}.npy"
    if not K_path.exists():
        raise SystemExit(f"Missing baseline K file: {K_path} (run ./run.sh run0004 first)")

    out_dir.mkdir(parents=True, exist_ok=True)

    K = np.load(K_path)
    Nw = int(K.shape[0])
    absK = np.abs(K)

    y_full = -np.log(np.clip(absK, eps, 1.0))
    iu, ju = np.triu_indices(Nw, k=1)

    rng = np.random.default_rng(seed)
    vals = y_full[iu, ju].copy()
    rng.shuffle(vals)

    y_shuf = np.zeros_like(y_full)
    y_shuf[iu, ju] = vals
    y_shuf[ju, iu] = vals
    np.fill_diagonal(y_shuf, 0.0)

    d_full = ell0 * y_shuf

    k_eff = min(k, max(Nw - 1, 1))
    adj = knn_graph_from_dist(d_full, k_eff, symmetrize=symm)
    Dmat = dijkstra_all_pairs(adj)
    x_full = Dmat / max(ell0, 1e-12)

    x = x_full[iu, ju]
    y = y_shuf[iu, ju]
    absK_pairs = absK[iu, ju]

    mask = np.isfinite(x) & np.isfinite(y) & (absK_pairs > K_min)
    M = int(len(x))
    n_macro = int(np.sum(mask))
    macro_frac = float(n_macro) / float(M) if M > 0 else float("nan")
    r2_macro = float(linear_fit_r2(x[mask], y[mask]))
    r2_all = float(linear_fit_r2(x, y))

    summary = {
        "run_id": "control_shuffle_edges",
        "results": [{
            "W_tau": W,
            "stride_s_tau": int(P.get("states", {}).get("stride_s_tau", 8)),
            "epsilon": eps,
            "knn_k": k_eff,
            "N_windows": Nw,
            "macro_def": f"|K|>{K_min:g}",
            "n_macro": n_macro,
            "macro_frac": macro_frac,
            "fit_macro_q20_q80": {"r2": r2_macro},
            "fit_all": {"r2": r2_all},
            "control": "shuffle_offdiag_y",
            "seed": seed,
            "note": "Off-diagonal y_ij values shuffled symmetrically before graph distances; should destroy macro structure.",
        }]
    }

    (out_dir / "sweeps_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 3.6))
    plt.scatter(x, y, s=8, alpha=0.25)
    if n_macro > 0:
        plt.scatter(x[mask], y[mask], s=10, alpha=0.9)
    plt.title(f"Shuffle control (seed={seed})  macro_frac={macro_frac:.3f}  R2_macro={r2_macro:.3f}")
    plt.xlabel("x = D(i,j)/ell0 (from shuffled y)")
    plt.ylabel("y = shuffled -log(clip(|K|,eps,1))")
    plt.tight_layout()
    plt.savefig(out_dir / "dph_plot.png", dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/worldsheet_polyakov/run0004/params.json")
    ap.add_argument("--out-root", default="outputs/worldsheet_polyakov")
    ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--stride-abort", type=int, default=32)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    args = ap.parse_args()

    repo = Path.cwd()
    base = repo / args.base
    out_root = repo / args.out_root
    if not base.exists():
        raise SystemExit(f"Base params not found: {base}")

    # Control 1 (abort)
    out1 = out_root / "control_stride_ge_W"
    make_control_abort(base, out1, W=args.W, stride=args.stride_abort)

    # Control 2 (shuffle)
    out2 = out_root / "control_shuffle_edges"
    make_control_shuffle_edges(base, out2, seed=args.shuffle_seed)

    print("[ok] wrote controls in:")
    print(" -", out1)
    print(" -", out2)
    print("[info] For Control 1, see control_result.json (expected abort).")
    print("[info] For Control 2, see sweeps_summary.json and dph_plot.png.")


if __name__ == "__main__":
    main()