#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List


W_LIST = [16, 24, 32]
S_LIST = [4, 8, 12]


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, d: Dict[str, Any]) -> None:
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def sweep_dir(root: Path, W: int, stride: int) -> Path:
    return root / f"sweep_W{W}_S{stride}"


def ensure_params(base_params: Path, out_root: Path, W: int, stride: int) -> Path:
    d = load_json(base_params)

    run_id = f"sweep_W{W}_S{stride}"
    out_dir = sweep_dir(out_root, W, stride)
    out_dir.mkdir(parents=True, exist_ok=True)

    d["run_id"] = run_id

    # IO output directory (if present)
    d.setdefault("io", {})
    d["io"]["output_dir"] = str(out_dir)

    # states
    d.setdefault("states", {})
    d["states"]["W_tau_sweep"] = [int(W)]
    d["states"]["stride_s_tau"] = int(stride)

    # keep baseline eps/k unless user changed them
    d.setdefault("kernel_K", {})
    d.setdefault("distance_graph", {})
    d.setdefault("dph", {})
    d.setdefault("pragmatics", {})

    # Safety / reproducibility
    d["pragmatics"]["strict_sanity"] = True

    # IMPORTANT: sweep runs must compute their own K (do not reuse unless rerunning)
    d["pragmatics"]["reuse_K_if_exists"] = False

    # Macro criterion B (keep your default unless already set)
    d["pragmatics"].setdefault("K_informative_min", 1e-4)

    # DPH outputs (ensure deterministic filenames)
    d.setdefault("dph", {})
    d["dph"].setdefault("outputs", {})
    d["dph"]["outputs"]["plot_file"] = "dph_plot.png"
    d["dph"]["outputs"]["summary_file"] = "sweeps_summary.json"

    p = out_dir / "params.json"
    save_json(p, d)
    return p


def run_one(params_json: Path, *, use_run_sh: bool = True) -> int:
    repo_root = Path.cwd()
    if use_run_sh and (repo_root / "run.sh").exists():
        cmd = ["bash", "./run.sh", str(params_json)]
    else:
        cmd = ["python3", "-u", "code/worldsheet_polyakov/compute_K_dph.py", str(params_json)]
    print("[cmd]", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def pick_row(results: List[Dict[str, Any]], W: int, stride: int) -> Optional[Dict[str, Any]]:
    # Prefer exact match on W/stride
    for r in results:
        if int(r.get("W_tau", -1)) == int(W) and int(r.get("stride_s_tau", -1)) == int(stride):
            return r
    return results[0] if results else None


def read_metrics(summary_json: Path, W: int, stride: int) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    d = load_json(summary_json)
    results = d.get("results", [])
    row = pick_row(results, W, stride)
    if not row:
        return None, None, None, None

    Nw = row.get("N_windows", None)
    macro_frac = row.get("macro_frac", None)

    fm = row.get("fit_macro_q20_q80") or row.get("fit_macro") or {}
    fa = row.get("fit_all") or {}

    r2_macro = fm.get("r2", None)
    r2_all = fa.get("r2", None)

    return (int(Nw) if Nw is not None else None,
            float(macro_frac) if macro_frac is not None else None,
            float(r2_macro) if r2_macro is not None else None,
            float(r2_all) if r2_all is not None else None)


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_prep = sub.add_parser("prepare", help="Create 3x3 sweep param files from a base params.json")
    ap_prep.add_argument("--base", default="outputs/worldsheet_polyakov/run0004/params.json")
    ap_prep.add_argument("--out-root", default="outputs/worldsheet_polyakov")

    ap_run = sub.add_parser("run", help="Run all 3x3 sweeps (or only missing)")
    ap_run.add_argument("--out-root", default="outputs/worldsheet_polyakov")
    ap_run.add_argument("--only-missing", action="store_true")
    ap_run.add_argument("--use-run-sh", action="store_true", default=True)
    ap_run.add_argument("--no-run-sh", action="store_true")

    ap_tab = sub.add_parser("table", help="Generate LaTeX or Markdown table from sweep outputs")
    ap_tab.add_argument("--out-root", default="outputs/worldsheet_polyakov")
    ap_tab.add_argument("--format", choices=["latex", "md"], default="latex")

    args = ap.parse_args()

    repo_root = Path.cwd()
    out_root = repo_root / args.out_root

    if args.cmd == "prepare":
        base = repo_root / args.base
        assert base.exists(), f"Base params not found: {base}"
        for W in W_LIST:
            for s in S_LIST:
                if s >= W:
                    continue
                p = ensure_params(base, out_root, W, s)
                print("[ok] wrote", p)
        return

    if args.cmd == "run":
        use_run_sh = True
        if args.no_run_sh:
            use_run_sh = False

        for W in W_LIST:
            for s in S_LIST:
                if s >= W:
                    continue
                ddir = sweep_dir(out_root, W, s)
                p = ddir / "params.json"
                summ = ddir / "sweeps_summary.json"

                if not p.exists():
                    raise SystemExit(f"Missing params: {p} (run prepare first)")

                if args.only_missing and summ.exists():
                    continue

                rc = run_one(p, use_run_sh=use_run_sh)
                if rc != 0:
                    raise SystemExit(f"Run failed for W={W}, stride={s} (rc={rc})")
        return

    if args.cmd == "table":
        rows = []
        for W in W_LIST:
            for s in S_LIST:
                if s >= W:
                    continue
                ddir = sweep_dir(out_root, W, s)
                summ = ddir / "sweeps_summary.json"
                if summ.exists():
                    Nw, mf, r2m, r2a = read_metrics(summ, W, s)
                else:
                    Nw = mf = r2m = r2a = None
                rows.append((W, s, Nw, mf, r2m, r2a))

        if args.format == "md":
            print("| W_tau | stride | Nw | macro_frac | R2_macro | R2_all |")
            print("|---:|---:|---:|---:|---:|---:|")
            for W, s, Nw, mf, r2m, r2a in rows:
                print(f"| {W} | {s} | {Nw if Nw is not None else '--'} | {fmt(mf)} | {fmt(r2m)} | {fmt(r2a)} |")
            return

        # LaTeX tabular (body + header) – ready to paste or \input{}
        print(r"\begin{tabular}{cccccc}")
        print(r"\toprule")
        print(r"$W_\tau$ & stride & $N_w$ & macro\_frac & $R^2_{\mathrm{macro}}$ & $R^2_{\mathrm{all}}$ \\")
        print(r"\midrule")
        for W, s, Nw, mf, r2m, r2a in rows:
            Nw_s = str(Nw) if Nw is not None else "--"
            print(f"{W} & {s} & {Nw_s} & {fmt(mf)} & {fmt(r2m)} & {fmt(r2a)} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        return


if __name__ == "__main__":
    main()