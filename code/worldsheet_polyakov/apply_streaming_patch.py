#!/usr/bin/env python3
"""
Apply streaming patches to compute_K_dph.py:

(A) aggregate_K_over_ensemble: streaming sum (no Ks list, no np.stack)
(B) sampling: optional memmap streaming in batches (no full samples array in RAM)

Creates a timestamped backup next to compute_K_dph.py before editing.
"""

from __future__ import annotations
import re
import shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("code/worldsheet_polyakov/compute_K_dph.py")


AGG_STREAMING_FUNC = r'''def aggregate_K_over_ensemble(
    samples: np.ndarray,
    W_tau: int,
    stride: int,
    delta_sigma: float,
    delta_tau: float,
    weights_type: str,
    store_abs: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute K for each sample and average (STREAMING, no Ks list, no stack).
    samples shape: (M, D, Ns, Nt)
    """
    starts_ref = None
    K_sum = None
    M = int(samples.shape[0])

    # IMPORTANT: per-sample abs is wasted; we define abs after averaging.
    for m in range(M):
        K, _K_abs_unused, starts = compute_K_fair(
            samples[m], W_tau=W_tau, stride=stride,
            delta_sigma=delta_sigma, delta_tau=delta_tau,
            weights_type=weights_type, store_abs=False
        )
        if starts_ref is None:
            starts_ref = starts
        else:
            if not np.array_equal(starts_ref, starts):
                raise RuntimeError("Window starts mismatch across samples (should not happen).")

        if K_sum is None:
            K_sum = K.copy()
        else:
            K_sum += K

    K_mean = K_sum / max(M, 1)
    if store_abs:
        K_abs_mean = np.abs(K_mean)
    else:
        K_abs_mean = K_mean.copy()
    return K_mean, K_abs_mean, starts_ref
'''


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, bak)
    return bak


def patch_aggregate(txt: str) -> tuple[str, bool]:
    if "STREAMING, no Ks list, no stack" in txt:
        return txt, False

    # Replace the whole function block up to next def knn_graph_from_dist
    pat = r"def aggregate_K_over_ensemble\([\s\S]*?\n\ndef knn_graph_from_dist"
    m = re.search(pat, txt)
    if not m:
        raise RuntimeError("Could not locate aggregate_K_over_ensemble block.")
    out = re.sub(pat, AGG_STREAMING_FUNC + "\n\n\ndef knn_graph_from_dist", txt, count=1)
    return out, True


def patch_sampling_to_memmap(txt: str) -> tuple[str, bool]:
    if "stream_samples_memmap" in txt:
        return txt, False

    # We replace this exact block:
    #   samples = langevin_sample(...)
    #   if save_samples: np.save(..., samples)
    #   samples_use = samples
    #   M_use = samples.shape[0]
    #
    # with a memmap streaming variant (batchwise), controlled by pragmatics:
    #   pragmatics.stream_samples_memmap (bool, default True)
    #   pragmatics.stream_saved_batch (int, default 1)
    #   pragmatics.samples_memmap_dtype (str, default "float32")
    #
    pat = (
        r"(?P<indent>^[ \t]*)samples\s*=\s*langevin_sample\(\n"
        r"(?P<callargs>[\s\S]*?)"
        r"(?P=indent)\)\n"
        r"(?P=indent)if\s+save_samples:\n"
        r"(?P=indent)[ \t]*np\.save\(\s*os\.path\.join\(\s*run_dir\s*,\s*\"samples\.npy\"\s*\)\s*,\s*samples\s*\)\n"
        r"(?P=indent)samples_use\s*=\s*samples\n"
        r"(?P=indent)M_use\s*=\s*samples\.shape\[0\]\n"
    )
    m = re.search(pat, txt, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Could not locate sampling block (samples = langevin_sample ... samples.npy ...).")

    indent = m.group("indent")
    callargs = m.group("callargs")

    # Batch-call args: n_saved_samples -> mb
    batch_args = re.sub(r"n_saved_samples\s*=\s*n_saved_effective", "n_saved_samples=mb", callargs)

    # If there is a seed=..., make it distinct per batch by adding + off
    def seed_repl(mm: re.Match) -> str:
        rhs = mm.group(1).strip()
        return f"seed=int({rhs})+off"
    batch_args = re.sub(r"seed\s*=\s*([^,\n]+)", seed_repl, batch_args)

    replacement = (
        f"{indent}# --- streaming samples (no full samples array in RAM) ---\n"
        f"{indent}stream_samples_memmap = bool(prag.get('stream_samples_memmap', True))\n"
        f"{indent}stream_saved_batch = int(prag.get('stream_saved_batch', 1))\n"
        f"{indent}mm_dtype = str(prag.get('samples_memmap_dtype', 'float32'))\n"
        f"\n"
        f"{indent}if stream_samples_memmap:\n"
        f"{indent}    # Write samples directly to a memmap file, batchwise.\n"
        f"{indent}    # Shape: (M, D, Ns, Nt) with M = n_saved_effective\n"
        f"{indent}    mm_path = os.path.join(run_dir, 'samples.mm')\n"
        f"{indent}    dtype = getattr(np, mm_dtype)\n"
        f"{indent}    samples_use = np.memmap(mm_path, mode='w+', dtype=dtype, shape=(n_saved_effective, D, Ns, Nt))\n"
        f"{indent}    for off in range(0, n_saved_effective, stream_saved_batch):\n"
        f"{indent}        mb = min(stream_saved_batch, n_saved_effective - off)\n"
        f"{indent}        samples_b = langevin_sample(\n"
        f"{batch_args}"
        f"{indent}        )\n"
        f"{indent}        samples_use[off:off+mb, :, :, :] = samples_b\n"
        f"{indent}        del samples_b\n"
        f"{indent}    # keep legacy flag meaning: save_samples_npy writes a .npy copy (optional)\n"
        f"{indent}    if save_samples:\n"
        f"{indent}        np.save(os.path.join(run_dir, 'samples.npy'), np.asarray(samples_use))\n"
        f"{indent}    M_use = samples_use.shape[0]\n"
        f"{indent}else:\n"
        f"{indent}    samples = langevin_sample(\n"
        f"{callargs}"
        f"{indent}    )\n"
        f"{indent}    if save_samples:\n"
        f"{indent}        np.save(os.path.join(run_dir, 'samples.npy'), samples)\n"
        f"{indent}    samples_use = samples\n"
        f"{indent}    M_use = samples.shape[0]\n"
    )

    out = re.sub(pat, replacement, txt, count=1, flags=re.MULTILINE)
    return out, True


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Target not found: {TARGET}")

    original = TARGET.read_text(encoding="utf-8")
    bak = backup_file(TARGET)

    txt, chg1 = patch_aggregate(original)
    txt, chg2 = patch_sampling_to_memmap(txt)

    TARGET.write_text(txt, encoding="utf-8")

    print("[patch] done")
    print(f"  backup: {bak}")
    print(f"  aggregate_K_over_ensemble patched: {chg1}")
    print(f"  sampling->memmap patched: {chg2}")
    print("  next: python3 -m py_compile code/worldsheet_polyakov/compute_K_dph.py")


if __name__ == "__main__":
    main()
