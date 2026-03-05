#!/usr/bin/env python3
from __future__ import annotations
import re, shutil
from datetime import datetime
from pathlib import Path

TARGET = Path("code/worldsheet_polyakov/compute_K_dph.py")

def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, bak)
    return bak

def ensure_imports(txt: str) -> str:
    if "from sanity_checks import preflight_windows, post_k" in txt:
        return txt
    m = re.search(r"^import numpy as np\s*$", txt, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Couldn't find 'import numpy as np' to insert sanity import.")
    ins = "\nfrom sanity_checks import preflight_windows, post_k\n"
    i = m.end()
    return txt[:i] + ins + txt[i:]

def ensure_strict_sanity(txt: str) -> str:
    if "strict_sanity = bool(prag.get('strict_sanity', True))" in txt:
        return txt
    # Insert after the line that defines prag (prag = ...)
    m = re.search(r"^\s*prag\s*=\s*.*$", txt, flags=re.MULTILINE)
    if not m:
        raise RuntimeError("Couldn't find line defining prag = ...")
    line_end = txt.find("\n", m.end())
    if line_end < 0: line_end = len(txt)
    ins = "\nstrict_sanity = bool(prag.get('strict_sanity', True))\n"
    return txt[:line_end] + ins + txt[line_end:]

def insert_sanity_before_saveK(txt: str) -> tuple[str, bool]:
    if "sanity_W{W_tau}.json" in txt or "sanity_W" in txt and "preflight_windows" in txt:
        # already inserted somewhere
        pass

    # Find the first save of K_W{W_tau}.npy
    pat = r"^(?P<ind>\s*)np\.save\(\s*os\.path\.join\(\s*run_dir\s*,\s*f\"K_W\{W_tau\}\.npy\"\s*\)\s*,\s*K\s*\)\s*$"
    m = re.search(pat, txt, flags=re.MULTILINE)
    if not m:
        # sometimes they use "K_W{W_tau}.npy" without f-string braces, fallback:
        pat2 = r"^(?P<ind>\s*)np\.save\(\s*os\.path\.join\(\s*run_dir\s*,\s*.*K_W.*\)\s*,\s*K\s*\)\s*$"
        m = re.search(pat2, txt, flags=re.MULTILINE)
        if not m:
            raise RuntimeError("Couldn't find np.save(... K_W{W_tau}.npy ..., K) line to hook sanity.")
    ind = m.group("ind")
    insert = (
        f"{ind}# --- sanity layer (hard stop on degenerate runs) ---\n"
        f"{ind}sanity_win = preflight_windows(Nt=Nt, W_tau=W_tau, stride=stride, strict=strict_sanity)\n"
        f"{ind}sanity_k = post_k(K, strict=strict_sanity)\n"
        f"{ind}try:\n"
        f"{ind}    import json\n"
        f"{ind}    with open(os.path.join(run_dir, f\"sanity_W{{W_tau}}.json\"), \"w\") as f:\n"
        f"{ind}        json.dump({{\"windows\": sanity_win, \"K\": sanity_k}}, f, indent=2)\n"
        f"{ind}except Exception as _e:\n"
        f"{ind}    print(f\"[sanity] warning: could not write sanity json: {{_e}}\")\n"
    )
    pos = m.start()
    return txt[:pos] + insert + txt[pos:], True

def main():
    if not TARGET.exists():
        raise SystemExit(f"Missing target: {TARGET}")
    bak = backup(TARGET)
    txt = TARGET.read_text(encoding="utf-8")

    txt = ensure_imports(txt)
    txt = ensure_strict_sanity(txt)
    txt, changed = insert_sanity_before_saveK(txt)

    TARGET.write_text(txt, encoding="utf-8")
    print("[sanity-layer-v2] patched:", TARGET)
    print("backup:", bak)
    print("changed:", changed)

if __name__ == "__main__":
    main()
