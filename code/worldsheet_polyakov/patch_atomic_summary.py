from __future__ import annotations
from pathlib import Path
import re
import shutil
from datetime import datetime

p = Path("code/worldsheet_polyakov/compute_K_dph.py")
txt = p.read_text(encoding="utf-8")

# already patched?
if "tmp_sum = out_sum + '.tmp'" in txt and "os.replace(tmp_sum, out_sum)" in txt:
    print("[ok] compute_K_dph.py already has atomic summary write")
    raise SystemExit(0)

# backup
bak = p.with_suffix(p.suffix + f".bak_atomic_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(p, bak)

pat = re.compile(
    r"""
(?P<indent>^[ \t]*)
with\ open\(\s*os\.path\.join\(\s*run_dir\s*,\s*summary_file\s*\)\s*,\s*[\"']w[\"'][^)]*\)\s*as\s*f\s*:\s*\n
(?P=indent)[ \t]*json\.dump\(\s*sweep_results\s*,\s*f\s*,\s*indent\s*=\s*2\s*\)\s*\n
""",
    re.VERBOSE | re.MULTILINE,
)

m = pat.search(txt)
if not m:
    raise SystemExit("Could not find the sweeps_summary write block to patch.")

ind = m.group("indent")

replacement = (
f"{ind}# ---- write summary atomically (prevents truncated json on abort) ----\n"
f"{ind}if not sweep_results.get('results'):\n"
f"{ind}    raise RuntimeError('[run] no sweep results collected; check eps_sweep/k_sweep and indentation upstream.')\n"
f"{ind}out_sum = os.path.join(run_dir, summary_file)\n"
f"{ind}tmp_sum = out_sum + '.tmp'\n"
f"{ind}with open(tmp_sum, 'w', encoding='utf-8') as f:\n"
f"{ind}    json.dump(sweep_results, f, indent=2)\n"
f"{ind}os.replace(tmp_sum, out_sum)\n"
)

txt2 = pat.sub(replacement, txt, count=1)
p.write_text(txt2, encoding="utf-8")

print("[ok] patched atomic summary write:", p)
print("[ok] backup:", bak)
