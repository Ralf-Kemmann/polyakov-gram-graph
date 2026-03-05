from __future__ import annotations
from pathlib import Path
import re

p = Path("code/worldsheet_polyakov/compute_K_dph.py")
txt = p.read_text(encoding="utf-8")

# Replace dict entries that reference missing variables with inline calls.
txt2, n1 = re.subn(
    r'("fit_all"\s*:\s*)metrics_all(\s*,)',
    r'\1linear_fit_metrics(x, y)\2',
    txt
)

txt2, n2 = re.subn(
    r'("fit_macro_q20_q80"\s*:\s*)metrics_macro(\s*,)',
    r'\1linear_fit_metrics(x[m_macro], y[m_macro])\2',
    txt2
)

# (Optional) also fix the common typo if it exists anywhere
txt2 = re.sub(r'\betrics_all\b', 'metrics_all', txt2)

p.write_text(txt2, encoding="utf-8")
print(f"[ok] patched {p} (fit_all inline: {n1}, fit_macro inline: {n2})")
