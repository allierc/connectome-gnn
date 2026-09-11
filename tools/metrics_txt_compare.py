"""Compare a legacy results/metrics.txt with a catalogue one through extraction_gate.RENAMES.

Usage: python tools/metrics_txt_compare.py <before/metrics.txt> <after/metrics.txt>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extraction_gate import RENAMES
def load(p):
    d = {}
    for ln in open(p):
        if ":" not in ln: continue
        k, v = ln.split(":", 1); k, v = k.strip(), v.strip()
        try: d[k] = float(v)
        except ValueError: d[k] = v
    return d
a = {RENAMES.get(k, k): v for k, v in load(sys.argv[1]).items()}
b = load(sys.argv[2])
print(f"{'key':34s} {'before':>18s} {'after':>18s}  note")
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k), b.get(k)
    if va is None: note = "NEW"
    elif vb is None: note = "GONE"
    elif isinstance(va, float) and isinstance(vb, float):
        note = "same" if abs(va - vb) <= 1e-3 * max(1.0, abs(va)) else "MOVED"
    else: note = "same" if va == vb else "MOVED"
    f = lambda v: "" if v is None else (f"{v:18.6g}" if isinstance(v, float) else f"{v:>18s}")
    print(f"{k:34s} {f(va):>18s} {f(vb):>18s}  {note}")
