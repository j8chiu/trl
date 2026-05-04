import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/pilot")
rows = []
for p in sorted(root.glob("*_seed*/eval_math500/metrics.json")):
    with open(p) as f:
        m = json.load(f)
    rows.append((p.parent.parent.name, m.get("n"), m.get("accuracy")))
print("run\tn\taccuracy")
for name, n, acc in rows:
    print(f"{name}\t{n}\t{acc:.4f}")
