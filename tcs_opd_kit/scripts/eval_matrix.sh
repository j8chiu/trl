#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT=${OUT_ROOT:-outputs/pilot}
for d in ${OUT_ROOT}/*_seed*; do
  [[ -d "$d" ]] || continue
  echo "===== Evaluating $d ====="
  bash scripts/eval_one.sh "$d"
done
python scripts/summarize_eval.py "${OUT_ROOT}"
