#!/usr/bin/env bash
set -euo pipefail

# First pilot: single-seed, cheap matrix.
# Expected runtime depends on CPU, disk, and model cache. Start with MAX_STEPS=100 for sanity.
VARIANTS=(off tip_h tip_hd eopd tcs_no_trust tcs_no_adaptive tcs)
SEED=${SEED:-42}
for v in "${VARIANTS[@]}"; do
  echo "===== Running ${v} seed ${SEED} ====="
  bash scripts/train_one.sh "${v}" "${SEED}"
done
