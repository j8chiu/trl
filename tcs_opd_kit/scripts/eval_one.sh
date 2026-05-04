#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/eval_one.sh outputs/pilot/tcs_seed42
MODEL_DIR=${1:?Model or adapter directory required}
NPROC=${NPROC:-4}
MAX_SAMPLES=${MAX_SAMPLES:-500}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
OUT_DIR=${MODEL_DIR}/eval_math500

export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes "${NPROC}" eval_math.py \
  --model "${MODEL_DIR}" \
  --dataset HuggingFaceH4/MATH-500 \
  --split test \
  --prompt_column problem \
  --answer_column answer \
  --max_samples "${MAX_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.0 \
  --dtype float16 \
  --output_dir "${OUT_DIR}"
