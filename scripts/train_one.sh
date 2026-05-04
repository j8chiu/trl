#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_one.sh tcs 42
#   bash scripts/train_one.sh off 42   # TRL vanilla OPD baseline

VARIANT=${1:-tcs}
SEED=${2:-42}
NPROC=${NPROC:-4}

STUDENT=${STUDENT:-Qwen/Qwen2.5-0.5B-Instruct}
TEACHER=${TEACHER:-Qwen/Qwen2.5-1.5B-Instruct}
DATASET=${DATASET:-trl-lib/DeepMath-103K}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-2000}
MAX_STEPS=${MAX_STEPS:-300}
MAX_LENGTH=${MAX_LENGTH:-768}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-384}
LOSS_TOP_K=${LOSS_TOP_K:-32}
KEEP_RATIO=${KEEP_RATIO:-0.5}
LR=${LR:-2e-4}
GAS=${GAS:-4}
BSZ=${BSZ:-1}
OUT_ROOT=${OUT_ROOT:-outputs/pilot}

OUT_DIR=${OUT_ROOT}/${VARIANT}_seed${SEED}
mkdir -p "${OUT_DIR}"

# 3090s are fp16-oriented. Disabling tokenizer parallelism avoids noisy CPU contention.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

COMMON_ARGS=(
  --model_name_or_path "${STUDENT}"
  --teacher_model_name_or_path "${TEACHER}"
  --dataset_name "${DATASET}"
  --dataset_train_split train
  --dataset_test_split test
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --output_dir "${OUT_DIR}"
  --seed "${SEED}"
  --max_steps "${MAX_STEPS}"
  --learning_rate "${LR}"
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --per_device_train_batch_size "${BSZ}"
  --gradient_accumulation_steps "${GAS}"
  --generation_batch_size "${GEN_BSZ:-$((BSZ * GAS))}"
  --num_generations 1
  --lmbda 1.0
  --temperature 0.7
  --top_p 0.95
  --top_k 0
  --max_length "${MAX_LENGTH}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --loss_top_k "${LOSS_TOP_K}"
  --loss_add_tail true
  --fp16 true
  --bf16 false
  --dtype float16
  --gradient_checkpointing true
  --use_peft true
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.05
  --attn_implementation sdpa
  --disable_dropout true
  --eval_strategy no
  --save_strategy steps
  --save_steps ${SAVE_STEPS:-100000}
  --logging_steps 10
  --log_completions ${LOG_COMPLETIONS:-false}
  --log_completions_steps 50
  --num_completions_to_print 4
  --report_to none
)

# The vanilla OPD baseline should call TRL's original loss path.
if [[ "${VARIANT}" == "off" || "${VARIANT}" == "vanilla" ]]; then
  EXTRA_ARGS=(--tcs_variant off --beta 1.0)
else
  EXTRA_ARGS=(
    --tcs_variant "${VARIANT}"
    --tcs_keep_ratio "${KEEP_RATIO}"
    --tcs_gamma_div 1.0
    --tcs_trust_floor 0.25
    --tcs_beta_min 0.05
    --tcs_beta_max 0.95
    --tcs_entropy_power 1.0
    --tcs_tail_uncertainty_weight 0.5
    --tcs_min_tokens_per_seq 8
    --beta 1.0
  )
fi

accelerate launch --num_processes "${NPROC}" tcs_distillation.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
