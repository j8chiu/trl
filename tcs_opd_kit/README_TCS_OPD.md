# TCS-OPD pilot kit for TRL DistillationTrainer

This kit adds a loss-only TCS-OPD pilot on top of Hugging Face TRL's experimental `DistillationTrainer`.

## Files

- `tcs_distillation.py`: standalone training script with `TcsDistillationTrainer`.
- `eval_math.py`: multi-GPU MATH-500 evaluator.
- `scripts/train_one.sh`: train one variant.
- `scripts/run_pilot_matrix.sh`: train all first-pilot variants.
- `scripts/eval_one.sh`: evaluate one checkpoint/adapter.
- `scripts/eval_matrix.sh`: evaluate all pilot outputs.

## Where to put the files

Recommended:

```bash
git clone https://github.com/huggingface/trl.git
cd trl
pip install -e ".[peft]"
cp /path/to/tcs_opd_kit/tcs_distillation.py .
cp /path/to/tcs_opd_kit/eval_math.py .
cp -r /path/to/tcs_opd_kit/scripts .
```

The script can also run outside the TRL repo if `trl` is already installed and new enough.

## Hardware target

The default config is designed for **4 × RTX 3090 24GB**:

- student: `Qwen/Qwen2.5-0.5B-Instruct`
- teacher: `Qwen/Qwen2.5-1.5B-Instruct`
- local teacher, same tokenizer
- LoRA training
- fp16
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `max_length=768`
- `max_completion_length=384`
- student rollout via colocated vLLM by default in `scripts/train_one.sh`
- sparse top-k loss with `loss_top_k=32` and tail bucket

## Sanity run

Start with 50-100 steps before running the full matrix:

```bash
export NPROC=4
export MAX_TRAIN_SAMPLES=512
export MAX_STEPS=50
bash scripts/train_one.sh tcs 42
bash scripts/eval_one.sh outputs/pilot/tcs_seed42
```

To disable vLLM for rollout generation and fall back to `transformers.generate()`:

```bash
export USE_VLLM=false
```

## First pilot matrix

```bash
export NPROC=4
export MAX_TRAIN_SAMPLES=2000
export MAX_STEPS=300
bash scripts/run_pilot_matrix.sh
bash scripts/eval_matrix.sh
```

Variants:

- `off`: original TRL vanilla OPD loss, reverse KL, `tcs_variant=off`.
- `tip_h`: high student-uncertainty selected tokens + reverse KL.
- `tip_hd`: high student-uncertainty or high disagreement selected tokens + reverse KL.
- `eopd`: all tokens + teacher-uncertainty adaptive KL.
- `tcs_no_trust`: selected tokens + adaptive KL, no teacher trust multiplier.
- `tcs_no_adaptive`: selected tokens + teacher trust, fixed reverse KL.
- `tcs`: full TCS-v1.

## Suggested second run

After sanity and first matrix, increase:

```bash
export MAX_TRAIN_SAMPLES=10000
export MAX_STEPS=800
export MAX_COMPLETION_LENGTH=512
export MAX_LENGTH=1024
bash scripts/train_one.sh off 42
bash scripts/train_one.sh tip_hd 42
bash scripts/train_one.sh eopd 42
bash scripts/train_one.sh tcs 42
bash scripts/eval_matrix.sh
```

Then repeat with seed 43 if the trend is promising.

## OOM fallback knobs

Try these in order:

```bash
export MAX_COMPLETION_LENGTH=256
export MAX_LENGTH=640
export LOSS_TOP_K=16
export GAS=2
```

The most memory-sensitive part is the full LM logits from both student and teacher. The custom TCS loss uses sparse top-k + tail after logits are materialized, which keeps the loss-side memory modest, but model forward logits still scale with sequence length.

## Main result table

Report at least:

| variant | train samples | steps | selected ratio | MATH-500 acc | avg length | truncation fraction |
|---|---:|---:|---:|---:|---:|---:|

The training logs include:

- `tcs/selected_ratio`
- `tcs/student_uncertainty`
- `tcs/teacher_uncertainty`
- `tcs/js_div`
- `tcs/beta`
- `tcs/trust`
- `completions/on_policy_mean_length`
- `completions/truncated_fraction`

## Notes

Use `trl-lib/DeepMath-103K` for training because it is already prompt-only conversational data with `prompt` and `solution` columns. `tcs_distillation.py` maps it to the `messages` column expected by TRL's collator.
