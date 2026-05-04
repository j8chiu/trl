# Lightweight multi-GPU math evaluation for TCS-OPD pilots.
# Supports HuggingFaceH4/MATH-500 and prompt-only conversational datasets.

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_or_adapter: str, dtype: str = "float16"):
    torch_dtype = getattr(torch, dtype)
    try:
        from peft import AutoPeftModelForCausalLM

        # Works for LoRA adapter directories produced by TRL.
        return AutoPeftModelForCausalLM.from_pretrained(
            model_or_adapter,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except Exception:
        return AutoModelForCausalLM.from_pretrained(
            model_or_adapter,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )


def get_prompt(example: dict[str, Any], prompt_column: str | None, prompt_suffix: str) -> list[dict[str, str]]:
    cols = []
    if prompt_column:
        cols.append(prompt_column)
    cols.extend(["prompt", "messages", "question", "problem", "instruction", "input"])
    for col in cols:
        if col in example and example[col] is not None:
            obj = example[col]
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "role" in obj and "content" in obj:
                return [obj]
            return [{"role": "user", "content": str(obj) + prompt_suffix}]
    raise KeyError(f"No prompt-like column in example keys: {list(example.keys())}")


def get_answer(example: dict[str, Any], answer_column: str | None) -> str:
    cols = []
    if answer_column:
        cols.append(answer_column)
    cols.extend(["answer", "solution", "final_answer", "target"])
    for col in cols:
        if col in example and example[col] is not None:
            return str(example[col])
    return ""


def extract_boxed(text: str) -> str | None:
    # Handles \boxed{...} with nested braces approximately.
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    i = idx + len(marker)
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def extract_answer(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    # Common final-answer patterns.
    patterns = [
        r"final answer is[:\s]*([^\n]+)",
        r"answer is[:\s]*([^\n]+)",
        r"therefore[,\s]*(?:the answer is)?[:\s]*([^\n]+)",
    ]
    lower = text.lower()
    for pat in patterns:
        m = re.search(pat, lower)
        if m:
            return text[m.start(1) : m.end(1)].strip()
    # Fallback: last non-empty line, stripped of leading words.
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[-1] if lines else text.strip()


def normalize_answer(x: str) -> str:
    x = extract_answer(x)
    x = x.strip()
    x = x.replace("$", "")
    x = x.replace(r"\left", "").replace(r"\right", "")
    x = x.replace(r"\,", "").replace(r"\!", "")
    x = x.replace(" ", "")
    x = x.replace("\\dfrac", "\\frac")
    x = x.replace("\\tfrac", "\\frac")
    x = x.strip(".。")
    return x


def verify_answer(gold: str, pred_text: str) -> bool:
    pred = extract_answer(pred_text)
    # Prefer math-verify if available.
    try:
        from math_verify import parse, verify

        gold_parsed = parse(gold)
        pred_parsed = parse(pred)
        if gold_parsed and pred_parsed:
            return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        pass
    return normalize_answer(gold) == normalize_answer(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model path or LoRA adapter directory.")
    ap.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--split", default="test")
    ap.add_argument("--prompt_column", default=None)
    ap.add_argument("--answer_column", default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--output_dir", default="eval_outputs")
    ap.add_argument(
        "--prompt_suffix",
        default="\n\nPlease reason step by step, and put your final answer in \\boxed{}.",
    )
    args = ap.parse_args()

    accelerator = Accelerator()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.shard(num_shards=accelerator.num_processes, index=accelerator.process_index, contiguous=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model, dtype=args.dtype)
    model.to(accelerator.device)
    model.eval()

    records = []
    iterator = tqdm(ds, disable=not accelerator.is_local_main_process)
    for ex in iterator:
        messages = get_prompt(ex, args.prompt_column, args.prompt_suffix)
        gold = get_answer(ex, args.answer_column)
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to(accelerator.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output_ids[0, input_ids.shape[1] :]
        pred_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        correct = verify_answer(gold, pred_text)
        records.append(
            {
                "correct": bool(correct),
                "gold": gold,
                "pred_answer": extract_answer(pred_text),
                "pred_text": pred_text,
                "prompt": messages[-1]["content"] if messages else "",
            }
        )

    rank_path = Path(args.output_dir) / f"predictions_rank{accelerator.process_index}.jsonl"
    with open(rank_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        all_records = []
        for p in sorted(Path(args.output_dir).glob("predictions_rank*.jsonl")):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    all_records.append(json.loads(line))
        acc = sum(r["correct"] for r in all_records) / max(1, len(all_records))
        metrics = {"n": len(all_records), "accuracy": acc}
        with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
