# Copyright 2026
# Minimal TCS-OPD trainer script built on Hugging Face TRL DistillationTrainer.
# Place this file at the root of a cloned TRL repo, or run it in an environment
# where `trl` is installed from source.

import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from transformers import GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.distillation import DistillationConfig, DistillationTrainer


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class TcsScriptArguments(ScriptArguments):
    """Extra dataset controls for quick pilot experiments."""

    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "Limit the number of training samples after loading the train split."},
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={"help": "Limit the number of eval samples after loading the eval split."},
    )
    prompt_column: str | None = field(
        default=None,
        metadata={
            "help": "Optional prompt column. If unset, the script auto-detects messages/prompt/question/problem."
        },
    )
    answer_column: str | None = field(
        default=None,
        metadata={"help": "Unused by training; kept for compatibility with math datasets."},
    )
    prompt_suffix: str = field(
        default="\n\nPlease reason step by step, and put your final answer in \\boxed{}.",
        metadata={"help": "Suffix appended to plain string prompts/questions/problems."},
    )


@dataclass
class TcsDistillationConfig(DistillationConfig):
    """DistillationConfig + TCS-OPD loss knobs.

    tcs_variant:
      off              -> use TRL's original DistillationTrainer loss
      tip_h            -> high student-uncertainty token selection + reverse KL
      tip_hd           -> high student-uncertainty OR high disagreement + reverse KL
      eopd             -> all-token entropy-adaptive KL
      tcs_no_trust     -> TIP-HD need score + entropy-adaptive KL
      tcs_no_adaptive  -> TCS weighting + fixed reverse KL
      tcs              -> full TCS-v1: trust-calibrated token selection + adaptive KL
    """

    tcs_variant: str = field(
        default="off",
        metadata={"help": "One of: off, tip_h, tip_hd, eopd, tcs_no_trust, tcs_no_adaptive, tcs."},
    )
    tcs_keep_ratio: float = field(default=0.5, metadata={"help": "Per-sequence fraction of valid tokens kept."})
    tcs_gamma_div: float = field(default=1.0, metadata={"help": "Weight for low-entropy/high-divergence need score."})
    tcs_trust_floor: float = field(default=0.25, metadata={"help": "Minimum teacher trust multiplier."})
    tcs_beta_min: float = field(default=0.05, metadata={"help": "Minimum adaptive reverse-KL weight."})
    tcs_beta_max: float = field(default=0.95, metadata={"help": "Maximum adaptive reverse-KL weight."})
    tcs_entropy_power: float = field(default=1.0, metadata={"help": "Power for uncertainty-to-beta schedule."})
    tcs_tail_uncertainty_weight: float = field(
        default=0.5,
        metadata={"help": "Blend coefficient for sparse entropy and tail mass uncertainty."},
    )
    tcs_min_tokens_per_seq: int = field(
        default=8,
        metadata={"help": "Minimum number of tokens selected per non-empty sequence."},
    )
    tcs_log_every_loss: bool = field(
        default=True,
        metadata={"help": "Whether to log diagnostic statistics from the custom loss."},
    )

    def __post_init__(self):
        super().__post_init__()
        valid = {"off", "tip_h", "tip_hd", "eopd", "tcs_no_trust", "tcs_no_adaptive", "tcs"}
        if self.tcs_variant not in valid:
            raise ValueError(f"tcs_variant must be one of {sorted(valid)}, got {self.tcs_variant!r}.")
        if not 0.0 < self.tcs_keep_ratio <= 1.0:
            raise ValueError(f"tcs_keep_ratio must be in (0, 1], got {self.tcs_keep_ratio}.")
        if not 0.0 <= self.tcs_trust_floor <= 1.0:
            raise ValueError(f"tcs_trust_floor must be in [0, 1], got {self.tcs_trust_floor}.")
        if not 0.0 <= self.tcs_beta_min <= self.tcs_beta_max <= 1.0:
            raise ValueError(
                f"Require 0 <= tcs_beta_min <= tcs_beta_max <= 1, got "
                f"{self.tcs_beta_min}, {self.tcs_beta_max}."
            )
        if not 0.0 <= self.tcs_tail_uncertainty_weight <= 1.0:
            raise ValueError(
                f"tcs_tail_uncertainty_weight must be in [0, 1], got {self.tcs_tail_uncertainty_weight}."
            )


# -----------------------------
# Dataset helpers
# -----------------------------


def _detect_prompt(example: dict[str, Any], preferred: str | None = None) -> Any:
    if "messages" in example and example["messages"] is not None:
        return example["messages"]
    columns = []
    if preferred:
        columns.append(preferred)
    columns.extend(["prompt", "question", "problem", "instruction", "input"])
    for col in columns:
        if col in example and example[col] is not None:
            return example[col]
    raise KeyError(f"Could not find a prompt column. Example keys: {list(example.keys())}")


def _to_messages(prompt_obj: Any, prompt_suffix: str) -> list[dict[str, str]]:
    # TRL prompt-only datasets often store prompt as a list of chat messages.
    if isinstance(prompt_obj, list):
        # Already a conversational prompt; do not append suffix because the dataset
        # likely already formats the user message as intended.
        return prompt_obj
    if isinstance(prompt_obj, dict) and "content" in prompt_obj and "role" in prompt_obj:
        return [prompt_obj]
    if not isinstance(prompt_obj, str):
        prompt_obj = str(prompt_obj)
    return [{"role": "user", "content": prompt_obj + prompt_suffix}]


def normalize_dataset_to_messages(dataset, script_args: TcsScriptArguments):
    """Convert common math dataset schemas to TRL DistillationTrainer's `messages` format."""

    def convert(example: dict[str, Any]):
        prompt_obj = _detect_prompt(example, script_args.prompt_column)
        return {"messages": _to_messages(prompt_obj, script_args.prompt_suffix)}

    def convert_split(ds):
        keep_cols = ds.column_names
        return ds.map(convert, remove_columns=keep_cols, desc="Converting to prompt-only messages")

    if isinstance(dataset, DatasetDict):
        return DatasetDict({k: convert_split(v) for k, v in dataset.items()})
    return convert_split(dataset)


# -----------------------------
# TCS loss helpers
# -----------------------------


def _add_tail_bucket_local(log_probs: torch.Tensor, valid_mask: torch.Tensor):
    """Append log(1 - sum(exp(selected_log_probs))) as a tail bucket."""
    neg_inf = torch.full((), float("-inf"), dtype=log_probs.dtype, device=log_probs.device)
    safe_log_probs = torch.where(valid_mask, log_probs, neg_inf)
    log_sum = torch.logsumexp(safe_log_probs, dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)  # ensures exp(log_sum) < 1
    tail = torch.log(-torch.expm1(log_sum))
    tail_mask = torch.ones_like(valid_mask[..., :1], dtype=torch.bool)
    return torch.cat([safe_log_probs, tail], dim=-1), torch.cat([valid_mask, tail_mask], dim=-1)


def _safe_kl_terms(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor, support_mask: torch.Tensor):
    """Return per-token FKL, RKL and JS on a sparse support+tail distribution.

    Shapes: [B, T, K]. Returned shapes: [B, T].
    """
    zero = torch.zeros((), dtype=student_log_probs.dtype, device=student_log_probs.device)
    s_lp = torch.where(support_mask, student_log_probs, zero)
    t_lp = torch.where(support_mask, teacher_log_probs, zero)
    s_p = torch.where(support_mask, student_log_probs.exp(), zero)
    t_p = torch.where(support_mask, teacher_log_probs.exp(), zero)

    fkl = torch.nan_to_num(t_p * (t_lp - s_lp), nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)
    rkl = torch.nan_to_num(s_p * (s_lp - t_lp), nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)

    m_p = 0.5 * (s_p + t_p)
    m_lp = torch.log(m_p.clamp_min(torch.finfo(m_p.dtype).tiny))
    js = 0.5 * (
        torch.nan_to_num(s_p * (s_lp - m_lp), nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)
        + torch.nan_to_num(t_p * (t_lp - m_lp), nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)
    )
    return fkl, rkl, js


def _sparse_entropy(log_probs: torch.Tensor, support_mask: torch.Tensor, eps: float = 1e-8):
    zero = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
    lp = torch.where(support_mask, log_probs, zero)
    p = torch.where(support_mask, log_probs.exp(), zero)
    ent = -torch.nan_to_num(p * lp, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=-1)
    support_count = support_mask.float().sum(dim=-1).clamp_min(2.0)
    return (ent / torch.log(support_count + eps)).clamp(0.0, 1.0)


def _tail_mass(log_probs: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
    # If add_tail=True, the last valid position is the tail bucket. In this script
    # add_tail is always strongly recommended. If tail is disabled, this still
    # returns zero because the final slot is not semantically guaranteed to be tail.
    return torch.where(support_mask[..., -1], log_probs[..., -1].exp(), torch.zeros_like(log_probs[..., -1]))


def _dedup_union_support(student_logits: torch.Tensor, teacher_logits: torch.Tensor, top_k: int):
    """Union of teacher and student top-k tokens with duplicate masking."""
    if top_k <= 0:
        raise ValueError("TCS sparse loss requires loss_top_k > 0. Use 16 or 32 on 3090 GPUs.")
    _, s_top = student_logits.topk(top_k, dim=-1)
    _, t_top = teacher_logits.topk(top_k, dim=-1)
    support = torch.cat([t_top, s_top], dim=-1)
    support_mask = torch.ones(support.shape, dtype=torch.bool, device=support.device)
    # Remove duplicates within the concatenated support. K is small, so this loop is cheap.
    for i in range(1, support.shape[-1]):
        prev_matches = support[..., i : i + 1] == support[..., :i]
        prev_valid = support_mask[..., :i]
        support_mask[..., i] &= ~(prev_matches & prev_valid).any(dim=-1)
    support = torch.where(support_mask, support, torch.zeros_like(support))
    return support, support_mask


class TcsDistillationTrainer(DistillationTrainer):
    """DistillationTrainer with optional TCS-OPD loss.

    The custom loss is implemented only for local teachers with same tokenizer.
    This is intentional for the first pilot: TCS needs teacher/student entropy and
    teacher-student divergence diagnostics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.args
        self.tcs_variant = getattr(cfg, "tcs_variant", "off")
        self.tcs_keep_ratio = getattr(cfg, "tcs_keep_ratio", 0.5)
        self.tcs_gamma_div = getattr(cfg, "tcs_gamma_div", 1.0)
        self.tcs_trust_floor = getattr(cfg, "tcs_trust_floor", 0.25)
        self.tcs_beta_min = getattr(cfg, "tcs_beta_min", 0.05)
        self.tcs_beta_max = getattr(cfg, "tcs_beta_max", 0.95)
        self.tcs_entropy_power = getattr(cfg, "tcs_entropy_power", 1.0)
        self.tcs_tail_uncertainty_weight = getattr(cfg, "tcs_tail_uncertainty_weight", 0.5)
        self.tcs_min_tokens_per_seq = getattr(cfg, "tcs_min_tokens_per_seq", 8)
        self.tcs_log_every_loss = getattr(cfg, "tcs_log_every_loss", True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.tcs_variant == "off":
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        self._raise_if_local_teacher_tokenizer_mismatch()
        if self.use_teacher_server:
            raise NotImplementedError("TCS-OPD v1 requires a local teacher; use_teacher_server is not supported.")
        if self.use_liger_loss:
            raise NotImplementedError("TCS-OPD v1 does not use the Liger fused JSD path.")
        if self.loss_top_k <= 0:
            raise ValueError("TCS-OPD sparse loss requires --loss_top_k > 0. Recommended: 16 or 32.")

        student_outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        prompt_length = self._compute_prompt_length(inputs)
        labels = inputs["labels"][:, prompt_length:]

        # Align logits with completion tokens: token at position t is predicted by previous hidden state.
        student_logits = student_outputs.logits[:, prompt_length - 1 : -1, :] / self.temperature
        with torch.no_grad():
            teacher_logits_full = self._get_teacher_logits(inputs)
            teacher_logits = teacher_logits_full[:, prompt_length - 1 : -1, :] / self.temperature
            del teacher_logits_full

        loss = self._tcs_sparse_loss(student_logits, teacher_logits, labels)
        return (loss, student_outputs) if return_outputs else loss

    def _tcs_sparse_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor):
        token_mask = labels != -100
        top_k = int(self.loss_top_k)

        support, support_mask = _dedup_union_support(student_logits, teacher_logits, top_k=top_k)
        neg_inf = torch.full((), float("-inf"), dtype=student_logits.dtype, device=student_logits.device)

        # Compute sparse log-probs without materializing full-vocab log_softmax.
        s_log_z = torch.logsumexp(student_logits, dim=-1, keepdim=True)
        t_log_z = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
        s_support_lp = student_logits.gather(-1, support) - s_log_z
        t_support_lp = teacher_logits.gather(-1, support) - t_log_z
        s_support_lp = torch.where(support_mask, s_support_lp, neg_inf)
        t_support_lp = torch.where(support_mask, t_support_lp, neg_inf)

        if self.loss_add_tail:
            s_lp, sparse_mask = _add_tail_bucket_local(s_support_lp, support_mask)
            t_lp, _ = _add_tail_bucket_local(t_support_lp, support_mask)
        else:
            # Renormalize within support. Tail metrics are disabled in this case.
            sparse_mask = support_mask
            s_lp = s_support_lp - torch.logsumexp(s_support_lp, dim=-1, keepdim=True)
            t_lp = t_support_lp - torch.logsumexp(t_support_lp, dim=-1, keepdim=True)

        fkl, rkl, js = _safe_kl_terms(s_lp, t_lp, sparse_mask)

        with torch.no_grad():
            h_s_sparse = _sparse_entropy(s_lp.detach(), sparse_mask)
            h_t_sparse = _sparse_entropy(t_lp.detach(), sparse_mask)
            tail_s = _tail_mass(s_lp.detach(), sparse_mask) if self.loss_add_tail else torch.zeros_like(h_s_sparse)
            tail_t = _tail_mass(t_lp.detach(), sparse_mask) if self.loss_add_tail else torch.zeros_like(h_t_sparse)

            w_tail = self.tcs_tail_uncertainty_weight
            u_s = ((1.0 - w_tail) * h_s_sparse + w_tail * tail_s).clamp(0.0, 1.0)
            u_t = ((1.0 - w_tail) * h_t_sparse + w_tail * tail_t).clamp(0.0, 1.0)

            div = (js.detach() / math.log(2.0)).clamp(0.0, 1.0)
            need_h = u_s
            need_hd = torch.maximum(u_s, self.tcs_gamma_div * (1.0 - u_s) * div).clamp(0.0, 1.0)
            trust = self.tcs_trust_floor + (1.0 - self.tcs_trust_floor) * (1.0 - u_t)
            beta_adapt = self.tcs_beta_min + (self.tcs_beta_max - self.tcs_beta_min) * (1.0 - u_t).pow(
                self.tcs_entropy_power
            )
            beta_adapt = beta_adapt.clamp(self.tcs_beta_min, self.tcs_beta_max)

            if self.tcs_variant == "tip_h":
                raw_weight = need_h
                beta = torch.ones_like(raw_weight)
                keep_ratio = self.tcs_keep_ratio
            elif self.tcs_variant == "tip_hd":
                raw_weight = need_hd
                beta = torch.ones_like(raw_weight)
                keep_ratio = self.tcs_keep_ratio
            elif self.tcs_variant == "eopd":
                raw_weight = torch.ones_like(need_h)
                beta = beta_adapt
                keep_ratio = 1.0
            elif self.tcs_variant == "tcs_no_trust":
                raw_weight = need_hd
                beta = beta_adapt
                keep_ratio = self.tcs_keep_ratio
            elif self.tcs_variant == "tcs_no_adaptive":
                raw_weight = need_hd * trust
                beta = torch.ones_like(raw_weight)
                keep_ratio = self.tcs_keep_ratio
            elif self.tcs_variant == "tcs":
                raw_weight = need_hd * trust
                beta = beta_adapt
                keep_ratio = self.tcs_keep_ratio
            else:
                raise ValueError(f"Unknown tcs_variant={self.tcs_variant!r}")

            raw_weight = raw_weight.masked_fill(~token_mask, -1e9)
            selected = self._select_tokens_per_sequence(raw_weight, token_mask, keep_ratio=keep_ratio)

            # For all-token EOPD, keep weights exactly 1 on valid tokens.
            if self.tcs_variant == "eopd":
                weights = token_mask.float()
            else:
                weights = raw_weight.clamp_min(0.0) * selected.float()
                # Per-sequence normalization keeps the loss scale comparable across keep ratios.
                denom = (weights.sum(dim=1, keepdim=True) / selected.float().sum(dim=1, keepdim=True).clamp_min(1.0)).clamp_min(
                    1e-6
                )
                weights = weights / denom
                weights = weights * selected.float()

        token_loss = beta * rkl + (1.0 - beta) * fkl
        token_loss = token_loss * weights * selected.float()
        denom = selected.float().sum().clamp_min(1.0)
        loss = token_loss.sum() / denom

        if self.tcs_log_every_loss:
            self._record_tcs_metrics(
                token_mask=token_mask,
                selected=selected,
                weights=weights,
                u_s=u_s,
                u_t=u_t,
                div=div,
                beta=beta,
                trust=trust,
                loss=loss,
                tail_s=tail_s,
                tail_t=tail_t,
            )

        return loss

    def _select_tokens_per_sequence(self, scores: torch.Tensor, token_mask: torch.Tensor, keep_ratio: float):
        if keep_ratio >= 1.0:
            return token_mask
        selected = torch.zeros_like(token_mask, dtype=torch.bool)
        B = scores.shape[0]
        for i in range(B):
            valid_idx = torch.nonzero(token_mask[i], as_tuple=True)[0]
            n_valid = int(valid_idx.numel())
            if n_valid == 0:
                continue
            k = int(math.ceil(keep_ratio * n_valid))
            k = min(n_valid, max(1, self.tcs_min_tokens_per_seq, k))
            top_local = torch.topk(scores[i, valid_idx], k=k, largest=True).indices
            selected[i, valid_idx[top_local]] = True
        return selected

    def _record_tcs_metrics(self, **kwargs):
        token_mask = kwargs["token_mask"]
        selected = kwargs["selected"]
        valid = token_mask
        if valid.sum() == 0:
            return

        def mean_valid(x):
            return x.detach()[valid].float().mean().item()

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["tcs/selected_ratio"].append(
            (selected.float().sum() / valid.float().sum().clamp_min(1.0)).detach().float().item()
        )
        self._metrics[mode]["tcs/student_uncertainty"].append(mean_valid(kwargs["u_s"]))
        self._metrics[mode]["tcs/teacher_uncertainty"].append(mean_valid(kwargs["u_t"]))
        self._metrics[mode]["tcs/js_div"].append(mean_valid(kwargs["div"]))
        self._metrics[mode]["tcs/beta"].append(mean_valid(kwargs["beta"]))
        self._metrics[mode]["tcs/trust"].append(mean_valid(kwargs["trust"]))
        self._metrics[mode]["tcs/tail_student"].append(mean_valid(kwargs["tail_s"]))
        self._metrics[mode]["tcs/tail_teacher"].append(mean_valid(kwargs["tail_t"]))
        self._metrics[mode]["tcs/custom_loss"].append(kwargs["loss"].detach().float().item())


# -----------------------------
# Main training script
# -----------------------------


def maybe_limit_split(ds, n: int | None):
    if n is None:
        return ds
    n = min(n, len(ds))
    return ds.select(range(n))


def main(script_args: TcsScriptArguments, training_args: TcsDistillationConfig, model_args: ModelConfig):
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    teacher_model_kwargs = dict(
        revision=training_args.teacher_model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        use_cache=True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    if training_args.teacher_model_init_kwargs is not None:
        teacher_model_kwargs.update(training_args.teacher_model_init_kwargs)
    training_args.teacher_model_init_kwargs = teacher_model_kwargs

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = normalize_dataset_to_messages(dataset, script_args)

    train_dataset = dataset[script_args.dataset_train_split]
    train_dataset = maybe_limit_split(train_dataset, script_args.max_train_samples)

    eval_dataset = None
    if training_args.eval_strategy != "no":
        if script_args.dataset_test_split in dataset:
            eval_dataset = dataset[script_args.dataset_test_split]
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
        elif "dev" in dataset:
            eval_dataset = dataset["dev"]
        if eval_dataset is not None:
            eval_dataset = maybe_limit_split(eval_dataset, script_args.max_eval_samples)

    trainer = TcsDistillationTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
            top_p=training_args.top_p,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser():
    return TrlParser((TcsScriptArguments, TcsDistillationConfig, ModelConfig))


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    main(script_args, training_args, model_args)
