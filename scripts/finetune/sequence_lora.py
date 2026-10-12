#!/usr/bin/env python3
"""Sequence LoRA SFT for Qwen2-Audio, Audio-Flamingo-3 or Qwen3-Omni on the concatenated-sequence JSONL.

Train and eval use ``configs/prompts/sequence.txt``; the gold ``(start, end, emotion)``
lines are only in the assistant target. The full recipe (training, full-Dev checkpoint
selection) is ``scripts/finetune/run_sequence_sft.sh``.

LoRA r16 / alpha 32 / dropout 0.05 on every module named q_proj / k_proj / v_proj / o_proj.
``exclude_modules=["audio_tower"]`` only excludes a module named exactly ``audio_tower``, so
the audio encoder's q/k/v projections carry LoRA too (its base weights stay frozen); the
reported Qwen2-Audio and AF3 adapters were trained this way. The audio-to-LLM connector is
trained in full: ``multi_modal_projector`` (Qwen2-Audio, AF3) or ``audio_tower.proj1/proj2``
(Qwen3-Omni, which has no separate projector module).

Qwen3-Omni (30B MoE thinker, 59 GiB of bf16 weights) is trained in bf16 on one 80 GB GPU:
peak 67 GiB at 180 s of audio and 71 GiB at 300 s; 600 s does not fit. For longer sequences use
``--device-map auto`` over two GPUs. 4-bit loading does not help here: transformers 5.x stores
the MoE experts as fused 3-D tensors, which bitsandbytes leaves unquantized.

    python scripts/finetune/sequence_lora.py --model qwen2_audio --limit 8 --epochs 0.05   # smoke test
    python scripts/finetune/sequence_lora.py --model qwen2_audio --length-bin-weight --save-total-limit 0
    python scripts/finetune/sequence_lora.py --model audio_flamingo3 --length-bin-weight --save-total-limit 0 \\
        --train-jsonl data/sequences/train_af3.jsonl --eval-jsonl data/sequences/dev_af3.jsonl
    python scripts/finetune/sequence_lora.py --model qwen3_omni --device-map auto --length-bin-weight --save-total-limit 0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from emoseq import config

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, Trainer, TrainingArguments

from emoseq.labels import IGNORE_INDEX
from emoseq.models.adapters import get_adapter, resolve_snapshot
from emoseq.provenance import packages
from emoseq.records import load_jsonl
from emoseq.span_token_weights import even_mass_for_labels
from emoseq.targets import target_string_from_record

CONNECTOR_MODULES = {
    "qwen2_audio": ["multi_modal_projector"],
    "audio_flamingo3": ["multi_modal_projector"],
    "qwen3_omni": ["proj1", "proj2"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen2_audio", choices=["qwen2_audio", "audio_flamingo3", "qwen3_omni"])
    p.add_argument("--train-jsonl", default=str(config.sequence_jsonl("train")))
    p.add_argument("--eval-jsonl", default=str(config.sequence_jsonl("dev")))
    p.add_argument("--output-dir", default=None, help="default: $EMOSEQ_RUNS/sft/sequence/<model>")
    p.add_argument("--base-model", default=None)
    p.add_argument("--model-cache", default=None)
    p.add_argument("--prompt", default=str(config.SEQUENCE_PROMPT))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--eval-limit", type=int, default=256)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Max epoch checkpoints to keep. 0 keeps every epoch.",
    )
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--checkpoint-above-sec", type=float, default=None,
                   help="Gradient checkpointing only for micro-batches whose padded audio (batch size x longest "
                        "clip) exceeds this many seconds; default: every step (Qwen2-Audio / Qwen3-Omni inputs).")
    p.add_argument("--group-by-length", action="store_true",
                   help="Draw each optimizer step's sequences from similar audio durations (less padding at batch > 1).")
    p.add_argument("--dataloader-workers", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (timing runs)")
    p.add_argument("--device-map", default=None, choices=["auto"],
                   help="auto = shard bf16 weights across all visible GPUs (Qwen3-Omni).")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument(
        "--length-bin-weight",
        action="store_true",
        help="Scale per-example CE by sklearn-balanced inverse bin frequency (bin_weight on each row).",
    )
    p.add_argument(
        "--prompt-layout",
        default="user",
        choices=["user", "system"],
        help="user = instruction in the user turn (used for all reported runs). "
             "system = instruction in the system role, user audio-only.",
    )
    p.add_argument(
        "--even-span-weights",
        action="store_true",
        help="Train CE: 0.5 mass on time tokens and 0.5 on emotion tokens per example. "
             "Punctuation mass is --even-span-punct-floor (default 0). eval_loss stays uniform token-mean.",
    )
    p.add_argument(
        "--even-span-punct-floor",
        type=float,
        default=0.0,
        help="With --even-span-weights: extra mass shared equally by punctuation "
             "tokens so format is not zeroed. 0 is the collapsed Qwen2 mask; "
             "0.1 keeps time:emotion 1:1 and gives format ~9%% of the example.",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to a checkpoint dir, or 'true' to pick the latest complete one.",
    )
    p.add_argument(
        "--eval-only-checkpoint",
        default=None,
        help="Load this LoRA checkpoint, run the Trainer eval (same unweighted eval_loss as "
             "during training) on --eval-jsonl/--eval-limit, write --eval-output, and exit.",
    )
    p.add_argument("--eval-output", default=None, help="JSON path for --eval-only-checkpoint results.")
    return p.parse_args()


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def lora_config(model_key: str) -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        exclude_modules=["audio_tower"],
        modules_to_save=CONNECTOR_MODULES[model_key],
        bias="none",
        task_type="CAUSAL_LM",
    )


def freeze_audio_tower(model) -> None:
    n = 0
    for name, param in model.named_parameters():
        if "audio_tower" in name:
            param.requires_grad = False
            n += 1
    if n == 0:
        raise RuntimeError("audio_tower freeze matched no parameters")


def _sft_turns(model_key: str, instruction: str, audio_path: str, answer: str, layout: str) -> list[dict]:
    if model_key == "audio_flamingo3":
        audio = {"type": "audio", "path": audio_path}
        system = {"role": "system", "content": [{"type": "text", "text": instruction}]}
        assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    else:
        audio = {"type": "audio", "audio_url": "sequence"}
        system = {"role": "system", "content": instruction}
        assistant = {"role": "assistant", "content": answer}
    if layout == "system":
        return [system, {"role": "user", "content": [audio]}, assistant]
    if layout != "user":
        raise ValueError(f"unknown prompt_layout {layout!r}")
    return [
        {"role": "user", "content": [audio, {"type": "text", "text": instruction}]},
        assistant,
    ]


def make_qwen2_collate(
    processor,
    instruction: str,
    sr: int,
    prompt_layout: str = "user",
    even_span_weights: bool = False,
    even_span_punct_floor: float = 0.0,
    model_key: str = "qwen2_audio",
):
    """Chat template rendered to text, then processor(text, audio). Used for Qwen2-Audio and Qwen3-Omni."""
    import librosa

    def collate_fn(batch: list[dict]) -> dict:
        conversations, prompts_only, audios, answers = [], [], [], []
        for row in batch:
            audio, _ = librosa.load(row["audio_path"], sr=sr)
            answer = target_string_from_record(row)
            turns = _sft_turns(model_key, instruction, "sequence", answer, prompt_layout)
            conversations.append(turns)
            prompts_only.append(turns[:-1])
            audios.append(audio)
            answers.append(answer)

        full_text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(prompts_only, tokenize=False, add_generation_prompt=True)
        full = processor(text=full_text, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        prompt = processor(text=prompt_text, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        labels = full["input_ids"].clone()
        pad_id = processor.tokenizer.pad_token_id
        prompt_lengths = (prompt["input_ids"] != pad_id).sum(dim=1)
        starts = (full["input_ids"] != pad_id).to(torch.int8).argmax(dim=1)
        assistant_starts = starts + prompt_lengths - 1
        positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
        labels[positions < assistant_starts.unsqueeze(1)] = IGNORE_INDEX
        labels[labels == pad_id] = IGNORE_INDEX
        full["labels"] = labels
        return _with_sample_weight(
            full, batch, processor.tokenizer, answers, even_span_weights, even_span_punct_floor
        )

    return collate_fn


def make_af3_collate(
    processor,
    instruction: str,
    prompt_layout: str = "user",
    even_span_weights: bool = False,
    even_span_punct_floor: float = 0.0,
):
    def collate_fn(batch: list[dict]) -> dict:
        conversations, prompts_only, answers = [], [], []
        for row in batch:
            answer = target_string_from_record(row)
            conv = _sft_turns("audio_flamingo3", instruction, row["audio_path"], answer, prompt_layout)
            conversations.append(conv)
            prompts_only.append(conv[:-1])
            answers.append(answer)
        full = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=False, return_dict=True, output_labels=True
        )
        masked = processor.apply_chat_template(
            prompts_only, tokenize=True, add_generation_prompt=False, return_dict=True
        )
        labels = full["labels"].clone()
        pad_id = processor.tokenizer.pad_token_id
        prompt_lengths = (masked["input_ids"] != pad_id).sum(dim=1)
        starts = (full["input_ids"] != pad_id).to(torch.int8).argmax(dim=1)
        assistant_starts = starts + prompt_lengths - 1
        positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
        labels[positions < assistant_starts.unsqueeze(1)] = IGNORE_INDEX
        full["labels"] = labels
        return _with_sample_weight(
            full, batch, processor.tokenizer, answers, even_span_weights, even_span_punct_floor
        )

    return collate_fn


def _with_sample_weight(
    full: dict,
    batch: list[dict],
    tokenizer=None,
    answers: list[str] | None = None,
    even_span_weights: bool = False,
    even_span_punct_floor: float = 0.0,
) -> dict:
    full["sample_weight"] = torch.tensor(
        [float(row.get("bin_weight") or 1.0) for row in batch], dtype=torch.float32
    )
    if even_span_weights and tokenizer is not None and answers is not None:
        rows = []
        for i, answer in enumerate(answers):
            rows.append(
                even_mass_for_labels(
                    tokenizer,
                    full["labels"][i],
                    answer,
                    IGNORE_INDEX,
                    punct_floor=even_span_punct_floor,
                )
            )
        full["token_type_weight"] = torch.stack(rows, dim=0)
    return full


def weighted_token_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    token_type_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum over examples of (token-mean or even-span CE × inverse-frequency bin weight).

    The Trainer does not divide by gradient accumulation for models whose forward takes **kwargs
    (all three here), so a batch 1 × accumulation 8 step is the sum of 8 per-example losses.
    Summing within the micro-batch keeps that step identical for any batch × accumulation = 8.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    bsz, tlen, vocab = shift_logits.shape
    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(bsz, tlen)
    mask = shift_labels.ne(IGNORE_INDEX)
    if token_type_weight is not None:
        tw = token_type_weight[..., 1:].to(device=token_loss.device, dtype=token_loss.dtype) * mask
        denom = tw.sum(dim=1).clamp(min=1e-8)
        per_ex = (token_loss * tw).sum(dim=1) / denom
    else:
        denom = mask.sum(dim=1).clamp(min=1)
        per_ex = (token_loss * mask).sum(dim=1) / denom
    return (per_ex * weights.to(device=per_ex.device, dtype=per_ex.dtype)).sum()


class LengthBinWeightedTrainer(Trainer):
    """Train loss uses bin_weight and optional even time/emotion mass; eval_loss stays unweighted."""

    def __init__(self, *args, apply_bin_weight: bool = False, apply_span_weight: bool = False,
                 checkpoint_above_sec: float | None = None, group_by_length: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_bin_weight = apply_bin_weight
        self.apply_span_weight = apply_span_weight
        self.checkpoint_above_sec = checkpoint_above_sec
        self.group_by_length = group_by_length

    def _get_train_sampler(self, train_dataset=None):
        if not self.group_by_length:
            return super()._get_train_sampler(train_dataset)
        from transformers.trainer_pt_utils import LengthGroupedSampler

        ds = train_dataset if train_dataset is not None else self.train_dataset
        return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                    lengths=[float(r["audio_duration"]) for r in ds.rows])

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Checkpointing only recomputes activations (dropout RNG is replayed), so the gradient is
        # the same either way; it is switched on only where memory needs it.
        if self.checkpoint_above_sec is not None and "feature_attention_mask" in inputs:
            fam = inputs["feature_attention_mask"]
            padded_sec = fam.shape[0] * float(fam.sum(dim=1).max()) / 100.0
            on = padded_sec > self.checkpoint_above_sec
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing") and isinstance(module.gradient_checkpointing, bool):
                    module.gradient_checkpointing = on
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("sample_weight", None)
        token_w = inputs.pop("token_type_weight", None)
        outputs = model(**inputs)
        loss = outputs.loss
        logits = getattr(outputs, "logits", None)
        labels = inputs.get("labels")
        if not model.training or logits is None or labels is None:
            return (loss, outputs) if return_outputs else loss
        use_span = self.apply_span_weight and token_w is not None
        use_bin = self.apply_bin_weight and weights is not None
        if use_span or use_bin:
            if weights is None:
                weights = torch.ones(labels.size(0), device=logits.device, dtype=logits.dtype)
            loss = weighted_token_ce(
                logits, labels, weights, token_type_weight=token_w if use_span else None
            )
        return (loss, outputs) if return_outputs else loss


def save_training_log(log_history: list[dict], out_dir: Path, title: str) -> None:
    (out_dir / "log_history.json").write_text(json.dumps(log_history, indent=2), encoding="utf-8")
    fieldnames = sorted({k for entry in log_history for k in entry})
    with open(out_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_history)
    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss = [e["loss"] for e in log_history if "loss" in e]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_steps:
        ax.plot(train_steps, train_loss, label="train loss", color="tab:blue")
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="eval loss", color="tab:orange", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


def run_eval_only(args, model, processor, collate_fn, eval_rows) -> None:
    from peft import PeftModel

    if not eval_rows:
        raise SystemExit("--eval-only-checkpoint needs a non-empty --eval-jsonl")
    ckpt = Path(args.eval_only_checkpoint)
    model = PeftModel.from_pretrained(model, str(ckpt))
    if torch.cuda.is_available() and getattr(model, "hf_device_map", None) is None:
        model = model.to("cuda")
    model.eval()
    tmp_dir = Path(args.eval_output or (ckpt / "eval_only.json")).parent / "_eval_only_tmp"
    trainer = LengthBinWeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_dir),
            per_device_eval_batch_size=args.eval_batch_size or 1,
            bf16=torch.cuda.is_available(),
            remove_unused_columns=False,
            report_to=[],
        ),
        eval_dataset=JsonlDataset(eval_rows),
        data_collator=collate_fn,
    )
    metrics = trainer.evaluate()
    out = {
        "checkpoint": str(ckpt),
        "model": args.model,
        "eval_jsonl": args.eval_jsonl,
        "n_eval": len(eval_rows),
        "eval_loss": metrics.get("eval_loss"),
        "metrics": metrics,
    }
    out_path = Path(args.eval_output or (ckpt / "eval_only.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"eval_only {ckpt.name} n={len(eval_rows)} eval_loss={out['eval_loss']} -> {out_path}")


def main() -> None:
    args = parse_args()
    instruction = Path(args.prompt).read_text(encoding="utf-8").strip()
    if not instruction:
        raise SystemExit(f"empty prompt: {args.prompt}")
    out_dir = Path(args.output_dir or (config.RUNS_DIR / "sft" / "sequence" / args.model))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_jsonl, args.limit)
    eval_rows = None
    if not args.no_eval and args.eval_jsonl:
        ev_path = Path(args.eval_jsonl)
        if ev_path.is_file():
            eval_rows = load_jsonl(ev_path, args.eval_limit)
        else:
            print(f"warning: eval JSONL missing ({ev_path}); training without dev eval")
    if args.length_bin_weight:
        from emoseq.bins import apply_bin_weights, sklearn_balanced_bin_weights

        wmap = sklearn_balanced_bin_weights(train_rows)
        apply_bin_weights(train_rows, wmap)
        if eval_rows:
            apply_bin_weights(eval_rows, wmap)
        print(f"  length-bin weights from Train: {wmap}")

    print("=" * 60)
    print(f"sequence ALM SFT (Train/Dev/Test1 concat) | {args.model} | train n={len(train_rows)}")
    print(f"  train  {args.train_jsonl}")
    print(f"  eval   {args.eval_jsonl if eval_rows else 'off'}  n={len(eval_rows) if eval_rows else 0}")
    print(f"  out    {out_dir}")
    if args.length_bin_weight:
        from collections import Counter

        ww = Counter(round(float(r.get("bin_weight") or 1.0), 6) for r in train_rows)
        print(f"  bin_weight train CE on  unique_weights={dict(ww)}")
    print(
        f"  prompt_layout={args.prompt_layout}  even_span_weights={args.even_span_weights}  "
        f"even_span_punct_floor={args.even_span_punct_floor}"
    )
    print("=" * 60)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.model == "qwen2_audio":
        base = args.base_model or resolve_snapshot("qwen2_audio", args.model_cache)
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        sr = processor.feature_extractor.sampling_rate
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            base, torch_dtype=dtype, trust_remote_code=True
        )
        collate_fn = make_qwen2_collate(
            processor,
            instruction,
            sr,
            args.prompt_layout,
            args.even_span_weights,
            args.even_span_punct_floor,
        )
    elif args.model == "qwen3_omni":
        adapter = get_adapter("qwen3_omni")
        base = args.base_model or resolve_snapshot("qwen3_omni", args.model_cache)
        processor = adapter.load_processor(base)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model = adapter.load_model(base, dtype, device_map=args.device_map)
        collate_fn = make_qwen2_collate(
            processor,
            instruction,
            adapter.sampling_rate(processor),
            args.prompt_layout,
            args.even_span_weights,
            args.even_span_punct_floor,
            model_key="qwen3_omni",
        )
    else:
        adapter = get_adapter("audio_flamingo3")
        base = args.base_model or resolve_snapshot("audio_flamingo3", args.model_cache)
        processor = adapter.load_processor(base)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model = adapter.load_model(base, dtype)
        collate_fn = make_af3_collate(
            processor,
            instruction,
            args.prompt_layout,
            args.even_span_weights,
            args.even_span_punct_floor,
        )
    if args.device_map and args.model != "qwen3_omni":
        raise SystemExit("--device-map is wired for qwen3_omni only")

    if args.eval_only_checkpoint:
        run_eval_only(args, model, processor, collate_fn, eval_rows)
        return

    freeze_audio_tower(model)
    model = get_peft_model(model, lora_config(args.model))
    if not args.no_gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        # eval_loss is the model's token mean per eval batch: batch 1 keeps it a mean over sequences.
        per_device_eval_batch_size=args.eval_batch_size or 1,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=args.dataloader_workers,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if eval_rows is not None else "no",
        save_strategy="epoch",
        save_total_limit=None if args.save_total_limit == 0 else args.save_total_limit,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=eval_rows is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer_cls = LengthBinWeightedTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=JsonlDataset(train_rows),
        eval_dataset=JsonlDataset(eval_rows) if eval_rows is not None else None,
        data_collator=collate_fn,
        apply_bin_weight=args.length_bin_weight,
        apply_span_weight=args.even_span_weights,
        checkpoint_above_sec=None if args.no_gradient_checkpointing else args.checkpoint_above_sec,
        group_by_length=args.group_by_length,
    )
    resume = args.resume_from_checkpoint
    if resume in {"true", "True", "1"}:
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    # With load_best_model_at_end the in-memory weights are the lowest-eval_loss epoch.
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    trainer.save_state()
    (out_dir / "prompt_train.txt").write_text(instruction + "\n", encoding="utf-8")
    (out_dir / "prompt_layout.txt").write_text(args.prompt_layout + "\n", encoding="utf-8")
    best = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "metric_for_best_model": "eval_loss" if eval_rows is not None else None,
        "greater_is_better": False,
        "load_best_model_at_end": eval_rows is not None,
        "saved_adapter": str(out_dir),
    }
    (out_dir / "best_checkpoint.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    meta = {
        "task": "sequence_sft",
        "model": args.model,
        "base_model": base,
        "train_jsonl": args.train_jsonl,
        "eval_jsonl": args.eval_jsonl if eval_rows is not None else None,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows) if eval_rows is not None else 0,
        "epochs": args.epochs,
        "lr": args.lr,
        "length_bin_weight": args.length_bin_weight,
        "prompt_layout": args.prompt_layout,
        "even_span_weights": args.even_span_weights,
        "even_span_punct_floor": args.even_span_punct_floor,
        "lora": {"r": 16, "alpha": 32, "modules_to_save": CONNECTOR_MODULES[args.model]},
        "device_map": args.device_map,
        "per_device_batch_size": args.per_device_batch_size,
        "grad_accum": args.grad_accum,
        "group_by_length": args.group_by_length,
        "gradient_checkpointing": "off" if args.no_gradient_checkpointing
        else (f"above {args.checkpoint_above_sec} s padded audio" if args.checkpoint_above_sec else "every step"),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "packages": packages(),
        "checkpoint": best,
    }
    (out_dir / "sequence_training.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    save_training_log(trainer.state.log_history, out_dir, f"{args.model} sequence LoRA")
    print(f"saved LoRA adapter → {out_dir}")
    if eval_rows is not None:
        print(
            f"best checkpoint metric=eval_loss value={trainer.state.best_metric}  "
            f"src={trainer.state.best_model_checkpoint}"
        )


if __name__ == "__main__":
    main()
