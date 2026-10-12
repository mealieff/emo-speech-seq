#!/usr/bin/env python3
"""Utterance LoRA SFT (discrete SER: one 8-way label per utterance) for Qwen2-Audio,
Audio-Flamingo-3 or Qwen3-Omni.

Hyperparameters come from ``configs/finetune/utterance_lora.json``: LoRA r16 / alpha 32 /
dropout 0.05 on every q_proj / k_proj / v_proj / o_proj (``exclude_modules=["audio_tower"]``
does not reach the audio encoder's own q/k/v, so they carry LoRA too; its base weights are
frozen), no connector training, batch 1 x 8 accumulation, lr 2e-4, 3 epochs, bf16. There is
no in-training evaluation; the adapter after the last step is saved to ``--output-dir``
together with the prompt and the config.

Qwen2-Audio is trained on ``<|audio_bos|><|AUDIO|><|audio_eos|>{instruction}\\nOutput:`` with
the label appended; Audio-Flamingo-3 and Qwen3-Omni on their chat templates (user turn =
audio + instruction, assistant turn = label). Loss is on the label tokens only.

Qwen3-Omni trains in bf16 on one 80 GB GPU (59 GiB of thinker weights); ``--qlora`` does not
shrink its fused MoE experts and is refused.

    python scripts/finetune/utterance_lora.py --model qwen2_audio --limit 16 --epochs 0.01   # smoke test
    python scripts/finetune/utterance_lora.py --model qwen2_audio
    python scripts/finetune/utterance_lora.py --model audio_flamingo3
    python scripts/finetune/utterance_lora.py --model qwen3_omni
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emoseq import config

import librosa
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration, Trainer, TrainingArguments

from emoseq.io import load_jsonl
from emoseq.labels import IGNORE_INDEX
from emoseq.models.adapters import get_adapter, resolve_snapshot
from emoseq.scoring.utterance import EMOTION_NAMES

QWEN2_PROMPT_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>{instruction}\nOutput:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["qwen2_audio", "audio_flamingo3", "qwen3_omni"])
    p.add_argument("--train-jsonl", default=str(config.utterance_jsonl("train")))
    p.add_argument("--output-dir", default=None, help="default: $EMOSEQ_RUNS/sft/utterance/<model>")
    p.add_argument("--training-config", default=str(config.REPO / "configs" / "finetune" / "utterance_lora.json"))
    p.add_argument("--prompt", default=str(config.UTTERANCE_PROMPT))
    p.add_argument("--base-model", default=None)
    p.add_argument("--epochs", type=float, default=None, help="override num_train_epochs (smoke tests)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--qlora", action="store_true", help="4-bit NF4 base weights (not used for reported runs)")
    p.add_argument("--device-map", default=None, choices=["auto"],
                   help="auto = shard bf16 weights across all visible GPUs (Qwen3-Omni).")
    p.add_argument("--per-device-batch-size", type=int, default=None,
                   help="override per_device_train_batch_size (keep batch x --grad-accum = 8)")
    p.add_argument("--grad-accum", type=int, default=None, help="override gradient_accumulation_steps")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--group-by-length", action="store_true",
                   help="batch clips of similar duration together (less padding at batch > 1)")
    p.add_argument("--resume", action="store_true",
                   help="continue from the latest checkpoint-* in the output directory (optimizer, scheduler, RNG)")
    args = p.parse_args()
    if args.qlora and args.model == "qwen3_omni":
        p.error("--qlora leaves Qwen3-Omni's fused MoE experts in bf16; use --device-map auto")
    if args.qlora and args.device_map:
        p.error("--qlora and --device-map are exclusive")
    return args


def target_string(label: int) -> str:
    return EMOTION_NAMES[int(label)]


def apply_freeze(model: torch.nn.Module, freeze: dict[str, bool], freeze_prefixes: dict[str, str]) -> None:
    for key, should_freeze in freeze.items():
        if key not in freeze_prefixes:
            raise KeyError(f"unknown freeze key {key!r}; known: {sorted(freeze_prefixes)}")
        if not should_freeze:
            continue
        prefix = freeze_prefixes[key]
        matched = 0
        for name, param in model.named_parameters():
            if name.startswith(prefix):
                param.requires_grad = False
                matched += 1
        if matched == 0:
            raise KeyError(f"freeze key {key!r} -> {prefix!r} matched no parameters")


def qwen2_collate(processor, instruction: str):
    sr = processor.feature_extractor.sampling_rate

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        audios = [librosa.load(item["audio_path"], sr=sr)[0] for item in batch]
        prompts = [QWEN2_PROMPT_TEMPLATE.format(instruction=instruction)] * len(batch)
        targets = [target_string(item["label"]) for item in batch]
        tok = processor.tokenizer
        prev_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"
        try:
            inputs = processor(text=prompts, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        finally:
            tok.padding_side = prev_side
        target_enc = tok(targets, return_tensors="pt", padding=True, add_special_tokens=False, padding_side="right")
        target_ids = target_enc["input_ids"]
        target_mask = target_enc["attention_mask"]
        prompt_len = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], target_ids], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], target_mask], dim=1)
        labels = torch.full_like(inputs["input_ids"], IGNORE_INDEX)
        tl = target_ids.clone()
        tl[target_mask == 0] = IGNORE_INDEX
        labels[:, prompt_len:] = tl
        inputs["labels"] = labels
        return inputs

    return collate_fn


def af3_collate(processor, instruction: str):
    def conversation(audio_path: str, answer: str) -> list[dict]:
        return [
            {"role": "user", "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": instruction},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        conversations = [conversation(item["audio_path"], target_string(item["label"])) for item in batch]
        prompts_only = [c[:-1] for c in conversations]
        full = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=False, return_dict=True, output_labels=True,
        )
        masked = processor.apply_chat_template(prompts_only, tokenize=True, add_generation_prompt=False, return_dict=True)
        labels = full["labels"].clone()
        pad_id = processor.tokenizer.pad_token_id
        prompt_lengths = (masked["input_ids"] != pad_id).sum(dim=1)
        starts = (full["input_ids"] != pad_id).to(torch.int8).argmax(dim=1)
        assistant_starts = starts + prompt_lengths - 1
        positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
        labels[positions < assistant_starts.unsqueeze(1)] = IGNORE_INDEX
        full["labels"] = labels
        return full

    return collate_fn


def qwen3_omni_collate(processor, instruction: str):
    """Chat template rendered to text, then processor(text, audio); same prompt as
    ``scripts/inference/utterance.py`` decodes Qwen3-Omni with."""
    sr = processor.feature_extractor.sampling_rate

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        audios = [librosa.load(item["audio_path"], sr=sr)[0] for item in batch]
        user = {"role": "user", "content": [
            {"type": "audio", "audio_url": "utterance"},
            {"type": "text", "text": instruction},
        ]}
        conversations = [[user, {"role": "assistant", "content": target_string(item["label"])}] for item in batch]
        full_text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template([[user]] * len(batch), tokenize=False, add_generation_prompt=True)
        full = processor(text=full_text, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        prompt = processor(text=prompt_text, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
        pad_id = processor.tokenizer.pad_token_id
        prompt_lengths = (prompt["input_ids"] != pad_id).sum(dim=1)
        starts = (full["input_ids"] != pad_id).to(torch.int8).argmax(dim=1)
        positions = torch.arange(full["input_ids"].size(1)).unsqueeze(0)
        labels = full["input_ids"].clone()
        labels[positions < (starts + prompt_lengths).unsqueeze(1)] = IGNORE_INDEX
        labels[full["attention_mask"] == 0] = IGNORE_INDEX
        full["labels"] = labels
        return full

    return collate_fn


def main() -> None:
    args = parse_args()
    tcfg = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    tcfg["model_key"] = args.model
    t = tcfg["training"]
    if args.epochs is not None:
        t["num_train_epochs"] = args.epochs
    if args.per_device_batch_size is not None:
        t["per_device_train_batch_size"] = args.per_device_batch_size
    if args.grad_accum is not None:
        t["gradient_accumulation_steps"] = args.grad_accum
    if args.no_gradient_checkpointing:
        t["gradient_checkpointing"] = False
    t["group_by_length"] = args.group_by_length
    instruction = Path(args.prompt).read_text(encoding="utf-8").strip()
    output_dir = Path(args.output_dir or (config.RUNS_DIR / "sft" / "utterance" / args.model))
    records = load_jsonl(args.train_jsonl, args.limit)
    if not records:
        raise SystemExit(f"empty training file: {args.train_jsonl}")
    if args.group_by_length:
        import soundfile as sf

        for r in records:
            r["duration"] = sf.info(r["audio_path"]).duration
    adapter = get_adapter(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = args.base_model or resolve_snapshot(args.model)
    print(f"utterance SFT | {args.model} | n={len(records)}\nbase={base_model}\nout={output_dir}")

    quant_cfg = None
    if args.qlora:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    device_map = {"": 0} if args.qlora and torch.cuda.is_available() else args.device_map
    if args.model == "qwen2_audio":
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True,
            quantization_config=quant_cfg, device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        collate_fn = qwen2_collate(processor, instruction)
    else:
        processor = adapter.load_processor(base_model)
        model = adapter.load_model(base_model, dtype, quantization_config=quant_cfg, device_map=device_map)
        collate = qwen3_omni_collate if args.model == "qwen3_omni" else af3_collate
        collate_fn = collate(processor, instruction)

    apply_freeze(model, tcfg.get("freeze", {}), adapter.freeze_prefixes)
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=t.get("gradient_checkpointing", False))
    model = get_peft_model(model, LoraConfig(**tcfg["lora"]))
    if t.get("gradient_checkpointing"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        bf16=t.get("bf16", False) and torch.cuda.is_available(),
        fp16=t.get("fp16", False),
        logging_steps=t.get("logging_steps", 10),
        save_steps=t.get("save_steps", 500),
        save_total_limit=t.get("save_total_limit", 2),
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        remove_unused_columns=False,
        report_to=[],
        **({"train_sampling_strategy": "group_by_length", "length_column_name": "duration"}
           if args.group_by_length else {}),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset.from_list(records), data_collator=collate_fn)
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "prompt.txt").write_text(instruction + "\n", encoding="utf-8")
    (output_dir / "ser_training.json").write_text(json.dumps(tcfg, indent=2), encoding="utf-8")
    print(f"Saved -> {output_dir}")


if __name__ == "__main__":
    main()
