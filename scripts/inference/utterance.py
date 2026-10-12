#!/usr/bin/env python3
"""Utterance-level inference (discrete SER) for the Hugging Face ALMs (Qwen2-Audio,
Audio-Flamingo-3, Qwen3-Omni).

One 8-way label per utterance. Greedy decoding, 16 new tokens. The label is the first
canonical emotion word in the output (``first_n``); an output with none counts as a parse
failure and is scored as Neutral. Strict-format failures are reported alongside.

Prompt modes:
  * ``--base-only``: the model's chat template (user turn = audio + instruction).
  * ``--model-dir`` (LoRA adapter, merged before decoding): ``--sft-prompt-mode``, default
    per model. Qwen2-Audio and Audio-Flamingo-3 use the plain-text
    ``{audio}{instruction}\\nOutput:`` prompt (the reported runs). That matches Qwen2-Audio's
    training collate; Audio-Flamingo-3 was trained on its chat template, so its reported
    fine-tuned numbers use a prompt layout different from training. Qwen3-Omni uses the chat
    template it is trained on.

Writes ``eval_<split>.json`` (metrics) and ``predictions_<split>.json`` (per-utterance
records) to ``--out-dir``.

    python scripts/inference/utterance.py --model qwen2_audio --base-only \\
        --out-dir runs/test1/utterance/qwen2_audio_pt
    python scripts/inference/utterance.py --model audio_flamingo3 \\
        --model-dir runs/sft/utterance/audio_flamingo3 --out-dir runs/test1/utterance/audio_flamingo3_sft
    python scripts/inference/utterance.py --model qwen3_omni --base-only --out-dir runs/test1/utterance/qwen3_omni_pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from emoseq import config

import librosa
import torch

from emoseq.io import load_jsonl
from emoseq.models.adapters import get_adapter, resolve_snapshot
from emoseq.scoring.utterance import (
    EMOTION_NAMES,
    classify_task_a_output,
    parse_task_a,
    parse_task_a_first_n,
    task_a_metrics,
)

SFT_PROMPT_MODE = {
    "qwen2_audio": "sft_output_suffix",
    "audio_flamingo3": "sft_output_suffix",
    "qwen3_omni": "chat_template",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["qwen2_audio", "audio_flamingo3", "qwen3_omni"])
    p.add_argument("--model-dir", default=None, help="LoRA adapter dir or merged checkpoint (not used with --base-only).")
    p.add_argument("--base-only", action="store_true", help="Base weights, no fine-tuned adapter.")
    p.add_argument("--eval-jsonl", default=str(config.utterance_jsonl("test1")))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-model", default=None, help="Base snapshot; resolved from MODEL_CACHE if omitted.")
    p.add_argument("--prompt", default=None,
                   help="Prompt file (default: <model-dir>/prompt.txt if present, else configs/prompts/utterance.txt).")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("EVAL_BATCH_SIZE", "16")),
                   help="Examples per generate() call (left-padded, greedy). Env: EVAL_BATCH_SIZE.")
    p.add_argument("--parse-mode", choices=["first_n", "strict"], default="first_n",
                   help="first_n (default): first canonical label in the output, which tolerates glued "
                        "repetitions like 'NeutralNeutral'. strict: whole output must be one label.")
    p.add_argument("--sft-prompt-mode", choices=["sft_output_suffix", "chat_template"], default=None,
                   help="Prompt for --model-dir decoding (default: sft_output_suffix for qwen2_audio and "
                        "audio_flamingo3, chat_template for qwen3_omni).")
    p.add_argument("--device", default=None)
    p.add_argument("--device-map", default=None, choices=["auto"],
                   help="auto = shard bf16 weights across all visible GPUs (Qwen3-Omni on GPUs < 80 GB).")
    return p.parse_args()


def load_prompt(args: argparse.Namespace, model_dir: Path | None) -> str:
    candidates = [Path(args.prompt)] if args.prompt else []
    if model_dir is not None:
        candidates.append(model_dir / "prompt.txt")
    candidates.append(config.UTTERANCE_PROMPT)
    for c in candidates:
        if c.is_file() and c.read_text(encoding="utf-8").strip():
            return c.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"no prompt among {candidates}")


def load_model(adapter, base_model: str, model_dir: Path | None, device: torch.device, *, base_only: bool = False,
               device_map: str | None = None):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if base_only:
        model = adapter.load_model(base_model, dtype, device_map=device_map)
    elif model_dir is None:
        raise ValueError("--model-dir is required unless --base-only is set.")
    elif (model_dir / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(adapter.load_model(base_model, dtype, device_map=device_map), str(model_dir))
        model = model.merge_and_unload()
    else:
        model = adapter.load_model(str(model_dir), dtype, device_map=device_map)
    return (model if device_map else model.to(device)).eval()


def build_batch_inputs(processor, adapter, instructions: list[str], audio_paths: list[str],
                       sr: int, *, chat_template: bool = False) -> tuple[Any, str]:
    """Tokenize a batch of (instruction, audio) pairs with left padding so
    ``gen[:, input_len:]`` slices the completions."""
    audios = [librosa.load(p, sr=sr)[0] for p in audio_paths]
    if chat_template:
        texts = []
        for instruction, path in zip(instructions, audio_paths):
            conversation = [{
                "role": "user",
                "content": [{"type": "audio", "audio_url": path}, {"type": "text", "text": instruction}],
            }]
            texts.append(processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
        prompt_mode = "chat_template"
    else:
        texts = [adapter.format_prompt(ins) for ins in instructions]
        prompt_mode = "sft_output_suffix"
    tok = getattr(processor, "tokenizer", None)
    prev_side = getattr(tok, "padding_side", None)
    if tok is not None:
        tok.padding_side = "left"
    try:
        inputs = adapter.process(processor, texts, audios, sr)
    finally:
        if tok is not None and prev_side is not None:
            tok.padding_side = prev_side
    return inputs, prompt_mode


def generate_batch(model, processor, adapter, instructions: list[str], audio_paths: list[str],
                   sr: int, max_new_tokens: int, device: torch.device,
                   *, chat_template: bool = False) -> tuple[list[str], str]:
    inputs, prompt_mode = build_batch_inputs(processor, adapter, instructions, audio_paths, sr,
                                             chat_template=chat_template)
    model_dtype = next(model.parameters()).dtype
    casted = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            casted[k] = v.to(device=device, dtype=model_dtype) if v.is_floating_point() else v.to(device)
        else:
            casted[k] = v
    with torch.no_grad():
        gen = model.generate(**casted, max_new_tokens=max_new_tokens, do_sample=False)
    out_ids = gen[:, casted["input_ids"].shape[1]:]
    return processor.batch_decode(out_ids, skip_special_tokens=True), prompt_mode


def batched(items: list, size: int):
    for i in range(0, len(items), max(1, size)):
        yield i, items[i:i + max(1, size)]


def main() -> None:
    args = parse_args()
    adapter = get_adapter(args.model)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_model = args.base_model or resolve_snapshot(args.model)
    model_dir = Path(args.model_dir) if args.model_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"utterance | model {args.model} | device {device}")
    print(f"Base model : {base_model}")
    print(f"Checkpoint : {'(base only)' if args.base_only else model_dir}")

    instruction = load_prompt(args, model_dir)
    processor = adapter.load_processor(base_model)
    model = load_model(adapter, base_model, model_dir, device, base_only=args.base_only, device_map=args.device_map)
    if args.device_map:
        device = model.device
    sr = adapter.sampling_rate(processor)
    if args.base_only:
        chat_template = adapter.use_chat_template_for_base_inference
    else:
        chat_template = (args.sft_prompt_mode or SFT_PROMPT_MODE[args.model]) == "chat_template"

    records = load_jsonl(args.eval_jsonl, args.limit)
    split = Path(args.eval_jsonl).stem.replace("msp_podcast_", "")
    print(f"Eval records: {len(records)} | batch {args.batch_size} | parse {args.parse_mode} | "
          f"max_new_tokens {args.max_new_tokens}")
    parse_a = parse_task_a_first_n if args.parse_mode == "first_n" else parse_task_a

    pred_records: list[dict] = []
    prompt_mode: str | None = None
    y_true: list[int] = []
    y_pred: list[int] = []
    y_pred_strict: list[int] = []
    n_fail = n_strict_fail = n_done = 0
    for start, chunk in batched(records, args.batch_size):
        texts, prompt_mode = generate_batch(
            model, processor, adapter, [instruction] * len(chunk),
            [it["audio_path"] for it in chunk], sr, args.max_new_tokens, device, chat_template=chat_template,
        )
        for j, (item, text) in enumerate(zip(chunk, texts)):
            gt = int(item["label"])
            pred = parse_a(text)
            parsed_primary = pred
            cls = classify_task_a_output(text, gt)
            if cls["bucket"] != "parsed_ok":
                n_strict_fail += 1
            y_pred_strict.append(7 if cls["parsed"] is None else cls["parsed"])
            if pred is None:
                n_fail += 1
                pred = 7  # Neutral, counted as a parse failure
            y_true.append(gt)
            y_pred.append(pred)
            pred_records.append({
                "idx": start + j,
                "audio_path": item.get("audio_path"),
                "label": gt,
                "label_name": EMOTION_NAMES[gt],
                "pred_for_metrics": pred,
                "pred_name": EMOTION_NAMES[pred],
                f"{args.parse_mode}_parsed": parsed_primary,
                "fallback_neutral": parsed_primary is None,
                **cls,
            })
        n_done += len(chunk)
        if n_done // 50 != (n_done - len(chunk)) // 50 or n_done == len(records):
            print(f"  [{n_done}/{len(records)}]", flush=True)

    metrics = task_a_metrics(y_true, y_pred)
    metrics["parse_mode"] = args.parse_mode
    metrics["parse_failures"] = n_fail
    metrics["parse_failure_rate"] = round(n_fail / len(records), 4) if records else 0.0
    metrics["strict_parse_failures"] = n_strict_fail
    metrics["strict_parse_failure_rate"] = round(n_strict_fail / len(records), 4) if records else 0.0
    strict_m = task_a_metrics(y_true, y_pred_strict)
    metrics["strict"] = {k: strict_m.get(k) for k in ("WA", "UA", "macro_F1")}

    header = {
        "task": "utt",
        "model": args.model,
        "base_only": args.base_only,
        "inference_prompt_mode": prompt_mode,
        "model_dir": str(model_dir) if model_dir else None,
        "base_model": base_model,
        "eval_jsonl": args.eval_jsonl,
        "split": split,
        "batch_size": args.batch_size,
    }
    pred_path = out_dir / f"predictions_{split}.json"
    pred_path.write_text(json.dumps({**header, "n": len(pred_records), "records": pred_records}, indent=2),
                         encoding="utf-8")
    metrics["predictions_file"] = str(pred_path)
    out_path = out_dir / f"eval_{split}.json"
    out_path.write_text(json.dumps({**header, "metrics": metrics}, indent=2), encoding="utf-8")

    print("\n" + "=" * 48)
    print(f"utterance results ({split})")
    print(f"  WA       : {metrics.get('WA')}")
    print(f"  UA       : {metrics.get('UA')}  (primary)")
    print(f"  macro-F1 : {metrics.get('macro_F1')}")
    print(f"  parse fail rate: {metrics.get('parse_failure_rate')} ({args.parse_mode}; "
          f"strict {metrics.get('strict_parse_failure_rate')})")
    print("=" * 48)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
