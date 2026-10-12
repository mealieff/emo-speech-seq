#!/usr/bin/env python3
"""Sequence inference (sequential SER) for the Hugging Face ALMs (Qwen2-Audio, Audio-Flamingo-3,
Qwen3-Omni), pretrained or LoRA.

Writes ``predictions.jsonl`` (resumable), ``metrics.json`` and ``quartile_report.json``
to ``--out-dir``.

    python scripts/inference/sequence.py --model qwen2_audio --out-dir runs/test1/sequence/qwen2_audio_pt
    python scripts/inference/sequence.py --model audio_flamingo3 \\
        --adapter runs/sft/sequence/audio_flamingo3 --out-dir runs/test1/sequence/audio_flamingo3_sft
    python scripts/inference/sequence.py --model qwen3_omni --out-dir runs/test1/sequence/qwen3_omni_pt

Pretrained Qwen2-Audio decodes audio longer than 30 s in 30 s windows (spans are shifted to
sequence time and merged only across window seams). With ``--adapter`` Qwen2-Audio sees only
the first 30 s (encoder limit), as in training. Audio-Flamingo-3 skips audio longer than 496 s
(its 8,192-token context); skipped rows are written but not scored. Qwen3-Omni takes the whole
sequence (its processor does not truncate audio).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from emoseq import config

import torch

from emoseq.bins import bin_of_record, canonicalize_model
from emoseq.io import load_jsonl
from emoseq.models import hf_generate
from emoseq.models.adapters import get_adapter, resolve_snapshot
from emoseq.scoring.report import format_phase2, format_quartile_table, pool_metrics, quartile_report
from emoseq.scoring.sequence import gold_segments, pack_scores, score_sequence, score_text

AF3_INFER_MAX_AUDIO_SEC = 496.0  # longest audio that fits AF3's 8,192-token context with the prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--jsonl", default=str(config.sequence_jsonl("test1")))
    p.add_argument("--adapter", default=None)
    p.add_argument("--prompt", default=str(config.SEQUENCE_PROMPT))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=800,
                   help="Default 800: 20 tokens per output line covers L=40 complete.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("SEQ_EVAL_BATCH_SIZE", "8")),
        help="Sequences per generate() call on the SFT / AF3 path (left-padded, greedy; "
             "batches grouped by audio duration). Env: SEQ_EVAL_BATCH_SIZE. Default 8.",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--chunk-sec", type=float, default=None)
    p.add_argument("--no-chunk", action="store_true")
    p.add_argument(
        "--sft-chunk",
        action="store_true",
        help="Qwen2-Audio + --adapter only: clips longer than --chunk-sec (default 30 s, the encoder "
             "window) are decoded per window with the SFT template and spans are stitched; shorter "
             "clips are unchanged.",
    )
    p.add_argument("--qlora", action="store_true", help="4-bit NF4 base weights (not used for reported runs).")
    p.add_argument("--device-map", default=None, choices=["auto"],
                   help="auto = shard bf16 weights across all visible GPUs (Qwen3-Omni on GPUs < 80 GB).")
    p.add_argument("--tag", default=None, help="Label written into metrics.json (pretrained vs sft).")
    p.add_argument(
        "--prompt-layout",
        default="user",
        choices=["user", "system"],
        help="user = prompt folded into the user turn (used for all reported runs). "
             "system = prompt in the system role, user turn audio-only. Ablation only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_key = canonicalize_model(args.model)
    instruction = Path(args.prompt).read_text(encoding="utf-8").strip()
    if not instruction:
        raise SystemExit(f"empty prompt: {args.prompt}")
    records = load_jsonl(args.jsonl, args.limit)
    if not records:
        raise SystemExit(f"empty jsonl {args.jsonl}")
    missing_audio = [r["seq_id"] for r in records if not r.get("audio_path") or not Path(r["audio_path"]).is_file()]
    if missing_audio:
        raise SystemExit(
            f"{len(missing_audio)} sequences missing concat wav, e.g. {missing_audio[:3]}; "
            "run python scripts/data/export_sequences.py"
        )

    if args.qlora and model_key == "qwen3_omni":
        raise SystemExit("--qlora leaves Qwen3-Omni's fused MoE experts in bf16; use --device-map auto")
    if args.qlora and args.device_map:
        raise SystemExit("--qlora and --device-map are exclusive")
    adapter = get_adapter(model_key)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base = resolve_snapshot(model_key)
    adapter_dir = Path(args.adapter) if args.adapter else None
    use_qlora = args.qlora
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    quant = hf_generate.bitsandbytes_config(dtype) if use_qlora else None
    processor = adapter.load_processor(base)
    if processor is not None and getattr(processor, "tokenizer", None) is not None:
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = adapter.load_model(
        base,
        dtype,
        quantization_config=quant,
        device_map={"": 0} if use_qlora and device.type == "cuda" else args.device_map,
    )
    if adapter_dir:
        from peft import PeftModel

        if not (adapter_dir / "adapter_config.json").is_file():
            raise SystemExit(f"no LoRA adapter at {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    if args.device_map:
        device = model.device
    elif not use_qlora:
        model = model.to(device)
    model.eval()
    sr = adapter.sampling_rate(processor)
    chunk_sec = args.chunk_sec if args.chunk_sec is not None else hf_generate.QWEN2_MAX_SEC
    chunk_qwen2 = model_key == "qwen2_audio" and not args.no_chunk and not adapter_dir
    if args.sft_chunk and not (model_key == "qwen2_audio" and adapter_dir):
        raise SystemExit("--sft-chunk applies to qwen2_audio with --adapter only")
    out_dir = Path(args.out_dir or (config.RUNS_DIR / "inference" / model_key))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "predictions.jsonl"
    if not args.resume and jsonl.is_file():
        jsonl.unlink()
    done = hf_generate.load_done(jsonl) if args.resume else set()
    todo = [r for r in records if r["seq_id"] not in done]
    tag = args.tag or ("sft" if adapter_dir else "pretrained")
    layout = args.prompt_layout
    # Pretrained AF3 / Qwen3-Omni get the same chat-template prompt through the batched SFT path.
    sft_path = bool(adapter_dir) or model_key in {"audio_flamingo3", "qwen3_omni"}
    batch_size = max(1, args.batch_size) if sft_path else 1
    if batch_size > 1:
        todo.sort(key=lambda r: float(r.get("audio_duration") or 0))
    print(
        f"sequence {tag} | {model_key} | layout={layout} | {len(done)} done, {len(todo)} remaining | "
        f"chunk={chunk_qwen2} | batch={batch_size} | max_new_tokens={args.max_new_tokens}"
    )

    def is_over_limit(rec: dict) -> bool:
        return model_key == "audio_flamingo3" and float(rec.get("audio_duration") or 0) > AF3_INFER_MAX_AUDIO_SEC

    def is_sft_chunked(rec: dict) -> bool:
        return args.sft_chunk and float(rec.get("audio_duration") or 0) > chunk_sec

    batch_texts: dict[str, str] = {}
    with jsonl.open("a", encoding="utf-8") as fout:
        for i, rec in enumerate(todo, start=1):
            if (sft_path and rec["seq_id"] not in batch_texts and not is_over_limit(rec)
                    and not is_sft_chunked(rec)):
                batch = [r for r in todo[i - 1 : i - 1 + batch_size]
                         if not is_over_limit(r) and not is_sft_chunked(r)]
                texts = hf_generate.generate_sft_batch(
                    model, processor, model_key, instruction, [r["audio_path"] for r in batch], sr,
                    args.max_new_tokens, device, prompt_layout=layout,
                )
                batch_texts = dict(zip((r["seq_id"] for r in batch), texts))
            wav = rec["audio_path"]
            n_chunks = 1
            over_limit = is_over_limit(rec)
            if over_limit:
                print(
                    f"    SKIP native-context {rec['seq_id']} dur={rec.get('audio_duration')}s "
                    f"> {AF3_INFER_MAX_AUDIO_SEC}s (AF3 8192-token window); not scored in primary pool"
                )
                text, pred, scores = "", [], {
                    "macro_f1": 0.0, "micro_f1": 0.0, "mean_tiou": 0.0, "parsed": 0.0,
                    "n_gold": len(rec.get("segments") or []), "n_pred": 0,
                    "_tp": {}, "_fp": {}, "_fn": {}, "_tious": [], "_be_errs": [],
                    "_hits": 0, "_n_g_tr": 0, "_n_p_tr": 0,
                }
            elif is_sft_chunked(rec):
                text, pred, n_chunks = hf_generate.generate_sft_chunked(
                    model, processor, instruction, wav, sr, args.max_new_tokens, device, chunk_sec,
                    prompt_layout=layout,
                )
                scores = score_sequence(gold_segments(rec), pred)
            elif sft_path:
                text = batch_texts.pop(rec["seq_id"])
                pred, scores = score_text(text, rec)
            elif chunk_qwen2:
                text, pred, n_chunks = hf_generate.generate_qwen2_chunked(
                    model, processor, adapter, instruction, wav, sr, args.max_new_tokens, device, chunk_sec,
                    prompt_layout=layout,
                )
                scores = score_sequence(gold_segments(rec), pred)
            else:
                text = hf_generate.generate_pretrained(
                    model, processor, adapter, instruction, wav, sr, args.max_new_tokens, device,
                    prompt_layout=layout,
                )
                pred, scores = score_text(text, rec)
            row = {
                "seq_id": rec["seq_id"],
                "population": rec.get("population"),
                "condition": rec.get("condition") or rec.get("population"),
                "seq_len": rec["seq_len"],
                "length_bin": rec.get("length_bin") or bin_of_record(rec),
                "conversation": rec.get("conversation"),
                "podcast_id": rec.get("podcast_id"),
                "audio_duration": rec.get("audio_duration"),
                "n_chunks": n_chunks,
                "over_native_context_limit": over_limit,
                "ground_truth": rec["segments"],
                "pred_text": text,
                "pred": pred,
                "macro_f1": scores["macro_f1"],
                "micro_f1": scores.get("micro_f1"),
                "mean_tiou": scores["mean_tiou"],
                "parsed": scores["parsed"],
                "_scores": pack_scores(scores),
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            if i == 1 or i % 10 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}  {rec['seq_id']}  F1={scores['macro_f1']:.3f}")
    all_rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    scored_rows = [r for r in all_rows if not r.get("over_native_context_limit")]
    packed = [r.get("_scores") or score_text(r.get("pred_text") or "", {"segments": r["ground_truth"]})[1] for r in scored_rows]
    # score_text returns tuple; normalize
    norm = []
    for item, row in zip(packed, scored_rows):
        if isinstance(item, tuple):
            norm.append(item[1])
        elif isinstance(item, dict) and item.get("n_gold") is not None:
            norm.append(item)
        else:
            _pred, scores = score_text(row.get("pred_text") or "", {"segments": row["ground_truth"]})
            norm.append(scores)
    summary = pool_metrics(norm) if norm else pool_metrics([])
    summary.update(
        {
            "model": model_key,
            "adapter": str(adapter_dir) if adapter_dir else None,
            "tag": tag,
            "prompt": str(args.prompt),
            "prompt_layout": layout,
            "jsonl": str(args.jsonl),
            "n_records": len(scored_rows),
            "n_records_written": len(all_rows),
            "n_over_native_context_limit": sum(1 for r in all_rows if r.get("over_native_context_limit")),
            "chunk_sec": chunk_sec if (chunk_qwen2 or args.sft_chunk) else None,
            "sft_chunk": bool(args.sft_chunk),
            "n_sft_chunked": sum(1 for r in all_rows if (r.get("n_chunks") or 1) > 1),
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = quartile_report(scored_rows)
    (out_dir / "quartile_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_quartile_table(report))
    print(format_phase2(summary))


if __name__ == "__main__":
    main()
