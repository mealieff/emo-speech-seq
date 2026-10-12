#!/usr/bin/env python3
"""Test1 inference for the ALMs with bespoke stacks: SALMONN and Kimi-Audio (pretrained or LoRA).

Runs in the legacy environment (``scripts/setup/install_legacy.sh``): SALMONN's vendored
LLaMA needs tuple KV caches and Kimi-Audio pins transformers 4.x. Writes the same files as
the Hugging Face scripts so every system is scored identically:

  --task utt : ``eval_<split>.json`` + ``predictions_<split>.json`` (as scripts/inference/utterance.py)
  --task seq : ``predictions.jsonl`` + ``metrics.json`` + ``quartile_report.json`` (as scripts/inference/sequence.py)

Decoding is greedy for both models (SALMONN's stock 4-beam config is overridden). SALMONN's
encoder takes <= 30 s, so sequences are decoded in 30 s windows with spans shifted back to
sequence time and merged only across window seams (same as pretrained Qwen2-Audio).
Kimi-Audio slices long audio into 30 s Whisper windows internally, so it gets the whole sequence.

    python scripts/inference/bespoke.py --model salmonn --task utt --out runs/test1/utterance/salmonn_pt
    python scripts/inference/bespoke.py --model salmonn --task seq --out runs/test1/sequence/salmonn_pt
    python scripts/inference/bespoke.py --model kimi_audio --task seq --out runs/test1/sequence/kimi_audio_pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emoseq import config

import librosa
import numpy as np

from emoseq.bins import bin_of_record
from emoseq.io import load_jsonl
from emoseq.models.bespoke import SR, load_backend
from emoseq.parsing.sequence import parse_prediction
from emoseq.scoring.report import format_phase2, format_quartile_table, pool_metrics, quartile_report
from emoseq.scoring.sequence import gold_segments, pack_scores, score_sequence, score_text
from emoseq.scoring.utterance import (
    EMOTION_NAMES,
    classify_task_a_output,
    parse_task_a,
    parse_task_a_first_n,
    task_a_metrics,
)
from emoseq.windows import QWEN2_MAX_SEC, split_audio, stitch_windows

UTT_PROMPT = config.UTTERANCE_PROMPT
PROMPT_PATH = config.SEQUENCE_PROMPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["salmonn", "kimi_audio"])
    p.add_argument("--task", required=True, choices=["utt", "seq"])
    p.add_argument("--jsonl", default=None,
                   help="Default: data/utterances/msp_podcast_test1.jsonl (utt), data/sequences/test1.jsonl (seq).")
    p.add_argument("--out", required=True, help="Output dir.")
    p.add_argument("--prompt", default=None, help="Default: configs/prompts/utterance.txt (utt), "
                                                   "configs/prompts/sequence.txt (seq).")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Default 16 (utt) / 800 (seq).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Clips (utt) or 30 s windows (seq) per forward pass; use only after the "
                        "batch-1 vs batched spot-check passes (scripts/evaluate/spotcheck.py).")
    p.add_argument("--sft-dir", default=None,
                   help="Checkpoint dir from scripts/finetune/bespoke_lora.py (trainable.pt); "
                        "decodes with the LoRA SFT model instead of the pretrained one.")
    args = p.parse_args()
    if args.jsonl is None:
        args.jsonl = str(config.utterance_jsonl("test1") if args.task == "utt" else config.sequence_jsonl("test1"))
    return args


# ---------------------------------------------------------------- tasks
def read_done(path: Path, key: str, resume: bool) -> dict[str, dict]:
    if not resume or not path.is_file():
        if path.is_file():
            path.unlink()
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done[str(r[key])] = r
    return done


def run_utt(args, backend, out_dir: Path) -> None:
    instruction = Path(args.prompt or UTT_PROMPT).read_text(encoding="utf-8").strip()
    max_new = args.max_new_tokens or 16
    records = load_jsonl(args.jsonl, args.limit)
    split = Path(args.jsonl).stem.replace("msp_podcast_", "")
    partial = out_dir / f"predictions_{split}.partial.jsonl"
    done = read_done(partial, "idx", not args.no_resume)
    print(f"utterance {'sft' if args.sft_dir else 'pretrained'} | {backend.name} | {len(done)} done, "
          f"{len(records) - len(done)} remaining | max_new={max_new} | batch={args.batch_size}")
    todo = [i for i in range(len(records)) if str(i) not in done]
    with partial.open("a", encoding="utf-8") as fout:
        for b in range(0, len(todo), args.batch_size):
            idxs = todo[b : b + args.batch_size]
            audios = [librosa.load(records[i]["audio_path"], sr=SR)[0] for i in idxs]
            texts = backend.generate_batch(instruction, audios, max_new)
            for i, text in zip(idxs, texts):
                item = records[i]
                gt = int(item["label"])
                pred = parse_task_a_first_n(text)
                rec = {"idx": i, "audio_path": item.get("audio_path"), "label": gt,
                       "label_name": EMOTION_NAMES[gt], "first_n_parsed": pred,
                       **classify_task_a_output(text, gt)}
                fout.write(json.dumps(rec) + "\n")
                done[str(i)] = rec
                if len(done) % 50 == 0:
                    print(f"  [{len(done)}/{len(records)}]", flush=True)
            fout.flush()

    recs = [done[str(i)] for i in range(len(records))]
    y_true, y_pred, n_fail, n_strict = [], [], 0, 0
    y_pred_strict = []
    for r in recs:
        pred = r["first_n_parsed"]
        strict = parse_task_a(r["raw_text"])
        n_strict += strict is None
        if pred is None:
            n_fail += 1
            pred = 7  # Neutral fallback, counted as a parse failure
        r["pred_for_metrics"] = pred
        r["pred_name"] = EMOTION_NAMES[pred]
        y_true.append(r["label"])
        y_pred.append(pred)
        y_pred_strict.append(7 if strict is None else strict)
    n = len(recs)
    metrics = task_a_metrics(y_true, y_pred)
    metrics.update(parse_mode="first_n", parse_failures=n_fail,
                   parse_failure_rate=round(n_fail / n, 4) if n else 0.0,
                   strict_parse_failures=n_strict,
                   strict_parse_failure_rate=round(n_strict / n, 4) if n else 0.0)
    strict_m = task_a_metrics(y_true, y_pred_strict)
    metrics["strict"] = {k: strict_m[k] for k in ("WA", "UA", "macro_F1")}
    header = {"task": "utt", "model": backend.name, "base_only": not args.sft_dir, "inference_prompt_mode": "native",
              "decoding": "greedy", "max_new_tokens": max_new, "batch_size": args.batch_size, "model_dir": args.sft_dir,
              "base_model": backend.base_model, "eval_jsonl": args.jsonl, "split": split}
    pred_path = out_dir / f"predictions_{split}.json"
    pred_path.write_text(json.dumps({**header, "n": n, "records": recs}, indent=2), encoding="utf-8")
    metrics["predictions_file"] = str(pred_path)
    (out_dir / f"eval_{split}.json").write_text(json.dumps({**header, "metrics": metrics}, indent=2),
                                                encoding="utf-8")
    print(f"WA {metrics['WA']}  UA {metrics['UA']}  macro-F1 {metrics['macro_F1']}  "
          f"parse fail {metrics['parse_failure_rate']} (strict {metrics['strict_parse_failure_rate']})")


def run_seq(args, backend, out_dir: Path) -> None:
    instruction = Path(args.prompt or PROMPT_PATH).read_text(encoding="utf-8").strip()
    max_new = args.max_new_tokens or 800
    records = load_jsonl(args.jsonl, args.limit)
    jsonl = out_dir / "predictions.jsonl"
    done = read_done(jsonl, "seq_id", not args.no_resume)
    todo = [r for r in records if r["seq_id"] not in done]
    chunked = not backend.native_long_audio
    tag = "sft" if args.sft_dir else "pretrained"
    bs = args.batch_size
    print(f"sequence {tag} | {backend.name} | {len(done)} done, {len(todo)} remaining | chunk={chunked} | "
          f"max_new={max_new} | batch={bs} windows")
    with jsonl.open("a", encoding="utf-8") as fout:
        for g in range(0, len(todo), bs):
            group = todo[g : g + bs]
            wins: list[tuple[int, float, np.ndarray]] = []  # (record index in group, offset, wav)
            for k, rec in enumerate(group):
                audio = librosa.load(rec["audio_path"], sr=SR)[0]
                for offset, wav in (split_audio(audio, SR, QWEN2_MAX_SEC) if chunked else [(0.0, audio)]):
                    wins.append((k, offset, wav))
            outs: list[str] = []
            for w in range(0, len(wins), bs):
                outs.extend(backend.generate_batch(instruction, [x[2] for x in wins[w : w + bs]], max_new))
            for k, rec in enumerate(group):
                mine = [(off, wav, t) for (kk, off, wav), t in zip(wins, outs) if kk == k]
                if chunked:
                    text = "\n\n".join(f"# chunk offset={off:.2f}s dur={len(wav)/SR:.2f}s\n{t}" for off, wav, t in mine)
                    pred = stitch_windows([(off, parse_prediction(t)) for off, _, t in mine])
                    scores = score_sequence(gold_segments(rec), pred)
                else:
                    text = mine[0][2]
                    pred, scores = score_text(text, rec)
                i, n_chunks = g + k + 1, len(mine)
                _write_seq_row(fout, rec, text, pred, scores, n_chunks)
                if i == 1 or i % 10 == 0 or i == len(todo):
                    print(f"    {i}/{len(todo)}  {rec['seq_id']}  F1={scores['macro_f1']:.3f}", flush=True)
    _finish_seq(args, backend, out_dir, jsonl, chunked, tag, max_new)


def _write_seq_row(fout, rec, text, pred, scores, n_chunks) -> None:
    row = {
        "seq_id": rec["seq_id"], "population": rec.get("population"),
        "condition": rec.get("condition") or rec.get("population"),
        "seq_len": rec["seq_len"], "length_bin": rec.get("length_bin") or bin_of_record(rec),
        "conversation": rec.get("conversation"), "podcast_id": rec.get("podcast_id"),
        "audio_duration": rec.get("audio_duration"), "n_chunks": n_chunks,
        "over_native_context_limit": False, "ground_truth": rec["segments"],
        "pred_text": text, "pred": pred, "macro_f1": scores["macro_f1"],
        "micro_f1": scores.get("micro_f1"), "mean_tiou": scores["mean_tiou"], "parsed": scores["parsed"],
        "_scores": pack_scores(scores),
    }
    fout.write(json.dumps(row) + "\n")
    fout.flush()


def _finish_seq(args, backend, out_dir: Path, jsonl: Path, chunked: bool, tag: str, max_new: int) -> None:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = pool_metrics([r["_scores"] for r in rows])
    summary.update({
        "model": backend.name, "adapter": args.sft_dir, "tag": tag, "prompt": str(args.prompt or PROMPT_PATH),
        "prompt_layout": "user", "jsonl": args.jsonl, "n_records": len(rows), "n_records_written": len(rows),
        "n_over_native_context_limit": 0, "chunk_sec": QWEN2_MAX_SEC if chunked else None,
        "decoding": "greedy", "max_new_tokens": max_new, "base_model": backend.base_model,
        "batch_size_windows": args.batch_size,
    })
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = quartile_report(rows)
    (out_dir / "quartile_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_quartile_table(report))
    print(format_phase2(summary))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = load_backend(args.model, args.sft_dir)
    (run_utt if args.task == "utt" else run_seq)(args, backend, out_dir)


if __name__ == "__main__":
    main()
