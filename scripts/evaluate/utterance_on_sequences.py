#!/usr/bin/env python3
"""Score utterance-level predictions on the Test1 sequences (oracle segmentation).

Each sequence segment is one original MSP-Podcast utterance, so the prediction for a
segment is the utterance model's prediction on that file. Predicted spans reuse the gold
boundaries, so tIoU is trivially 1 and is not reported; Acc / F1 use the same quartile
scorer as the sequence systems.

``--preds`` is either the WavLM IS2025 CSV (``scripts/inference/wavlm_utterance.py`` writes
``cat_test1_preds.csv``) or an ALM ``predictions_test1.json`` from
``scripts/inference/utterance.py`` (segment label = ``pred_for_metrics``: first-N parse,
Neutral fallback).

    python scripts/evaluate/utterance_on_sequences.py \\
        --preds runs/test1/utterance/wavlm_is2025/cat_test1_preds.csv \\
        --out-dir runs/test1/utt_on_sequences/wavlm_is2025
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from emoseq import config
from emoseq.bins import bin_of_record
from emoseq.scoring.report import format_quartile_table, pool_metrics, quartile_report
from emoseq.scoring.sequence import gold_segments, pack_scores, score_sequence

TO_SEQ_LABEL = {"Surprise": "Surprised"}  # IS2025 class name -> sequence label vocabulary
EMOTION_NAMES = ["Angry", "Sad", "Happy", "Surprised", "Fear", "Disgust", "Contempt", "Neutral"]  # emoseq.scoring.utterance


def load_preds(path: str) -> dict[str, str]:
    if path.endswith(".json"):
        records = json.loads(Path(path).read_text(encoding="utf-8"))["records"]
        return {Path(r["audio_path"]).name: EMOTION_NAMES[int(r["pred_for_metrics"])] for r in records}
    with open(path, newline="") as f:
        return {r["FileName"]: TO_SEQ_LABEL.get(r["pred_name"], r["pred_name"]) for r in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", required=True)
    p.add_argument("--jsonl", default=str(config.sequence_jsonl("test1")))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default="wavlm_is2025_cat_ser_seed7")
    args = p.parse_args()

    pred_of = load_preds(args.preds)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], 0
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gold = gold_segments(rec)
        pred = []
        for seg, fn in zip(gold, rec["file_names"]):
            lab = pred_of.get(fn)
            missing += lab is None
            pred.append({"start": seg["start"], "end": seg["end"], "emotion": lab or "Neutral"})
        scores = score_sequence(gold, pred)
        rows.append({
            "seq_id": rec["seq_id"], "seq_len": rec["seq_len"],
            "length_bin": rec.get("length_bin") or bin_of_record(rec),
            "audio_duration": rec.get("audio_duration"), "ground_truth": rec["segments"], "pred": pred,
            "macro_f1": scores["macro_f1"], "mean_tiou": scores["mean_tiou"], "parsed": scores["parsed"],
            "_scores": pack_scores(scores),
        })
    if missing:
        raise SystemExit(f"{missing} segments have no utterance prediction in {args.preds}")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = pool_metrics([r["_scores"] for r in rows])
    summary.update({"model": args.model, "tag": "utterance_oracle_segments",
                    "preds": args.preds, "jsonl": args.jsonl, "n_records": len(rows),
                    "note": "gold boundaries; tIoU trivially 1, not reported"})
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = quartile_report(rows)
    (out / "quartile_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_quartile_table(report))


if __name__ == "__main__":
    main()
