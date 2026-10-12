#!/usr/bin/env python3
"""Evaluate WavLM-TimeAug on a sequence split (default Test1), per length bin.

Writes ``predictions.jsonl``, ``quartile_report.json`` and ``metrics.json`` to ``--out-dir``.
``--oracle`` decodes emotions on the gold boundaries instead of the predicted ones.

    python scripts/inference/wavlm_timeaug.py --split test1
    python scripts/inference/wavlm_timeaug.py --split test1 --oracle --out-dir runs/test1/sequence/wavlm_timeaug_oracle
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from emoseq import config

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from emoseq.io import write_json, write_jsonl
from emoseq.scoring.report import format_quartile_table, quartile_report
from emoseq.wavlm import WAVLM_ID, WAVLM_REVISION
from emoseq.wavlm.dataset import SeqDataset
from emoseq.wavlm.decode import (
    boundary_ap,
    boundary_f1,
    bounds_to_intervals,
    gold_to_e5,
    local_maxima,
    peak_pick,
    to_pred_segments,
)
from emoseq.wavlm.model import FrameASP, WavLMSeq, freeze_bottom_layers, load_norm_stat

CAT_SER = config.WAVLM_IS2025_DIR
MANIFESTS = config.MANIFEST_DIR


def collate_one(batch):
    assert len(batch) == 1
    return batch[0]


def load_pt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "dev", "test1"], default="test1")
    p.add_argument("--ckpt", default=str(config.WAVLM_TIMEAUG_CKPT))
    p.add_argument("--theta-b", type=float, default=-1.0, help="<0 = use threshold stored in the checkpoint")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--out-dir", default=str(config.RUNS_DIR / "test1" / "sequence" / "wavlm_timeaug"))
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def build_model(ckpt_path: Path) -> tuple[WavLMSeq, dict]:
    ssl = AutoModel.from_pretrained(WAVLM_ID, revision=WAVLM_REVISION)
    asp = FrameASP(ssl.config.hidden_size)
    model = WavLMSeq(ssl, asp)
    blob = load_pt(ckpt_path)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    freeze_bottom_layers(model.ssl)
    return model, blob if isinstance(blob, dict) else {}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav_mean, wav_std = load_norm_stat(CAT_SER / "train_norm_stat.pkl")
    ds = SeqDataset(MANIFESTS / f"{args.split}.jsonl", wav_mean, wav_std, crop_sec=None)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_one)
    print(f"loading {args.ckpt}", flush=True)
    model, blob = build_model(Path(args.ckpt))
    model.to(device)
    model.eval()
    theta = float(args.theta_b)
    if theta < 0:
        theta = float(blob.get("theta") or 0.05)
    print(f"theta={theta} oracle={args.oracle}", flush=True)
    rows = []
    aps = []
    for i, batch in enumerate(tqdm(loader, desc=args.split)):
        wav = batch["wav"].to(device)
        rec = batch["rec"]
        dur = float(batch["duration"])
        gold = batch["intervals"]
        internals = [iv[1] for iv in gold[:-1]]
        frames_emo, frames_bound = model.encode_frames(wav)
        b_logits = model.boundary_logits(frames_bound)
        times, scores = local_maxima(b_logits, dur)
        aps.append(boundary_ap(times, scores, internals))
        if args.oracle:
            spans = list(gold)
        else:
            spans = bounds_to_intervals(peak_pick(b_logits, theta, dur), dur)
        e_logits = model.emotion_logits(frames_emo, spans)
        rows.append(
            {
                "seq_id": rec["seq_id"],
                "seq_len": rec["seq_len"],
                "length_bin": rec.get("length_bin") or rec.get("bin"),
                "audio_duration": dur,
                "ground_truth": gold_to_e5(rec),
                "pred": to_pred_segments(spans, e_logits),
                "oracle_boundaries": bool(args.oracle),
                "boundary_f1": boundary_f1(rec["gold_intervals"], spans),
            }
        )
        if args.limit and len(rows) >= args.limit:
            break
    report = quartile_report(rows)
    report["boundary_ap"] = float(np.mean(aps)) if aps else 0.0
    report["theta"] = None if args.oracle else theta
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "predictions.jsonl", rows)
    write_json(out / "quartile_report.json", report)
    write_json(out / "metrics.json", {**report["additional"]["overall"], "boundary_ap": report["boundary_ap"],
                                      "theta": report["theta"], "ckpt": str(args.ckpt), "n_records": len(rows)})
    print(format_quartile_table(report), flush=True)
    print(f"boundary_ap={report['boundary_ap']:.4f} n={len(rows)} -> {out}", flush=True)

if __name__ == "__main__":
    main()
