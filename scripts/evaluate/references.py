#!/usr/bin/env python3
"""Content-free reference predictions on the Test1 sequences, scored like every system.

    python scripts/evaluate/references.py

grid5_neutral : back-to-back 5 s Neutral spans from 0 s to the end of the audio
                (the last span is whatever remains).
single_neutral: one Neutral span covering the whole audio.

Writes runs/test1/references/<name>/{predictions.jsonl,quartile_report.json,metrics.json}.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from emoseq import config
from emoseq.io import load_jsonl, write_json, write_jsonl
from emoseq.scoring.report import format_quartile_table, quartile_report
from emoseq.scoring.sequence import gold_segments

GRID_SEC = 5.0


def grid(duration: float, step: float) -> list[dict]:
    n = max(1, math.ceil(duration / step - 1e-9))
    return [{"start": i * step, "end": min((i + 1) * step, duration), "emotion": "Neutral"} for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", default=str(config.sequence_jsonl("test1")))
    p.add_argument("--out", default=str(config.RUNS_DIR / "test1" / "references"))
    args = p.parse_args()

    recs = load_jsonl(Path(args.jsonl))
    makers = {
        "grid5_neutral": lambda d: grid(d, GRID_SEC),
        "single_neutral": lambda d: [{"start": 0.0, "end": d, "emotion": "Neutral"}],
    }
    for name, make in makers.items():
        rows = []
        for r in recs:
            dur = float(r["audio_duration"])
            rows.append({
                "seq_id": r["seq_id"],
                "seq_len": r["seq_len"],
                "length_bin": r["length_bin"],
                "audio_duration": dur,
                "ground_truth": gold_segments(r),
                "pred": make(dur),
            })
        report = quartile_report(rows)
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "predictions.jsonl", rows)
        write_json(out / "quartile_report.json", report)
        write_json(out / "metrics.json", report["additional"]["overall"])
        print(name)
        print(format_quartile_table(report))


if __name__ == "__main__":
    main()
