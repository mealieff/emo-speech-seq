#!/usr/bin/env python3
"""Label table for the IS 2025 WavLM utterance classifier: one-hot 8-class columns + Split_Set.

    python scripts/data/wavlm_labels.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emoseq import config

EMOTIONS = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
EMOTION_CODES = ["A", "S", "H", "U", "F", "D", "C", "N"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=config.CONSENSUS_CSV)
    p.add_argument("--dst", type=Path, default=config.WAVLM_LABELS_CSV)
    args = p.parse_args()
    df = pd.read_csv(args.src)
    df = df[df["EmoClass"].isin(EMOTION_CODES)].copy()
    for name, code in zip(EMOTIONS, EMOTION_CODES):
        df[name] = (df["EmoClass"] == code).astype(float)
    out = df[["FileName", *EMOTIONS, "Split_Set"]]
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.dst, index=False)
    print(f"wrote {args.dst} n={len(out)} splits={out['Split_Set'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
