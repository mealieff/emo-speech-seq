#!/usr/bin/env python3
"""Utterance lists for single-clip SER: one JSONL per split (Train / Development / Test1).

Keeps the eight emotion classes (A S H U F D C N → 0..7); O (other) and X (no agreement)
are dropped. Each line: {"audio_path", "label", "file_name"}.

    python scripts/data/utterances.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from emoseq import config

LABEL_MAP = {"A": 0, "S": 1, "H": 2, "U": 3, "F": 4, "D": 5, "C": 6, "N": 7}
SPLITS = {"Train": "train", "Development": "dev", "Test1": "test1"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--msp-root", default=str(config.MSP_ROOT))
    p.add_argument("--out-dir", default=str(config.UTT_DIR))
    args = p.parse_args()

    msp_root = Path(args.msp_root)
    out_dir = Path(args.out_dir)
    label_path = msp_root / "Labels" / "labels_consensus.csv"
    part_path = msp_root / "Partitions.txt"
    for f in (label_path, part_path):
        if not f.exists():
            raise SystemExit(f"missing {f}; set MSP_ROOT in configs/paths.env")

    df = pd.read_csv(label_path)
    df = df[df["EmoClass"].isin(LABEL_MAP)].copy()
    df["label"] = df["EmoClass"].map(LABEL_MAP)
    partition = pd.read_csv(part_path, sep=";", header=None, names=["split", "FileName"], skipinitialspace=True)
    df = df.merge(partition, on="FileName", how="inner")
    print(f"8-class utterances: {len(df)}  by split: {df['split'].value_counts().to_dict()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, slug in SPLITS.items():
        rows, missing = [], 0
        for _, row in df[df["split"] == split_name].iterrows():
            audio_path = msp_root / "Audios" / row["FileName"]
            if not audio_path.exists():
                missing += 1
                continue
            rows.append({"audio_path": str(audio_path), "label": int(row["label"]), "file_name": row["FileName"]})
        out = out_dir / f"msp_podcast_{slug}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  {split_name:12s} {len(rows):6d} → {out}" + (f"  ({missing} missing audio)" if missing else ""))


if __name__ == "__main__":
    main()
