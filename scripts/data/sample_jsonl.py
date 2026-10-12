#!/usr/bin/env python3
"""Seeded random sample of N rows from a JSONL (file order kept).

    python scripts/data/sample_jsonl.py data/sequences/test1.jsonl 30 7 runs/test1/spotcheck/seq.jsonl
"""
from __future__ import annotations

import random
import sys
from pathlib import Path


def main() -> None:
    src, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    lines = [l for l in Path(src).read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = sorted(random.Random(seed).sample(range(len(lines)), min(n, len(lines))))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines[i] + "\n" for i in keep), encoding="utf-8")
    print(f"sampled {len(keep)}/{len(lines)} rows (seed {seed}) -> {out}")


if __name__ == "__main__":
    main()
