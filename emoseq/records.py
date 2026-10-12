"""JSONL loader for sequence ALM SFT (concat wav + gold spans)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emoseq.labels import record_has_excluded_emotion


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dropped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if record_has_excluded_emotion(rec):
                dropped += 1
                continue
            rows.append(rec)
            if limit is not None and len(rows) >= limit:
                break
    if dropped:
        print(f"dropped {dropped} sequences with emotion X or O")
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows
