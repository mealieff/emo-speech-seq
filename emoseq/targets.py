"""Sequence SFT target strings: chronological ``(start, end, emotion)`` lines.

Must match ``scripts/prompts/sequence.txt`` (train and eval use the same instruction;
gold times are only in the assistant target).
"""
from __future__ import annotations

from typing import Any


def format_target(segments: list[dict[str, Any]]) -> str:
    lines = []
    for seg in segments:
        t0 = float(seg["start"])
        t1 = float(seg["end"])
        emo = seg["emotion"]
        lines.append(f"({t0:.2f}, {t1:.2f}, {emo})")
    return "\n".join(lines)


def target_string_from_record(record: dict[str, Any]) -> str:
    if record.get("target"):
        return str(record["target"])
    segs = record.get("segments")
    if not segs:
        raise ValueError("record missing segments/target")
    return format_target(segs)
