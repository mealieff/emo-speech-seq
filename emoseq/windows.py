"""Split long audio into ≤30 s windows and stitch emotion spans."""
from __future__ import annotations

from typing import Any

import numpy as np

QWEN2_MAX_SEC = 30.0


def split_audio(audio: np.ndarray, sr: int, max_sec: float = QWEN2_MAX_SEC) -> list[tuple[float, np.ndarray]]:
    """Non-overlapping windows. Returns (offset_sec, waveform) pairs."""
    audio = np.asarray(audio)
    max_n = max(1, int(round(max_sec * sr)))
    if len(audio) <= max_n:
        return [(0.0, audio)]
    out: list[tuple[float, np.ndarray]] = []
    start = 0
    while start < len(audio):
        end = min(start + max_n, len(audio))
        out.append((start / float(sr), audio[start:end]))
        start = end
    return out


def shift_spans(spans: list[dict[str, Any]], offset_sec: float) -> list[dict[str, Any]]:
    shifted = []
    for s in spans:
        shifted.append(
            {
                "start": float(s["start"]) + offset_sec,
                "end": float(s["end"]) + offset_sec,
                "emotion": s["emotion"],
            }
        )
    return shifted


def stitch_windows(
    windows: list[tuple[float, list[dict[str, Any]]]], *, tol_sec: float = 0.05
) -> list[dict[str, Any]]:
    """Shift each window's spans by its offset and concatenate in window order.

    ``windows`` are consecutive, contiguous (offset, spans) pairs, so window k+1 starts at
    the boundary where window k ends. Spans A (window k) and B (window k+1) are merged
    only if A ends at that boundary, B starts at it (both within ``tol_sec``), and they
    share a label. Spans inside a window are never merged.
    """
    out: list[dict[str, Any]] = []
    prev: list[int] = []  # indices in ``out`` of spans belonging to the previous window
    for offset, spans in windows:
        cur = sorted(shift_spans(spans, offset), key=lambda s: (s["start"], s["end"]))
        now: list[int] = []
        merged_b: int | None = None
        for j, b in enumerate(cur):
            if abs(b["start"] - offset) > tol_sec:
                continue
            for i in prev:
                a = out[i]
                if a["emotion"] == b["emotion"] and abs(a["end"] - offset) <= tol_sec:
                    a["end"] = b["end"]
                    now.append(i)
                    merged_b = j
                    break
            if merged_b is not None:
                break
        for j, b in enumerate(cur):
            if j != merged_b:
                out.append(b)
                now.append(len(out) - 1)
        prev = now
    return out
