"""MSP-Podcast filename parsing and consecutive-run grouping."""
from __future__ import annotations

import re
from typing import Any

NAME_RE = r"^MSP-PODCAST_(\d+)((?:_\d+)+)\.wav$"

_NAME = re.compile(NAME_RE, re.I)


def parse_podcast_name(filename: str) -> tuple[str, tuple[int, ...], int]:
    """Return (podcast_id zero-padded, full suffix tuple, primary consecutive index).

    MSP-PODCAST_0023_0321.wav → ('0023', (321,), 321)
    MSP-PODCAST_5727_0060_0003.wav → ('5727', (60, 3), 60)
    Consecutive-ID runs use the first suffix integer only. Sub-suffixes of the
    same primary index stay in the same run; a gap is a jump of 2+ in that index.
    """
    fn = filename if filename.lower().endswith(".wav") else f"{filename}.wav"
    m = _NAME.match(fn)
    if not m:
        raise ValueError(f"unrecognized MSP filename: {filename!r}")
    podcast_id = m.group(1).zfill(4)
    suffixes = tuple(int(x) for x in m.group(2).strip("_").split("_"))
    return podcast_id, suffixes, suffixes[0]


def conversation_id(podcast_id: str) -> str:
    return f"MSP-Conversation_{str(podcast_id).zfill(4)}"


def load_semicolon_times(path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        name, start, end = [x.strip() for x in line.split(";")]
        out[name] = (float(start), float(end))
    return out


def n_skipped_ids(run: list[int]) -> int:
    """Count of missing unique IDs between adjacent members (gap of 1 each)."""
    return sum(1 for a, b in zip(run, run[1:]) if b - a > 1)


def unique_id_runs(
    seq_indices: list[int],
    max_skip: int = 0,
    alternate_skips: bool = False,
) -> list[list[int]]:
    """Non-overlapping maximal unique-ID sequences. Not a sliding window.

    ``max_skip=0``: next unique ID must be exactly +1 (strict back-to-back).
    ``max_skip=1``: next unique ID may be +1 or +2 (skip at most one ID on a join).
    ``alternate_skips=True``: a skip cannot follow a skip. After skipping one ID,
    the next join must be strictly consecutive; after a consecutive join another
    skip is allowed. A sequence may therefore contain multiple skips, just not
    on consecutive turns.

    Length of a run is the number of unique IDs present, not the ID span.
    """
    ids = sorted(set(int(x) for x in seq_indices))
    if not ids:
        return []
    runs: list[list[int]] = []
    cur = [ids[0]]
    last_join_skipped = False
    for nxt in ids[1:]:
        gap = nxt - cur[-1] - 1
        skip_ok = 0 <= gap <= max_skip and not (alternate_skips and last_join_skipped and gap > 0)
        if skip_ok:
            cur.append(nxt)
            last_join_skipped = gap > 0
        else:
            runs.append(cur)
            cur = [nxt]
            last_join_skipped = False
    runs.append(cur)
    return runs


def consecutive_runs(segments: list[dict[str, Any]], max_skip: int = 0) -> list[list[dict[str, Any]]]:
    """Non-overlapping maximal consecutive-suffix runs. Not a sliding window."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda r: (r["suffixes"], r["segment_id"]))
    runs: list[list[dict[str, Any]]] = []
    cur = [ordered[0]]
    for rec in ordered[1:]:
        prev = cur[-1]["seq_idx"]
        nxt = rec["seq_idx"]
        if nxt == prev or nxt <= prev + 1 + max_skip:
            cur.append(rec)
        else:
            runs.append(cur)
            cur = [rec]
    runs.append(cur)
    return runs


def quartile_breakpoints(values: list[float]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    q1, q2, q3 = (float(x) for x in np.quantile(arr, [0.25, 0.5, 0.75]))
    return {"q1": q1, "q2": q2, "q3": q3, "min": float(arr.min()), "max": float(arr.max())}


def quartile_bin(x: float, bp: dict[str, float]) -> str:
    if x <= bp["q1"]:
        return "Q1"
    if x <= bp["q2"]:
        return "Q2"
    if x <= bp["q3"]:
        return "Q3"
    return "Q4"


def merge_intervals(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    ordered = sorted((float(a), float(b)) for a, b in spans if b > a)
    if not ordered:
        return []
    out = [ordered[0]]
    for a, b in ordered[1:]:
        if a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def gap_spans(total: tuple[float, float], covered: list[tuple[float, float]]) -> list[tuple[float, float]]:
    t0, t1 = float(total[0]), float(total[1])
    merged = merge_intervals([(max(a, t0), min(b, t1)) for a, b in covered])
    gaps = []
    cursor = t0
    for a, b in merged:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if t1 > cursor:
        gaps.append((cursor, t1))
    return gaps
