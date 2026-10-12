"""Parse span lists: ``(start, end, emotion)`` plus dict lists and JSON fallbacks."""
from __future__ import annotations

import ast
import json
import re
from typing import Any

from emoseq.labels import EMOTION_LOOKUP
from emoseq.parsing.free_form import parse_prediction as parse_e2
from emoseq.parsing.timestamps import to_sec, ts_group

# Parentheses (prompt) or square brackets (Audio-Flamingo-3 often emits these).
_TUPLE_RE = {
    clock: re.compile(
        rf"[\(\[]\s*{ts_group(clock)}\s*,\s*{ts_group(clock)}\s*,\s*"
        r"['\"]?([A-Za-z][A-Za-z/ ]*)['\"]?\s*[\)\]]"
    )
    for clock in (True, False)
}


def _canon(raw: str) -> str | None:
    return EMOTION_LOOKUP.get(str(raw).strip().lower())


def _row_from_dict(row: dict, clock: bool) -> dict[str, Any] | None:
    start = row.get("start", row.get("Start", row.get("START_TIME", row.get("start_time"))))
    end = row.get("end", row.get("End", row.get("END_TIME", row.get("end_time"))))
    emo_raw = row.get("emotion", row.get("Emotion", row.get("EMOTION")))
    if start is None or end is None or emo_raw is None:
        return None
    emo = _canon(str(emo_raw))
    if emo is None:
        return None
    t0, t1 = to_sec(start, clock), to_sec(end, clock)
    if t0 is None or t1 is None:
        return None
    if t1 < t0:
        t0, t1 = t1, t0
    return {"start": t0, "end": t1, "emotion": emo}


def _parse_list_blob(text: str, clock: bool) -> list[dict[str, Any]]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    blob = text[start : end + 1]
    data = None
    try:
        data = ast.literal_eval(blob)
    except (ValueError, SyntaxError):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            try:
                data = json.loads(blob.replace("'", '"'))
            except json.JSONDecodeError:
                return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    out = []
    for row in data:
        if isinstance(row, dict):
            rec = _row_from_dict(row, clock)
            if rec:
                out.append(rec)
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def parse_prediction(text: str, *, clock: bool = True) -> list[dict[str, Any]]:
    """Prefer prompt tuples, then dict lists (JSON or Python), then JSON/prose.

    ``clock=False`` restricts timestamps to plain seconds (strict format).
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        tuples: list[dict[str, Any]] = []
        for m in _TUPLE_RE[clock].finditer(text):
            emo = _canon(m.group(3))
            if emo is None:
                continue
            t0, t1 = to_sec(m.group(1), clock), to_sec(m.group(2), clock)
            if t0 is None or t1 is None:
                continue
            if t1 < t0:
                t0, t1 = t1, t0
            tuples.append({"start": t0, "end": t1, "emotion": emo})
        if tuples:
            tuples.sort(key=lambda s: (s["start"], s["end"]))
            return tuples
        listed = _parse_list_blob(text, clock)
        if listed:
            return listed
        return parse_e2(text, clock=clock)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return []


_CHUNK_HEADER = re.compile(r"^# chunk offset=([0-9.]+)s dur=[0-9.]+s$", re.MULTILINE)


def parse_saved_output(text: str, *, clock: bool = True) -> list[dict[str, Any]]:
    """Re-parse a saved ``pred_text``; chunked runs (``# chunk offset=..`` headers) are
    parsed per window, shifted and seam-merged exactly as at inference time."""
    from emoseq.windows import stitch_windows

    text = text or ""
    heads = list(_CHUNK_HEADER.finditer(text))
    if not heads:
        return parse_prediction(text, clock=clock)
    windows = []
    for k, h in enumerate(heads):
        body = text[h.end() : heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        windows.append((float(h.group(1)), parse_prediction(body, clock=clock)))
    return stitch_windows(windows)
