"""Fallback parsers for JSON / brace / prose span lists.

Primary eval uses ``emoseq.parsing.sequence`` tuples ``(start, end, emotion)``. This module
still accepts compact JSON and looser brace formats so a model that copies
an example style is not scored as empty.
"""

from __future__ import annotations

import json
import re
from typing import Any

from emoseq.labels import EMOTION_LOOKUP, EMOTION_NAMES, TARGET_KEYS
from emoseq.parsing.timestamps import to_sec, ts_group

_CLOCKS = (True, False)
# {0.00 - 5.35 - Sad}  or  {00.000 - 05.346 - SAD}
_LINE_RE = {
    c: re.compile(rf"\{{\s*{ts_group(c)}\s*[-–]\s*{ts_group(c)}\s*[-–]\s*([A-Za-z]+)\s*\}}")
    for c in _CLOCKS
}
# Charlotte trainer target: {START_TIME: 0.00, END_TIME: 5.32, EMOTION: Neutral}
_CHARLOTTE_RE = {
    c: re.compile(
        rf"\{{\s*START_TIME\s*:\s*{ts_group(c)}\s*,\s*"
        rf"END_TIME\s*:\s*{ts_group(c)}\s*,\s*"
        r"EMOTION\s*:\s*([^},]+?)\s*\}",
        re.IGNORECASE,
    )
    for c in _CLOCKS
}


def emotion_name(label: int | str) -> str:
    if isinstance(label, str):
        key = label.strip()
        if key in EMOTION_LOOKUP:
            return EMOTION_LOOKUP[key]
        lower = key.lower()
        if lower in EMOTION_LOOKUP:
            return EMOTION_LOOKUP[lower]
        raise KeyError(f"unknown emotion {label!r}")
    return EMOTION_NAMES[int(label)]


def format_target(segments: list[dict[str, Any]]) -> str:
    """Compact JSON list, matching VibeVoice ``json.dumps(..., separators=(',', ':'))``."""
    rows = []
    for seg in segments:
        emotion = seg.get("emotion") or emotion_name(seg["label"])
        rows.append(
            {
                TARGET_KEYS[0]: round(float(seg["start"]), 2),
                TARGET_KEYS[1]: round(float(seg["end"]), 2),
                TARGET_KEYS[2]: emotion_name(emotion),
            }
        )
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def target_string_from_record(record: dict[str, Any]) -> str:
    segs = record.get("segments")
    if not segs:
        raise ValueError("record missing segments; rebuild with scripts/data/export_sequences.py")
    return format_target(segs)


def _canon_emotion(raw: str) -> str | None:
    return EMOTION_LOOKUP.get(raw.strip().lower())


def parse_prediction(text: str, *, clock: bool = True) -> list[dict[str, Any]]:
    """Parse a model string into ``[{start, end, emotion}, ...]``.

    Order: compact JSON, Charlotte ``{START_TIME, END_TIME, EMOTION}`` braces,
    E2.MD ``{start - end - EMO}`` lines, then free-form prose (emotion words
    plus ``from X to Y`` / ``(start, end)`` intervals). Empty JSON with no
    usable rows falls through so prose still gets a chance.
    """
    text = (text or "").strip()
    if not text:
        return []

    parsed = _parse_json(text, clock)
    if parsed:
        return parsed

    out: list[dict[str, Any]] = []
    for m in _CHARLOTTE_RE[clock].finditer(text):
        emo = _canon_emotion(m.group(3))
        if emo is None:
            continue
        out.append({"start": to_sec(m.group(1)), "end": to_sec(m.group(2)), "emotion": emo})
    if out:
        return out

    for m in _LINE_RE[clock].finditer(text):
        emo = _canon_emotion(m.group(3))
        if emo is None:
            continue
        out.append({"start": to_sec(m.group(1)), "end": to_sec(m.group(2)), "emotion": emo})
    if out:
        return out

    return _parse_prose(text, clock)


def _json_sec(value, clock: bool) -> float:
    sec = to_sec(value, clock)
    if sec is None:
        raise ValueError(f"unparseable timestamp {value!r}")
    return sec


def _parse_json(text: str, clock: bool) -> list[dict[str, Any]] | None:
    blob = text
    # Strip markdown fences if the model wraps the JSON.
    fence = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        blob = fence.group(1)
    else:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    out: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        start = row.get("Start", row.get("start", row.get("START_TIME")))
        end = row.get("End", row.get("end", row.get("END_TIME")))
        emo_raw = row.get("Emotion", row.get("emotion", row.get("EMOTION")))
        if start is None or end is None or emo_raw is None:
            continue
        emo = _canon_emotion(str(emo_raw))
        if emo is None:
            continue
        out.append({"start": _json_sec(start, clock), "end": _json_sec(end, clock), "emotion": emo})
    return out or None


_UNIT = r"(?:s|sec|secs|seconds|m|min|mins|minutes)?"
_EMO_ALT = "|".join(
    sorted((re.escape(k) for k in EMOTION_LOOKUP), key=len, reverse=True)
)
_EMO_RE = re.compile(rf"(?i)(?<![\w])({_EMO_ALT})(?![\w])")


def _interval_res(clock: bool) -> tuple[re.Pattern, ...]:
    n = ts_group(clock)
    return (
        re.compile(rf"\(\s*{n}\s*[,;–-]\s*{n}\s*\)"),
        re.compile(rf"\bfrom\s+{n}\s*{_UNIT}\s+(?:to|-|–)\s+{n}\s*{_UNIT}", re.I),
        re.compile(rf"\bbetween\s+{n}\s+and\s+{n}\s*{_UNIT}", re.I),
        re.compile(rf"\bat\s+{n}\s*[-–]\s*{n}\s*{_UNIT}", re.I),
        re.compile(rf"\bduring\s+{n}\s*[-–]\s*{n}\s*{_UNIT}", re.I),
    )


_INTERVAL_RES = {c: _interval_res(c) for c in _CLOCKS}


def _negated_emotion(text: str, start: int, raw: str) -> bool:
    """Drop 'no other' / 'not sad'; keep the class 'no agreement'."""
    if raw.strip().lower() == "no agreement":
        return False
    prefix = text[max(0, start - 16) : start].lower()
    return bool(re.search(r"\b(no|not|without|neither)\s+$", prefix))


def _clean_interval(t0: float, t1: float) -> tuple[float, float] | None:
    if t1 < t0:
        t0, t1 = t1, t0
    if t1 <= t0 or t1 > 600.0:
        return None
    return (t0, t1)


def _parse_prose(text: str, clock: bool) -> list[dict[str, Any]]:
    """Emotion words + from/to or parenthesized intervals (open test-prompt style)."""
    emos: list[tuple[int, int, str]] = []
    for m in _EMO_RE.finditer(text):
        raw = m.group(1)
        if _negated_emotion(text, m.start(), raw):
            continue
        emo = _canon_emotion(raw)
        if emo is None:
            continue
        emos.append((m.start(), m.end(), emo))

    intervals: list[tuple[int, int, float, float]] = []
    seen: set[tuple[int, float, float]] = set()
    for cre in _INTERVAL_RES[clock]:
        for m in cre.finditer(text):
            cleaned = _clean_interval(to_sec(m.group(1)), to_sec(m.group(2)))
            if cleaned is None:
                continue
            t0, t1 = cleaned
            key = (m.start(), round(t0, 3), round(t1, 3))
            if key in seen:
                continue
            seen.add(key)
            intervals.append((m.start(), m.end(), t0, t1))
    intervals.sort(key=lambda row: row[0])

    if not emos or not intervals:
        return []

    segs: list[dict[str, Any]] = []
    for ist, _ien, t0, t1 in intervals:
        chosen: str | None = None
        for est, een, emo in emos:
            if est <= ist:
                chosen = emo
            elif chosen is None and est - ist <= 48:
                chosen = emo
                break
        if chosen is None:
            continue
        segs.append({"start": t0, "end": t1, "emotion": chosen})
    segs.sort(key=lambda s: (s["start"], s["end"]))
    return segs
