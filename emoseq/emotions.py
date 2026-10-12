"""MSP-Podcast EmoClass codes used in the gold sequence spans."""
from __future__ import annotations

from typing import Any

CODE_TO_NAME = {
    "A": "Angry",
    "S": "Sad",
    "H": "Happy",
    "U": "Surprised",
    "F": "Fear",
    "D": "Disgust",
    "C": "Contempt",
    "N": "Neutral",
}
EXCLUDED_EMO_CODES = frozenset({"X", "O"})


def emotion_name(code: object) -> str | None:
    raw = str(code or "").strip()
    if not raw or raw.upper() in EXCLUDED_EMO_CODES:
        return None
    if raw in CODE_TO_NAME:
        return CODE_TO_NAME[raw]
    key = raw.lower()
    for name in CODE_TO_NAME.values():
        if name.lower() == key:
            return name
    return None


def is_excluded_emotion(token: object) -> bool:
    t = str(token or "").strip()
    if not t:
        return False
    if t.upper() in EXCLUDED_EMO_CODES:
        return True
    return t.lower() in {"other", "no agreement", "noagreement"}


def record_has_excluded_emotion(rec: dict[str, Any]) -> bool:
    for seg in rec.get("segments") or rec.get("ground_truth") or []:
        if not isinstance(seg, dict):
            continue
        if is_excluded_emotion(seg.get("emotion_code")) or is_excluded_emotion(seg.get("emotion")):
            return True
    return False
