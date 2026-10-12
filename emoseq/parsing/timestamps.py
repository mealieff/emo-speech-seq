"""Timestamp grammar shared by every sequence-parser path.

Accepted: seconds ``ss.ff`` (any digits), ``m:ss[.ff]``, ``h:mm:ss[.ff]``.
Clock fields after the first must be exactly two digits and below 60.
Anything else (``1:5``, ``1:75``, ``1:2:03``) does not parse. ``clock=False``
is the seconds-only grammar used for the strict-format parse rate.
"""
from __future__ import annotations

import re

SEC = r"[0-9]+(?:\.[0-9]+)?"
CLOCK = r"[0-9]+:[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?|[0-9]+:[0-5][0-9](?:\.[0-9]+)?"
# A token may not run on into more digits / clock fields ("1" out of "1:5").
_END = r"(?![0-9:]|\.[0-9])"

_CLOCK_FULL = re.compile(rf"(?:{CLOCK})")


def ts_group(clock: bool = True) -> str:
    """One capturing group matching a single timestamp token."""
    return rf"((?:{CLOCK}|{SEC}){_END})" if clock else rf"({SEC}{_END})"


def to_sec(value, clock: bool = True) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().strip("'\"")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if not clock or not _CLOCK_FULL.fullmatch(s):
        return None
    parts = s.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
