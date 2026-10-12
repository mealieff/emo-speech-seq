"""Phase 2 length bins: {2}, {3}, {4-6}, {7+}."""
from __future__ import annotations

from typing import Any

BIN_NAMES = ("2", "3", "4-6", "7+")


def length_bin(n: int) -> str:
    n = int(n)
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if 4 <= n <= 6:
        return "4-6"
    if n >= 7:
        return "7+"
    raise ValueError(f"length {n} is below the Phase 2 minimum of 2")


def clip_keys(rec: dict[str, Any]) -> set[tuple[str, int]]:
    """(podcast_id, unique_id) pairs in a sequence. Overlap unit for leak checks."""
    pod = str(rec["podcast_id"])
    ids = rec.get("unique_ids")
    if ids:
        return {(pod, int(u)) for u in ids}
    return {(pod, int(seg["unique_id"])) for seg in rec.get("segments") or [] if "unique_id" in seg}


def bin_of_record(rec: dict[str, Any]) -> str:
    n = rec.get("seq_len")
    if n is None:
        n = rec.get("n_unique_ids")
    return length_bin(int(n))


def sklearn_balanced_bin_weights(rows: list[dict[str, Any]], bins: tuple[str, ...] = BIN_NAMES) -> dict[str, float]:
    """sklearn `compute_class_weight('balanced')` over length bins.

    w_b = n / (n_bins_present * n_b). Same formula as Section 3's
    VAD-to-category SVM `class_weight='balanced'`.
    """
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight

    y = np.array([r.get("length_bin") or bin_of_record(r) for r in rows], dtype=object)
    present = [b for b in bins if np.any(y == b)]
    if not present:
        return {}
    weights = compute_class_weight("balanced", classes=np.array(present, dtype=object), y=y)
    return {str(b): float(w) for b, w in zip(present, weights)}


def apply_bin_weights(rows: list[dict[str, Any]], weight_map: dict[str, float]) -> list[dict[str, Any]]:
    for rec in rows:
        b = rec.get("length_bin") or bin_of_record(rec)
        rec["bin_weight"] = float(weight_map[b])
    return rows


def canonicalize_model(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key in {"af3", "audio_flamingo3", "audioflamingo3"}:
        return "audio_flamingo3"
    if key in {"qwen2", "qwen2_audio", "qwen2audio"}:
        return "qwen2_audio"
    if key in {"qwen3", "qwen3_omni", "qwen3omni"}:
        return "qwen3_omni"
    raise ValueError(f"unknown model {name!r}")
