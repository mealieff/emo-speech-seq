#!/usr/bin/env python3
"""
Metrics + output parsing for the evaluation scripts (both tasks).

Label space (canonical order):
  0 Angry, 1 Sad, 2 Happy, 3 Surprised, 4 Fear, 5 Disgust, 6 Contempt, 7 Neutral

Parsing is deliberately strict for Task B: an output only counts as parsed if it
yields exactly `seq_len` recognized labels. Parse failures are returned/counted
separately and never silently coerced into the metrics.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

EMOTION_NAMES = ["Angry", "Sad", "Happy", "Surprised", "Fear", "Disgust", "Contempt", "Neutral"]
LABEL_TO_INT = {name.lower(): i for i, name in enumerate(EMOTION_NAMES)}
# Longest first so e.g. "Surprised" wins over any shorter prefix ambiguity.
_LABELS_BY_LEN = sorted(EMOTION_NAMES, key=len, reverse=True)
N_CLASSES = len(EMOTION_NAMES)
MAX_POSITIONS = 9
LENGTH_GROUPS = [("3-4", (3, 4)), ("5-6", (5, 6)), ("7-9", (7, 9))]


# ── normalization / parsing ──────────────────────────────────────────────────
def normalize_token(token: str) -> str:
    """Lowercase and strip surrounding non-alphabetic characters/punctuation."""
    return re.sub(r"[^a-z]", "", token.strip().lower())


def parse_task_a(text: str) -> Optional[int]:
    """Map a free-text Task A output to a label int, or None on failure."""
    whole = normalize_token(text)
    if whole in LABEL_TO_INT:
        return LABEL_TO_INT[whole]
    for tok in re.split(r"[\s,]+", text.strip()):
        norm = normalize_token(tok)
        if norm in LABEL_TO_INT:
            return LABEL_TO_INT[norm]
    return None


def parse_task_b(text: str, expected_n: int) -> Optional[list[int]]:
    """Parse a comma-separated Task B output into exactly `expected_n` label ints.

    Returns None (parse failure) if the count differs or any token is unknown.
    """
    parts = [normalize_token(p) for p in text.split(",")]
    parts = [p for p in parts if p]
    if len(parts) != expected_n:
        return None
    out: list[int] = []
    for p in parts:
        if p not in LABEL_TO_INT:
            return None
        out.append(LABEL_TO_INT[p])
    return out


def _skip_label_separators(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\n\r,;:.":
        i += 1
    return i


def extract_first_n_labels(text: str, n: int) -> Optional[list[int]]:
    """Greedy left-to-right scan for exactly *n* canonical emotion labels.

    Handles glued repetitions (``HappyHappyHappy``) and comma lists alike by
    repeatedly matching the longest label name at each position. Separators
    (whitespace, commas, punctuation) are skipped between labels.
    """
    if n <= 0:
        return []
    if not text or not text.strip():
        return None
    s = text.strip()
    out: list[int] = []
    i = 0
    while len(out) < n and i < len(s):
        i = _skip_label_separators(s, i)
        if i >= len(s):
            break
        matched = False
        for name in _LABELS_BY_LEN:
            end = i + len(name)
            if s[i:end].lower() == name.lower():
                out.append(LABEL_TO_INT[name.lower()])
                i = end
                matched = True
                break
        if not matched:
            i += 1
    return out if len(out) == n else None


def parse_task_a_first_n(text: str) -> Optional[int]:
    """Task A: first of 1 expected label from *text*."""
    seq = extract_first_n_labels(text, 1)
    return seq[0] if seq else None


def parse_task_b_first_n(text: str, expected_n: int) -> Optional[list[int]]:
    """Task B: first *expected_n* labels from *text*."""
    return extract_first_n_labels(text, expected_n)


def mentions_canonical_label(text: str) -> list[str]:
    """Return canonical emotion names that appear as whole words in *text*."""
    if not text or not text.strip():
        return []
    found: list[str] = []
    for name in EMOTION_NAMES:
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            found.append(name)
    return found


def classify_task_a_output(text: str, gt_label: int | None = None) -> dict:
    """Bucket a Task A raw output into parsed vs parse-failure subtypes."""
    parsed = parse_task_a(text)
    mentioned = mentions_canonical_label(text)
    gt_name = EMOTION_NAMES[gt_label] if gt_label is not None else None
    if parsed is not None:
        bucket = "parsed_ok"
        failure_mode = None
    elif mentioned:
        bucket = "parse_fail_names_emotion"
        failure_mode = diagnose_task_a_failure_mode(text)
    else:
        bucket = "parse_fail_no_emotion"
        failure_mode = diagnose_task_a_failure_mode(text)
    return {
        "parse_ok": parsed is not None,
        "bucket": bucket,
        "failure_mode": failure_mode,
        "parsed": parsed,
        "mentioned_labels": mentioned,
        "mentions_gt": gt_name in mentioned if gt_name else None,
        "raw_text": text,
    }


def diagnose_task_a_failure_mode(text: str) -> str:
    """Finer-grained Task A failure mode for outputs that failed parse_task_a.

    Invariant: if any whitespace/comma token normalizes to a canonical label,
    parse_task_a would have succeeded — so failures are never plain casing or
    trailing punctuation on an isolated label token.
    """
    if not text or not text.strip():
        return "empty"
    for tok in re.split(r"[\s,]+", text.strip()):
        if normalize_token(tok) in LABEL_TO_INT:
            return "parser_invariant_violation"
    mentioned = mentions_canonical_label(text)
    if mentioned:
        if re.search(r"\w[/\\|&]\w", text):
            return "glued_labels"
        if re.search(r"[^\s,][.!?;:][A-Za-z]", text):
            return "inline_punctuation_between_tokens"
        if len(text.split()) >= 4:
            return "sentence_mentions_label"
        return "label_mentioned_not_split_as_token"
    word_count = len(text.split())
    if word_count >= 4:
        return "full_sentence_no_canonical"
    if word_count >= 2:
        return "short_phrase_no_canonical"
    return "single_non_canonical_word"


def classify_task_b_output(text: str, expected_n: int, gt_labels: list[int] | None = None) -> dict:
    """Bucket a Task B raw output into parsed vs parse-failure subtypes."""
    parsed = parse_task_b(text, expected_n)
    mentioned = mentions_canonical_label(text)
    gt_names = [EMOTION_NAMES[i] for i in gt_labels] if gt_labels else []
    if parsed is not None:
        bucket = "parsed_ok"
    elif mentioned:
        bucket = "parse_fail_names_emotion"
    else:
        bucket = "parse_fail_no_emotion"
    parts = [normalize_token(p) for p in text.split(",")]
    parts = [p for p in parts if p]
    return {
        "parse_ok": parsed is not None,
        "bucket": bucket,
        "parsed": parsed,
        "mentioned_labels": mentioned,
        "mentions_any_gt": bool(set(mentioned) & set(gt_names)) if gt_names else None,
        "n_commas": text.count(","),
        "n_tokens": len(parts),
        "expected_n": expected_n,
        "raw_text": text,
    }


# ── shared metric helpers ────────────────────────────────────────────────────
def _round(v: float | None) -> float | None:
    return round(float(v), 4) if v is not None else None


def per_label_from_cm(cm: list[list[int]] | np.ndarray) -> dict[str, dict[str, float | None]]:
    """Per-class precision, recall, and F1 from a confusion matrix."""
    arr = np.asarray(cm, dtype=np.int64)
    out: dict[str, dict[str, float | None]] = {}
    for i, name in enumerate(EMOTION_NAMES):
        tp = int(arr[i, i])
        fn = int(arr[i, :].sum() - tp)
        fp = int(arr[:, i].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = None
        out[name] = {
            "precision": _round(prec),
            "recall": _round(rec),
            "f1": _round(f1),
        }
    return out


def scalar_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """WA, UA, macro/micro F1, confusion matrix, and per-label P/R/F1."""
    if not y_true:
        return {"n": 0}
    wa = accuracy_score(y_true, y_pred)
    ua = balanced_accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    return {
        "n": len(y_true),
        "WA": _round(wa),
        "UA": _round(ua),
        "macro_F1": _round(macro),
        "micro_F1": _round(micro),
        "confusion_matrix": cm.tolist(),
        "per_label": per_label_from_cm(cm),
        "labels": EMOTION_NAMES,
    }


def enrich_metrics(metrics: dict) -> dict:
    """Backfill derived fields on older eval JSON payloads."""
    if not metrics:
        return metrics
    if "confusion_matrix" in metrics and "per_label" not in metrics:
        cm = metrics["confusion_matrix"]
        metrics["per_label"] = per_label_from_cm(cm)
        if "micro_F1" not in metrics and metrics.get("n", 0):
            y_true, y_pred = _unpack_cm(cm)
            if y_true:
                metrics["micro_F1"] = _round(
                    f1_score(y_true, y_pred, average="micro", zero_division=0)
                )
    if "degradation_curve" in metrics and "length_group_counts" not in metrics:
        metrics["length_group_counts"] = seq_counts_from_degradation(
            metrics["degradation_curve"]
        )
    return metrics


def _unpack_cm(cm: list[list[int]]) -> tuple[list[int], list[int]]:
    """Reconstruct (y_true, y_pred) label lists from a confusion matrix."""
    y_true: list[int] = []
    y_pred: list[int] = []
    for ti, row in enumerate(cm):
        for pi, count in enumerate(row):
            y_true.extend([ti] * count)
            y_pred.extend([pi] * count)
    return y_true, y_pred


def seq_counts_from_degradation(degradation: list[dict]) -> dict[str, int]:
    """Infer parsed-sequence counts per length group from degradation-curve n's."""
    by_pos = {int(pt["position"]): int(pt["n"]) for pt in degradation}
    n_ge = {k: by_pos.get(k, 0) for k in range(1, MAX_POSITIONS + 2)}
    return {
        "3-4": max(n_ge[3] - n_ge[5], 0),
        "5-6": max(n_ge[5] - n_ge[7], 0),
        "7-9": max(n_ge[7], 0),
    }


# ── Task A metrics ───────────────────────────────────────────────────────────
def task_a_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return scalar_metrics(y_true, y_pred)


# ── Task B metrics ───────────────────────────────────────────────────────────
def task_b_metrics(
    true_seqs: list[list[int]],
    pred_seqs: list[Optional[list[int]]],
) -> dict:
    """Compute Task B metrics.

    `pred_seqs[i]` is None for parse failures (already aligned to true_seqs[i]).
    """
    total = len(true_seqs)
    n_fail = sum(1 for p in pred_seqs if p is None)
    n_ok = total - n_fail

    exact = sum(
        1 for t, p in zip(true_seqs, pred_seqs) if p is not None and p == t
    )

    pos_true: dict[int, list[int]] = {k: [] for k in range(MAX_POSITIONS)}
    pos_pred: dict[int, list[int]] = {k: [] for k in range(MAX_POSITIONS)}
    group_pairs: dict[str, tuple[list[int], list[int]]] = {g: ([], []) for g, _ in LENGTH_GROUPS}
    group_n_seqs: dict[str, int] = {g: 0 for g, _ in LENGTH_GROUPS}
    seq_len_counts: dict[str, int] = {}

    pooled_true: list[int] = []
    pooled_pred: list[int] = []

    for t, p in zip(true_seqs, pred_seqs):
        seq_len_counts[str(len(t))] = seq_len_counts.get(str(len(t)), 0) + 1
        if p is None:
            continue
        group = _length_group(len(t))
        if group is not None:
            group_n_seqs[group] += 1
        for k, (tl, pl) in enumerate(zip(t, p)):
            if k < MAX_POSITIONS:
                pos_true[k].append(tl)
                pos_pred[k].append(pl)
            pooled_true.append(tl)
            pooled_pred.append(pl)
            if group is not None:
                group_pairs[group][0].append(tl)
                group_pairs[group][1].append(pl)

    per_position_f1: dict[str, Optional[float]] = {}
    degradation: list[dict] = []
    for k in range(MAX_POSITIONS):
        if pos_true[k]:
            f1 = f1_score(
                pos_true[k], pos_pred[k], average="macro", zero_division=0,
                labels=list(range(N_CLASSES)),
            )
            val = _round(f1)
        else:
            val = None
        per_position_f1[f"pos_{k + 1}"] = val
        degradation.append({"position": k + 1, "macro_F1": val, "n": len(pos_true[k])})

    valid_f1s = [v for v in per_position_f1.values() if v is not None]
    avg_pos_f1 = _round(float(np.mean(valid_f1s))) if valid_f1s else None

    length_group_ua: dict[str, Optional[float]] = {}
    length_groups: dict[str, dict] = {}
    for g, _ in LENGTH_GROUPS:
        tl, pl = group_pairs[g]
        if tl:
            block = scalar_metrics(tl, pl)
            length_group_ua[g] = block["UA"]
            length_groups[g] = {
                "n_seqs": group_n_seqs[g],
                "n_pairs": block["n"],
                "WA": block["WA"],
                "UA": block["UA"],
                "macro_F1": block["macro_F1"],
                "micro_F1": block["micro_F1"],
                "per_label": block["per_label"],
            }
        else:
            length_group_ua[g] = None
            length_groups[g] = {
                "n_seqs": 0,
                "n_pairs": 0,
                "WA": None,
                "UA": None,
                "macro_F1": None,
                "micro_F1": None,
                "per_label": {name: {"precision": None, "recall": None, "f1": None}
                              for name in EMOTION_NAMES},
            }

    pooled = scalar_metrics(pooled_true, pooled_pred) if pooled_true else {"n": 0}

    return {
        "n": total,
        "parse_failures": n_fail,
        "parse_failure_rate": round(n_fail / total, 4) if total else 0.0,
        "n_parsed": n_ok,
        "exact_match": exact,
        "exact_match_rate": round(exact / total, 4) if total else 0.0,
        "avg_per_position_macro_F1": avg_pos_f1,
        "per_position_macro_F1": per_position_f1,
        "degradation_curve": degradation,
        "length_group_UA": length_group_ua,
        "length_group_counts": group_n_seqs,
        "seq_len_counts": seq_len_counts,
        "length_groups": length_groups,
        "WA": pooled.get("WA"),
        "UA": pooled.get("UA"),
        "macro_F1": pooled.get("macro_F1"),
        "micro_F1": pooled.get("micro_F1"),
        "confusion_matrix": pooled.get("confusion_matrix"),
        "per_label": pooled.get("per_label"),
        "labels": EMOTION_NAMES,
    }


def _length_group(n: int) -> Optional[str]:
    for name, (lo, hi) in LENGTH_GROUPS:
        if lo <= n <= hi:
            return name
    return None
