"""Pooled Phase 2 metrics: pool segment-level counts, then score.

Primary report is by quartile bin {2, 3, 4-6, 7+}. Overall, per-length,
micro-F1, boundary error, and per-emotion counts are additional.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from emoseq.bins import BIN_NAMES, bin_of_record
from emoseq.scoring.sequence import format_metrics, reduce_scores, score_sequence

PRIMARY_METRIC_KEYS = (
    "n",
    "n_gold",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "parse_rate",
    "mean_tiou",
    "mean_tiou_p",
    "tiou_f",
)
ADDITIONAL_METRIC_KEYS = (
    "n_correct",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "tf1",
    "boundary_error_sec",
    "mean_tiou_matched",
    "pooling",
    "per_emotion",
)


def pool_metrics(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = reduce_scores(score_rows)
    n_gold = sum(int(r.get("n_gold") or 0) for r in score_rows)
    tot_tp = 0
    for r in score_rows:
        tp = r.get("_tp") or {}
        tot_tp += sum(int(v) for v in tp.values())
    summary["accuracy"] = (tot_tp / n_gold) if n_gold else 0.0
    summary["n_gold"] = n_gold
    summary["n_correct"] = tot_tp
    summary["pooling"] = "segment_level_then_score"
    return summary


def scores_from_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    """Always rescore from gold/pred so tIoU picks up gold-indexed zeros.

    Packed ``_scores`` from older inference runs stored match-only tIoU.
    """
    gold = row.get("ground_truth") or row.get("segments") or []
    pred = row.get("pred") or []
    gold_norm = [{"start": float(s["start"]), "end": float(s["end"]), "emotion": s["emotion"]} for s in gold]
    dur = row.get("audio_duration")
    return score_sequence(gold_norm, pred, duration=float(dur) if dur else None)


def compact_primary(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: summary.get(k) for k in PRIMARY_METRIC_KEYS}


def compact_additional(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: summary.get(k) for k in ADDITIONAL_METRIC_KEYS}


def compact_full(summary: dict[str, Any]) -> dict[str, Any]:
    out = compact_primary(summary)
    out.update(compact_additional(summary))
    return out


def quartile_report(pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Primary block is by_quartile {2,3,4-6,7+}; everything else is additional."""
    by_bin: dict[str, list[dict[str, Any]]] = {b: [] for b in BIN_NAMES}
    by_len: dict[int, list[dict[str, Any]]] = defaultdict(list)
    packed: list[dict[str, Any]] = []
    for row in pred_rows:
        scores = scores_from_prediction_row(row)
        packed.append(scores)
        b = row.get("length_bin") or bin_of_record(row)
        by_bin.setdefault(b, []).append(scores)
        by_len[int(row.get("seq_len") or 0)].append(scores)
    pooled_bin = {b: pool_metrics(by_bin.get(b, [])) for b in BIN_NAMES}
    per_length = []
    for L in sorted(k for k in by_len if k > 0):
        rec = compact_full(pool_metrics(by_len[L]))
        rec["l"] = L
        per_length.append(rec)
    return {
        "primary": "by_quartile",
        "quartile_bins": list(BIN_NAMES),
        "by_quartile": {b: compact_primary(pooled_bin[b]) for b in BIN_NAMES},
        "additional": {
            "overall": compact_full(pool_metrics(packed)),
            "by_quartile_extra": {b: compact_additional(pooled_bin[b]) for b in BIN_NAMES},
            "per_length": per_length,
        },
    }


def format_quartile_table(report: dict[str, Any]) -> str:
    lines = [
        "primary: by_quartile {2, 3, 4-6, 7+}",
        f"{'bin':<6} {'n':>5} {'acc':>6} {'P':>6} {'R':>6} {'F1':>6} {'tIoU-R':>6} {'tIoU-P':>6} {'tIoU-F':>6} {'parse':>6}",
    ]
    for b in report.get("quartile_bins") or BIN_NAMES:
        m = (report.get("by_quartile") or {}).get(b) or {}
        lines.append(
            f"{b:<6} {int(m.get('n') or 0):>5} "
            f"{float(m.get('accuracy') or 0):>6.3f} "
            f"{float(m.get('macro_precision') or 0):>6.3f} "
            f"{float(m.get('macro_recall') or 0):>6.3f} "
            f"{float(m.get('macro_f1') or 0):>6.3f} "
            f"{float(m.get('mean_tiou') or 0):>6.3f} "
            f"{float(m.get('mean_tiou_p') or 0):>6.3f} "
            f"{float(m.get('tiou_f') or 0):>6.3f} "
            f"{float(m.get('parse_rate') or 0):>6.3f}"
        )
    ov = ((report.get("additional") or {}).get("overall")) or {}
    be = ov.get("boundary_error_sec")
    be_s = f"{float(be):.3f}" if be is not None else "na"
    lines.append(
        "additional overall  "
        f"n={int(ov.get('n') or 0)}  acc={float(ov.get('accuracy') or 0):.3f}  "
        f"F1={float(ov.get('macro_f1') or 0):.3f}  tIoU={float(ov.get('mean_tiou') or 0):.3f}  "
        f"parse={float(ov.get('parse_rate') or 0):.3f}  BE={be_s}"
    )
    return "\n".join(lines)


def format_phase2(summary: dict[str, Any]) -> str:
    extra = (
        f"acc={summary.get('accuracy', 0):.3f}  "
        f"P={summary.get('macro_precision', 0):.3f}  "
        f"R={summary.get('macro_recall', 0):.3f}  "
        f"macro-F1={summary.get('macro_f1', 0):.3f}"
    )
    return extra + "\n" + format_metrics(summary)
