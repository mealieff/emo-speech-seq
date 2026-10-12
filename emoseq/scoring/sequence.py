"""Sequence metrics: closest-span F1, TIoU, boundary error, transition F1."""
from __future__ import annotations

from typing import Any

from emoseq.parsing.sequence import parse_prediction

EMOTIONS = ("Angry", "Sad", "Happy", "Surprised", "Fear", "Disgust", "Contempt", "Neutral")
TF1_COLLAR_SEC = 0.50


def _safe(num: float, den: float) -> float:
    return num / den if den else 0.0


def interval_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    inter = max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))
    union = max(float(a["end"]), float(b["end"])) - min(float(a["start"]), float(b["start"]))
    return _safe(inter, union)


def midpoint(seg: dict[str, Any]) -> float:
    return 0.5 * (float(seg["start"]) + float(seg["end"]))


def match_closest(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
) -> list[tuple[int, int, float]]:
    """One-to-one greedy match: highest IoU, then nearest midpoint."""
    pairs: list[tuple[float, float, int, int]] = []
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            iou = interval_iou(g, p)
            dist = abs(midpoint(g) - midpoint(p))
            pairs.append((-iou, dist, i, j))
    pairs.sort()
    used_g: set[int] = set()
    used_p: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for neg_iou, _dist, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append((i, j, -neg_iou))
    return matched


def transitions(segs: list[dict[str, Any]]) -> list[float]:
    """Internal boundaries: end of each span except the last, chronological."""
    ordered = sorted(segs, key=lambda s: (float(s["start"]), float(s["end"])))
    return [float(s["end"]) for s in ordered[:-1]] if len(ordered) >= 2 else []


def match_points(gold_t: list[float], pred_t: list[float], collar: float) -> tuple[int, int, int]:
    g = sorted(gold_t)
    p = sorted(pred_t)
    used: set[int] = set()
    hits = 0
    for gt in g:
        best_j, best_d = -1, collar + 1.0
        for j, pt in enumerate(p):
            if j in used:
                continue
            d = abs(pt - gt)
            if d <= collar and d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used.add(best_j)
            hits += 1
    return hits, len(g), len(p)


def _gold_indexed_tious(row: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Full-population gold-span IoUs, plus the matched-subset list.

    Unparsed rows and unmatched gold spans are 0. Old packed scores stored
    match-only ``_tious``; those are padded to ``n_gold``.
    """
    n_gold = int(row.get("n_gold") or 0)
    parsed = int(row.get("parsed") or 0)
    raw = list(row.get("_tious") or [])
    matched = list(row.get("_tious_matched") or [])
    if not matched and parsed:
        matched = list(raw)
    if n_gold <= 0:
        return raw, matched
    if not parsed:
        return [0.0] * n_gold, matched
    if len(raw) < n_gold:
        raw = raw + [0.0] * (n_gold - len(raw))
    return raw, matched


def clip_spans(pred: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """Clip spans to [0, duration]; a span wholly past the end becomes zero-length."""
    out = []
    for p in pred:
        s = min(max(float(p["start"]), 0.0), duration)
        e = min(max(float(p["end"]), 0.0), duration)
        out.append({"start": s, "end": max(s, e), "emotion": p["emotion"]})
    return out


def score_sequence(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    *,
    collar: float = TF1_COLLAR_SEC,
    duration: float | None = None,
) -> dict[str, Any]:
    """Closest-span F1 plus gold-indexed (recall-side) and prediction-indexed
    (precision-side) tIoU.

    ``mean_tiou`` (recall side) averages IoU over every gold span. Unparsed outputs
    (empty ``pred``) and unmatched gold spans contribute 0. Extra unmatched
    predictions are FPs for F1 and do not enter its denominator.
    ``mean_tiou_p`` (precision side) clips predictions to [0, duration] (default: the
    last gold end), matches them to gold with the same one-to-one rule, and averages
    IoU over every predicted span; unmatched predictions, including spans wholly past
    the end, contribute 0.
    ``mean_tiou_matched`` is the old match-only mean, kept as a diagnostic.
    """
    matched = match_closest(gold, pred)
    if duration is None:
        duration = max((float(g["end"]) for g in gold), default=0.0)
    tiou_p_sum = sum(iou for _, _, iou in match_closest(gold, clip_spans(pred, duration)))
    tp = {e: 0 for e in EMOTIONS}
    fp = {e: 0 for e in EMOTIONS}
    fn = {e: 0 for e in EMOTIONS}
    used_g = {i for i, _, _ in matched}
    used_p = {j for _, j, _ in matched}
    # Gold-indexed tIoU: every gold span contributes its match IoU, or 0 if
    # unparsed / unmatched. Matched-subset mean is retained separately.
    iou_by_gold = [0.0] * len(gold)
    matched_ious: list[float] = []
    for i, j, iou in matched:
        g, p = gold[i], pred[j]
        iou_by_gold[i] = iou
        matched_ious.append(iou)
        if g["emotion"] == p["emotion"] and g["emotion"] in tp:
            tp[g["emotion"]] += 1
        else:
            if g["emotion"] in fn:
                fn[g["emotion"]] += 1
            if p["emotion"] in fp:
                fp[p["emotion"]] += 1
    for i, g in enumerate(gold):
        if i not in used_g and g["emotion"] in fn:
            fn[g["emotion"]] += 1
    for j, p in enumerate(pred):
        if j not in used_p and p["emotion"] in fp:
            fp[p["emotion"]] += 1

    per: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    precs: list[float] = []
    recs: list[float] = []
    for e in EMOTIONS:
        prec = _safe(tp[e], tp[e] + fp[e])
        rec = _safe(tp[e], tp[e] + fn[e])
        f1 = _safe(2 * prec * rec, prec + rec)
        per[e] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp[e], "fp": fp[e], "fn": fn[e]}
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    tot_tp = sum(tp[e] for e in EMOTIONS)
    tot_fp = sum(fp[e] for e in EMOTIONS)
    tot_fn = sum(fn[e] for e in EMOTIONS)
    micro_p = _safe(tot_tp, tot_tp + tot_fp)
    micro_r = _safe(tot_tp, tot_tp + tot_fn)
    micro_f = _safe(2 * micro_p * micro_r, micro_p + micro_r)

    g_tr = transitions(gold)
    p_tr = transitions(pred)
    hits, n_g, n_p = match_points(g_tr, p_tr, collar)
    tf1 = _safe(2 * _safe(hits, n_p) * _safe(hits, n_g), _safe(hits, n_p) + _safe(hits, n_g))
    be_errs: list[float] = []
    if g_tr and p_tr:
        gs, ps = sorted(g_tr), sorted(p_tr)
        n_align = min(len(gs), len(ps))
        be_errs = [abs(ps[k] - gs[k]) for k in range(n_align)]

    return {
        "parsed": 1.0 if pred else 0.0,
        "n_gold": len(gold),
        "n_pred": len(pred),
        "n_matched": len(matched),
        "macro_precision": sum(precs) / len(EMOTIONS),
        "macro_recall": sum(recs) / len(EMOTIONS),
        "macro_f1": sum(f1s) / len(EMOTIONS),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f,
        "mean_tiou": (sum(iou_by_gold) / len(iou_by_gold)) if iou_by_gold else 0.0,
        "mean_tiou_p": _safe(tiou_p_sum, len(pred)),
        "mean_tiou_matched": (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0,
        "boundary_error_sec": (sum(be_errs) / len(be_errs)) if be_errs else None,
        "n_boundary": len(be_errs),
        "_tiou_p_sum": tiou_p_sum,
        "tf1": tf1,
        "tf1_collar_sec": collar,
        "n_gold_transitions": n_g,
        "n_pred_transitions": n_p,
        "n_tf1_hits": hits,
        "per_emotion": per,
        "_tious": iou_by_gold,
        "_tious_matched": matched_ious,
        "_be_errs": be_errs,
        "_tp": tp,
        "_fp": fp,
        "_fn": fn,
        "_hits": hits,
        "_n_g_tr": n_g,
        "_n_p_tr": n_p,
    }


PACKED_KEYS = frozenset({"_tp", "_fp", "_fn", "_tious", "_tious_matched", "_tiou_p_sum", "_be_errs",
                         "_hits", "_n_g_tr", "_n_p_tr"})


def pack_scores(scores: dict[str, Any]) -> dict[str, Any]:
    """The part of ``score_sequence`` output that ``reduce_scores`` needs to pool rows."""
    return {k: v for k, v in scores.items() if not k.startswith("_") or k in PACKED_KEYS}


def reduce_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "parse_rate": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "micro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "mean_tiou": 0.0,
            "mean_tiou_p": 0.0,
            "tiou_f": 0.0,
            "mean_tiou_matched": 0.0,
            "boundary_error_sec": None,
            "tf1": 0.0,
        }
    tp = {e: 0 for e in EMOTIONS}
    fp = {e: 0 for e in EMOTIONS}
    fn = {e: 0 for e in EMOTIONS}
    tious: list[float] = []
    matched_tious: list[float] = []
    be: list[float] = []
    hits = n_g = n_p = 0
    parsed = 0
    tiou_p_sum = 0.0
    n_pred = 0
    for r in rows:
        parsed += int(r.get("parsed") or 0)
        tiou_p_sum += float(r.get("_tiou_p_sum") or 0.0)
        n_pred += int(r.get("n_pred") or 0)
        gold_t, match_t = _gold_indexed_tious(r)
        tious.extend(gold_t)
        matched_tious.extend(match_t)
        be.extend(r.get("_be_errs") or [])
        hits += int(r.get("_hits") or 0)
        n_g += int(r.get("_n_g_tr") or 0)
        n_p += int(r.get("_n_p_tr") or 0)
        for e in EMOTIONS:
            tp[e] += int((r.get("_tp") or {}).get(e, 0))
            fp[e] += int((r.get("_fp") or {}).get(e, 0))
            fn[e] += int((r.get("_fn") or {}).get(e, 0))
    per = {}
    f1s = []
    precs = []
    recs = []
    for e in EMOTIONS:
        prec = _safe(tp[e], tp[e] + fp[e])
        rec = _safe(tp[e], tp[e] + fn[e])
        f1 = _safe(2 * prec * rec, prec + rec)
        per[e] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp[e], "fp": fp[e], "fn": fn[e]}
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)
    tot_tp = sum(tp[e] for e in EMOTIONS)
    tot_fp = sum(fp[e] for e in EMOTIONS)
    tot_fn = sum(fn[e] for e in EMOTIONS)
    micro_p = _safe(tot_tp, tot_tp + tot_fp)
    micro_r = _safe(tot_tp, tot_tp + tot_fn)
    micro_f = _safe(2 * micro_p * micro_r, micro_p + micro_r)
    tf1 = _safe(2 * _safe(hits, n_p) * _safe(hits, n_g), _safe(hits, n_p) + _safe(hits, n_g))
    tiou_r = sum(tious) / len(tious) if tious else 0.0
    tiou_p = _safe(tiou_p_sum, n_pred)
    return {
        "n": len(rows),
        "mean_tiou_p": tiou_p,
        "tiou_f": _safe(2 * tiou_r * tiou_p, tiou_r + tiou_p),
        "n_pred": n_pred,
        "parse_rate": parsed / len(rows),
        "macro_precision": sum(precs) / len(EMOTIONS),
        "macro_recall": sum(recs) / len(EMOTIONS),
        "macro_f1": sum(f1s) / len(EMOTIONS),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f,
        "mean_tiou": sum(tious) / len(tious) if tious else 0.0,
        "mean_tiou_matched": sum(matched_tious) / len(matched_tious) if matched_tious else 0.0,
        "boundary_error_sec": (sum(be) / len(be)) if be else None,
        "tf1": tf1,
        "tf1_collar_sec": TF1_COLLAR_SEC,
        "per_emotion": per,
    }


def format_metrics(summary: dict[str, Any]) -> str:
    lines = [
        f"n={summary.get('n')} parse={summary.get('parse_rate', 0):.3f}",
        (
            f"macro P/R/F1={summary.get('macro_precision', 0):.3f}/"
            f"{summary.get('macro_recall', 0):.3f}/{summary.get('macro_f1', 0):.3f}  "
            f"micro P/R/F1={summary.get('micro_precision', 0):.3f}/"
            f"{summary.get('micro_recall', 0):.3f}/{summary.get('micro_f1', 0):.3f}"
        ),
        (
            f"TIoU={summary.get('mean_tiou', 0):.3f}  "
            f"BE={summary.get('boundary_error_sec')}  TF1={summary.get('tf1', 0):.3f}"
        ),
        "per-label P/R/F1:",
    ]
    for e, row in (summary.get("per_emotion") or {}).items():
        lines.append(
            f"  {e:10s}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
            f"F1={row['f1']:.3f}  tp={int(row['tp'])} fp={int(row['fp'])} fn={int(row['fn'])}"
        )
    return "\n".join(lines)


def gold_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"start": float(s["start"]), "end": float(s["end"]), "emotion": s["emotion"]}
        for s in record["segments"]
    ]


def score_text(pred_text: str, record: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred = parse_prediction(pred_text)
    gold = gold_segments(record)
    return pred, score_sequence(gold, pred)
