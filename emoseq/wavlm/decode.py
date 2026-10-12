"""Peak picking and span decode for WavLM-TimeAug."""
from __future__ import annotations

import numpy as np
import torch

from emoseq.scoring.sequence import match_points, transitions
from emoseq.wavlm.model import BOUND_HZ, N_EMO

E5_NAMES = ["Angry", "Sad", "Happy", "Surprised", "Fear", "Disgust", "Contempt", "Neutral"]
MIN_SEG_SEC = 2.0
SMOOTH_SIGMA_FRAMES = 1.0
THETA_GRID = tuple(round(x, 3) for x in np.arange(0.01, 0.31, 0.01))


def _smooth(x: np.ndarray, sigma: float = SMOOTH_SIGMA_FRAMES) -> np.ndarray:
    if x.size < 3 or sigma <= 0:
        return x
    radius = max(1, int(3 * sigma))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    kern = np.exp(-0.5 * (k / sigma) ** 2)
    kern /= kern.sum()
    return np.convolve(x, kern, mode="same")


def _probs(logits: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(logits):
        return torch.sigmoid(logits.detach().float().cpu()).numpy()
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def local_maxima(logits: torch.Tensor | np.ndarray, duration: float) -> tuple[list[float], list[float]]:
    """Smoothed local maxima: times (s) and scores in (0, 1)."""
    s = _smooth(np.asarray(_probs(logits), dtype=np.float64))
    times, scores = [], []
    for i in range(1, len(s) - 1):
        if s[i] >= s[i - 1] and s[i] > s[i + 1]:
            t = (i + 0.5) / BOUND_HZ
            if 0.0 < t < duration:
                times.append(float(t))
                scores.append(float(s[i]))
    return times, scores


def peak_pick(logits: torch.Tensor | np.ndarray, theta: float, duration: float) -> list[float]:
    """Internal boundaries (seconds) from 10 Hz boundary logits. Start/end are given."""
    times, scores = local_maxima(logits, duration)
    peaks = [t for t, v in zip(times, scores) if v >= theta]
    return _enforce_min_gap(peaks, duration, MIN_SEG_SEC)


def boundary_ap(
    peak_times: list[float],
    peak_scores: list[float],
    gold_times: list[float],
    collar: float = 0.5,
) -> float:
    """PASCAL-style AP: mean precision at hits, missed golds count as 0. / n_gold."""
    if not gold_times:
        return 1.0 if not peak_times else 0.0
    if not peak_times:
        return 0.0
    order = np.argsort(-np.asarray(peak_scores, dtype=np.float64))
    matched: set[int] = set()
    hits: list[int] = []
    for i in order:
        t = peak_times[int(i)]
        best_j, best_d = -1, collar + 1.0
        for j, g in enumerate(gold_times):
            if j in matched:
                continue
            d = abs(t - g)
            if d <= collar and d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            matched.add(best_j)
            hits.append(1)
        else:
            hits.append(0)
    tp = 0
    prec_at_hit = []
    for k, h in enumerate(hits, 1):
        if h:
            tp += 1
            prec_at_hit.append(tp / k)
    return float(sum(prec_at_hit) / len(gold_times))


def sweep_theta(
    items: list[tuple[object, float, list[float]]],
    collar: float = 0.5,
    grid: tuple[float, ...] = THETA_GRID,
) -> dict:
    """Pick θ maximizing mean boundary F1 @ collar. ``items``: (logits, duration, gold internals)."""
    best_t, best_f1 = 0.05, -1.0
    by_t = {}
    for theta in grid:
        f1s = []
        for logits, dur, gold in items:
            peaks = peak_pick(logits, float(theta), dur)
            spans = bounds_to_intervals(peaks, dur)
            gold_iv = gold_intervals_from_internals(gold, dur)
            f1s.append(boundary_f1(gold_iv, spans, collar=collar))
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        by_t[str(theta)] = mean_f1
        if mean_f1 > best_f1:
            best_f1, best_t = mean_f1, float(theta)
    return {"best_theta": best_t, "best_f1": best_f1, "by_theta": by_t}


def _enforce_min_gap(peaks: list[float], duration: float, min_len: float) -> list[float]:
    bounds = [0.0] + sorted(peaks) + [float(duration)]
    changed = True
    while changed and len(bounds) > 2:
        changed = False
        gaps = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        j = int(np.argmin(gaps))
        if gaps[j] >= min_len - 1e-9:
            break
        if j == 0:
            del bounds[1]
        elif j == len(gaps) - 1:
            del bounds[-2]
        elif gaps[j] <= gaps[j - 1]:
            del bounds[j + 1]
        else:
            del bounds[j]
        changed = True
    return bounds[1:-1]


def bounds_to_intervals(internal: list[float], duration: float) -> list[tuple[float, float]]:
    edges = [0.0] + [float(b) for b in internal if 0.0 < b < duration] + [float(duration)]
    edges = sorted(edges)
    out = []
    for a, b in zip(edges, edges[1:]):
        if b > a + 1e-4:
            out.append((float(a), float(b)))
    if not out:
        out = [(0.0, float(duration))]
    return out


def gold_intervals_from_internals(internals: list[float], duration: float) -> list[tuple[float, float]]:
    return bounds_to_intervals(list(internals), duration)


def boundary_f1(gold_intervals, pred_intervals, collar: float = 0.5) -> float:
    gold = [{"start": a, "end": b, "emotion": "Neutral"} for a, b in gold_intervals]
    pred = [{"start": a, "end": b, "emotion": "Neutral"} for a, b in pred_intervals]
    hits, n_g, n_p = match_points(transitions(gold), transitions(pred), collar)
    if n_g + n_p == 0:
        return 1.0
    p = hits / n_p if n_p else 0.0
    r = hits / n_g if n_g else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def logits_to_dist(logits: torch.Tensor) -> np.ndarray:
    p = torch.softmax(logits.detach().float(), dim=-1).cpu().numpy()
    return p


def to_pred_segments(
    intervals: list[tuple[float, float]],
    logits: torch.Tensor,
) -> list[dict]:
    dist = logits_to_dist(logits)
    if dist.ndim == 1:
        dist = dist[None, :]
    segs = []
    k = min(len(intervals), dist.shape[0])
    for i in range(k):
        on, off = intervals[i]
        d = dist[i].tolist()
        if len(d) < N_EMO:
            d = d + [0.0] * (N_EMO - len(d))
        lab = E5_NAMES[int(np.argmax(d[:N_EMO]))]
        segs.append({"start": float(on), "end": float(off), "emotion": lab, "dist": d[:N_EMO]})
    return segs


def gold_to_e5(rec: dict) -> list[dict]:
    segs = []
    for (on, off), lab in zip(rec["gold_intervals"], rec["primary_labels"]):
        segs.append({"start": float(on), "end": float(off), "emotion": lab})
    return segs
