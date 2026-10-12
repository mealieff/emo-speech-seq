"""Even time vs emotion token mass on SFT assistant targets.

Target lines look like ``(12.34, 56.78, Angry)``. Uniform token-mean CE
gives timestamps more gradient because they tokenize into more pieces
than emotion words. Per example, time tokens share mass 0.5 and emotion
tokens share mass 0.5.

Punctuation defaults to mass 0. That lets Qwen2 drift from gold tuples
to pretrained dict-style output. ``punct_floor`` (total mass, split
evenly across punct tokens) keeps time:emotion at 1:1 while giving
``( ) ,`` a non-zero format signal. Loss reduction is
``(w * ce).sum() / w.sum()``, so a floor of 0.1 is ~9% of the example
after normalization.
"""
from __future__ import annotations

import re
from typing import Any

import torch

SPAN_RE = re.compile(
    r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([A-Za-z]+)\s*\)"
)
KIND_TIME = "time"
KIND_EMOTION = "emotion"
KIND_PUNCT = "punct"
# Extra mass shared by format tokens when even-span CE is on. 0 reproduces
# the collapsed Qwen2 mask. 0.1 is the first recovery attempt.
DEFAULT_PUNCT_FLOOR = 0.1


def tag_target_chars(text: str) -> list[str]:
    tags = [KIND_PUNCT] * len(text)
    for m in SPAN_RE.finditer(text):
        for group in (1, 2):
            for i in range(m.start(group), m.end(group)):
                tags[i] = KIND_TIME
        for i in range(m.start(3), m.end(3)):
            tags[i] = KIND_EMOTION
    return tags


def kinds_for_target_tokens(tokenizer, text: str) -> tuple[list[int], list[str]]:
    """Return (token_ids, kinds) for ``text`` with no special tokens."""
    kwargs: dict[str, Any] = {"add_special_tokens": False}
    # Fast tokenizers expose character offsets; slow ones fall back below.
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, **kwargs)
        offsets = encoded.get("offset_mapping")
    except Exception:
        encoded = tokenizer(text, **kwargs)
        offsets = None
    ids = list(encoded["input_ids"])
    char_tags = tag_target_chars(text)
    kinds: list[str] = []
    if offsets is not None:
        for start, end in offsets:
            slice_tags = char_tags[start:end] if end > start else []
            if any(t == KIND_EMOTION for t in slice_tags):
                kinds.append(KIND_EMOTION)
            elif any(t == KIND_TIME for t in slice_tags):
                kinds.append(KIND_TIME)
            else:
                kinds.append(KIND_PUNCT)
        return ids, kinds
    pieces = tokenizer.convert_ids_to_tokens(ids)
    for piece in pieces:
        raw = str(piece).lstrip("Ġ▁")
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw) or raw == ".":
            kinds.append(KIND_TIME)
        elif re.fullmatch(r"[A-Za-z]+", raw):
            kinds.append(KIND_EMOTION)
        else:
            kinds.append(KIND_PUNCT)
    return ids, kinds


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def even_mass_for_labels(
    tokenizer,
    labels_row: torch.Tensor,
    target_text: str,
    ignore_index: int,
    punct_floor: float = 0.0,
) -> torch.Tensor:
    """Weight tensor aligned with ``labels_row``. Time mass 0.5, emotion mass 0.5.

    ``punct_floor`` is extra total mass split equally across punctuation
    tokens. Time:emotion stays 1:1. ``punct_floor=0`` zeros format tokens.
    """
    weights = torch.zeros(labels_row.shape[0], dtype=torch.float32)
    labeled = (labels_row != ignore_index).nonzero(as_tuple=False).view(-1)
    if labeled.numel() == 0:
        return weights
    assistant_ids = [int(x) for x in labels_row[labeled].tolist()]
    target_ids, kinds = kinds_for_target_tokens(tokenizer, target_text)
    start = _find_subsequence(assistant_ids, target_ids)
    if start is None:
        # Fallback: uniform over labeled assistant tokens so training still runs.
        weights[labeled] = 1.0 / float(labeled.numel())
        return weights
    n_time = sum(1 for k in kinds if k == KIND_TIME)
    n_emo = sum(1 for k in kinds if k == KIND_EMOTION)
    n_punct = sum(1 for k in kinds if k == KIND_PUNCT)
    time_w = (0.5 / n_time) if n_time else 0.0
    emo_w = (0.5 / n_emo) if n_emo else 0.0
    if n_time == 0 and n_emo == 0:
        weights[labeled] = 1.0 / float(labeled.numel())
        return weights
    if n_time == 0:
        emo_w = 1.0 / n_emo
    if n_emo == 0:
        time_w = 1.0 / n_time
    floor = max(float(punct_floor), 0.0)
    punct_w = (floor / n_punct) if floor > 0.0 and n_punct else 0.0
    for j, kind in enumerate(kinds):
        pos = int(labeled[start + j])
        if kind == KIND_TIME:
            weights[pos] = time_w
        elif kind == KIND_EMOTION:
            weights[pos] = emo_w
        elif kind == KIND_PUNCT and punct_w:
            weights[pos] = punct_w
    return weights
