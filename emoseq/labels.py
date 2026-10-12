"""Shared emotion names and Trainer constants for sequence ALM SFT.

Emotion names and integer ids match MSP-Podcast ``EmoClass``
(A S H U F D C N → 0..7). ``IGNORE_INDEX`` is the HuggingFace Trainer
convention (``label_pad_token_id = -100``).
"""

from __future__ import annotations

# MSP-Podcast consensus classes, same order as scripts/preprocess.py LABEL_MAP.
EMOTION_NAMES: list[str] = [
    "Angry",
    "Sad",
    "Happy",
    "Surprised",
    "Fear",
    "Disgust",
    "Contempt",
    "Neutral",
]

EMOTION_TO_ID: dict[str, int] = {n: i for i, n in enumerate(EMOTION_NAMES)}

# Canonical lookup is case-insensitive; training targets use the Title-case names.
# Charlotte labels_chain.csv uses MSP readme names (Anger/Sadness/Happiness/Surprise).
EMOTION_LOOKUP: dict[str, str] = {n.lower(): n for n in EMOTION_NAMES}
EMOTION_LOOKUP.update(
    {
        "anger": "Angry",
        "sadness": "Sad",
        "happiness": "Happy",
        "surprise": "Surprised",
    }
)

# MSP consensus: X = no annotator agreement, O = Other. Never gold in train/test.
EXCLUDED_EMO_CODES: frozenset[str] = frozenset({"X", "O"})
EXCLUDED_EMO_NAMES: frozenset[str] = frozenset({"No agreement", "Other"})

IGNORE_INDEX: int = -100

# VibeVoice target keys were Start / End / Speaker / Content.
# E2 drops transcript and speaker from the supervised string.
TARGET_KEYS: tuple[str, str, str] = ("Start", "End", "Emotion")

LORA_PEFT_KEYS: tuple[str, ...] = (
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
    "task_type",
)

VIBEVOICE_MLP_MODULES: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")
E1_ATTN_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def is_excluded_emotion(token: object) -> bool:
    """True for consensus codes X/O and their name forms."""
    t = str(token or "").strip()
    if not t:
        return False
    if t.upper() in EXCLUDED_EMO_CODES:
        return True
    key = t.lower().replace("_", " ")
    if key in {"x", "o", "other", "no agreement", "noagreement"}:
        return True
    return EMOTION_LOOKUP.get(key) in EXCLUDED_EMO_NAMES


def record_has_excluded_emotion(rec: dict) -> bool:
    for seg in rec.get("segments") or rec.get("ground_truth") or rec.get("gold") or []:
        if not isinstance(seg, dict):
            continue
        if is_excluded_emotion(seg.get("emotion_code")) or is_excluded_emotion(seg.get("emotion")):
            return True
    labels = rec.get("emo_labels")
    if isinstance(labels, str) and any(is_excluded_emotion(x) for x in labels.split(";")):
        return True
    return False
