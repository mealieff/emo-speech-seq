"""Paths and machine settings, read from the environment.

Every value can be set as an environment variable or in ``configs/paths.env``
(copy ``configs/paths.env.example``). Environment variables win over the file;
relative paths are resolved against the repository root.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV_FILE = Path(os.environ.get("EMOSEQ_ENV_FILE", REPO / "configs" / "paths.env"))


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        value = re.sub(r"\$\{(\w+)\}|\$(\w+)", lambda m: os.environ.get(m.group(1) or m.group(2), ""), value)
        os.environ.setdefault(key, os.path.expanduser(value))


def _path(name: str, default: str | Path) -> Path:
    p = Path(os.path.expanduser(os.environ.get(name) or str(default)))
    return p if p.is_absolute() else (REPO / p).resolve()


_load_env_file(ENV_FILE)

# MSP-Podcast 1.12 release (Audios/, Labels/, Partitions.txt).
MSP_ROOT = _path("MSP_ROOT", REPO / "data" / "MSP-PODCAST-Publish-1.12")
MSP_AUDIO = MSP_ROOT / "Audios"
CONSENSUS_CSV = MSP_ROOT / "Labels" / "labels_consensus.csv"
DETAILED_CSV = MSP_ROOT / "Labels" / "labels_detailed.csv"
PARTITIONS = MSP_ROOT / "Partitions.txt"

# Everything this repo writes.
DATA_DIR = _path("EMOSEQ_DATA", REPO / "data")
RUNS_DIR = _path("EMOSEQ_RUNS", REPO / "runs")
CONCAT_DIR = _path("CONCAT_DIR", DATA_DIR / "sequences" / "audio")

UTT_DIR = DATA_DIR / "utterances"
SEQ_DIR = DATA_DIR / "sequences"
MANIFEST_DIR = SEQ_DIR / "manifests"
WAVLM_LABELS_CSV = DATA_DIR / "wavlm" / "processed_labels.csv"
DURATION_CACHE = DATA_DIR / "cache" / "segment_durations.json"

# Model weights. MODEL_CACHE is a Hugging Face hub cache (models--org--name/snapshots/...).
MODEL_CACHE = _path("MODEL_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
SALMONN_DIR = _path("SALMONN_DIR", MODEL_CACHE / "SALMONN-7B")
KIMI_AUDIO_ROOT = _path("KIMI_AUDIO_ROOT", REPO / "third_party" / "Kimi-Audio")
WAVLM_IS2025_DIR = _path("WAVLM_IS2025_DIR", MODEL_CACHE / "is2025" / "cat_ser" / "7")
WAVLM_TIMEAUG_CKPT = _path("WAVLM_TIMEAUG_CKPT", RUNS_DIR / "wavlm_timeaug" / "seed7" / "best.pt")

PROMPTS = REPO / "configs" / "prompts"
UTTERANCE_PROMPT = PROMPTS / "utterance.txt"
SEQUENCE_PROMPT = PROMPTS / "sequence.txt"

SR = 16000
SPLITS = ("train", "dev", "test1")

for _var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
    os.environ.setdefault(_var, str(MODEL_CACHE))


def utterance_jsonl(split: str) -> Path:
    return UTT_DIR / f"msp_podcast_{split}.jsonl"


def sequence_jsonl(split: str) -> Path:
    return SEQ_DIR / f"{split}.jsonl"


def manifest_jsonl(split: str) -> Path:
    return MANIFEST_DIR / f"{split}.jsonl"


if __name__ == "__main__":
    import shlex

    for _name in ("REPO", "MSP_ROOT", "DATA_DIR", "RUNS_DIR", "CONCAT_DIR", "UTT_DIR", "SEQ_DIR", "MANIFEST_DIR",
                  "MODEL_CACHE", "SALMONN_DIR", "KIMI_AUDIO_ROOT", "WAVLM_IS2025_DIR", "WAVLM_TIMEAUG_CKPT",
                  "UTTERANCE_PROMPT", "SEQUENCE_PROMPT"):
        print(f"{_name}={shlex.quote(str(globals()[_name]))}")
