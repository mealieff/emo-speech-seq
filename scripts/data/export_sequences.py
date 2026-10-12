#!/usr/bin/env python3
"""Write the concatenated sequence WAVs and the sequence JSONL used by the audio LMs.

Reads ``data/sequences/manifests/{train,dev,test1}.jsonl`` and writes, per split:
16 kHz PCM-16 WAVs under ``CONCAT_DIR/<split>/`` and ``data/sequences/<split>.jsonl``
(``audio_path`` + gold ``segments``). Also writes ``train_af3.jsonl`` / ``dev_af3.jsonl``,
the <=180 s subsets used to fine-tune Audio Flamingo 3.

    python scripts/data/export_sequences.py
    python scripts/data/export_sequences.py --limit 8   # smoke test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from emoseq import config
from emoseq.emotions import emotion_name
from emoseq.wavlm.dataset import SR, concat_files

MANIFESTS = config.MANIFEST_DIR
SFT_DIR = config.SEQ_DIR
CONCAT_DIR = config.CONCAT_DIR
SPLITS = ("train", "dev", "test1")
AF3_TRAIN_MAX_AUDIO_SEC = 180.0
CANON = {
    "Surprise": "Surprised",
    "surprised": "Surprised",
    "surprise": "Surprised",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest-dir", default=str(MANIFESTS))
    p.add_argument("--out-dir", default=str(SFT_DIR))
    p.add_argument("--concat-dir", default=str(CONCAT_DIR))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    return p.parse_args()


def canon_emotion(raw: object) -> str:
    name = emotion_name(raw) or str(raw).strip()
    return CANON.get(name, name)


def to_sft_record(rec: dict, wav_path: Path) -> dict:
    labels = rec.get("primary_labels") or []
    codes = rec.get("emotion_codes") or []
    segments = []
    for i, (t0, t1) in enumerate(rec["gold_intervals"]):
        code = codes[i] if i < len(codes) else None
        emo = canon_emotion(labels[i] if i < len(labels) else code)
        segments.append(
            {
                "start": round(float(t0), 2),
                "end": round(float(t1), 2),
                "emotion": emo,
                "emotion_code": code,
            }
        )
    return {
        "seq_id": rec["seq_id"],
        "partition": rec.get("partition"),
        "seq_len": rec["seq_len"],
        "length_bin": rec.get("length_bin") or rec.get("bin"),
        "bin": rec.get("bin") or rec.get("length_bin"),
        "podcast_id": rec.get("podcast_id"),
        "unique_ids": rec.get("unique_ids"),
        "file_names": rec["file_names"],
        "audio_path": str(wav_path),
        "audio_duration": rec["audio_duration"],
        "segments": segments,
        "audio_source": "podcast_concat",
    }


def export_split(manifest: Path, out_jsonl: Path, concat_dir: Path, *, limit: int | None, skip_existing: bool) -> int:
    concat_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with manifest.open(encoding="utf-8") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            wav_path = concat_dir / f"{rec['seq_id']}.wav"
            if not (skip_existing and wav_path.is_file()):
                wav = concat_files(rec["file_names"])
                sf.write(str(wav_path), np.asarray(wav, dtype=np.float32), SR, subtype="PCM_16")
            dst.write(json.dumps(to_sft_record(rec, wav_path), ensure_ascii=False) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
            if n % 500 == 0:
                print(f"  {manifest.name} {n}", flush=True)
    return n


def filter_af3(src: Path, dst: Path, max_sec: float) -> tuple[int, int]:
    kept = dropped = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            if float(rec.get("audio_duration") or 0) > max_sec:
                dropped += 1
                continue
            out.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    return kept, dropped


def main() -> None:
    args = parse_args()
    man_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    concat_dir = Path(args.concat_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = man_dir / f"{split}.jsonl"
        if not src.is_file():
            raise SystemExit(f"missing {src}; run python scripts/data/sequence_manifests.py")
        n = export_split(
            src,
            out_dir / f"{split}.jsonl",
            concat_dir / split,
            limit=args.limit,
            skip_existing=args.skip_existing,
        )
        print(f"{split}: {n} → {out_dir / f'{split}.jsonl'}", flush=True)
    for split in ("train", "dev"):
        kept, dropped = filter_af3(
            out_dir / f"{split}.jsonl",
            out_dir / f"{split}_af3.jsonl",
            AF3_TRAIN_MAX_AUDIO_SEC,
        )
        print(f"{split} AF3 ≤{AF3_TRAIN_MAX_AUDIO_SEC:.0f}s: kept={kept} dropped={dropped}", flush=True)


if __name__ == "__main__":
    main()
