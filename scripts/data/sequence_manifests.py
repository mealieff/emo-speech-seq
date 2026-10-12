#!/usr/bin/env python3
"""Build the sequence manifests: runs of consecutive MSP-Podcast segments, 8 classes, 2 <= L <= 40.

A sequence is a run of strictly consecutive unique IDs from the same podcast within one
partition (Train / Development / Test1). O and X segments are excluded and split runs; runs
longer than 40 are cut into chunks of 40. Gold span boundaries are the cumulative clip
durations. Test3 is omitted (its IDs are packed challenge IDs, so runs cannot be rebuilt).

    python scripts/data/sequence_manifests.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from emoseq import config
from emoseq.bins import length_bin
from emoseq.emotions import emotion_name
from emoseq.io import write_json, write_jsonl
from emoseq.names import parse_podcast_name, unique_id_runs

POD_AUDIO = config.MSP_AUDIO
CONSENSUS = config.CONSENSUS_CSV
DETAILED = config.DETAILED_CSV
OUT_DIR = config.MANIFEST_DIR
DUR_CACHE = config.DURATION_CACHE

SPLITS = ("Train", "Development", "Test1")
SPLIT_SLUG = {"Train": "train", "Development": "dev", "test1": "test1", "Test1": "test1"}
KEEP_CODES = frozenset({"A", "S", "H", "U", "F", "D", "C", "N"})
EMO_CAT = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
VOTE_ALIASES = {
    "angry": "Angry",
    "sad": "Sad",
    "happy": "Happy",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "fear": "Fear",
    "disgust": "Disgust",
    "contempt": "Contempt",
    "neutral": "Neutral",
}
MAX_LEN = 40
MIN_LEN = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--max-skip", type=int, default=0)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    return p.parse_args()


def load_durations(names: list[str]) -> dict[str, float]:
    cache: dict[str, float] = {}
    if DUR_CACHE.is_file():
        cache = json.loads(DUR_CACHE.read_text(encoding="utf-8"))
    missing = [n for n in names if n not in cache]
    print(f"durations cached={len(cache)} missing={len(missing)}", flush=True)
    if missing:
        import wave

        for i, fn in enumerate(missing, 1):
            with wave.open(str(POD_AUDIO / fn), "rb") as w:
                cache[fn] = w.getnframes() / float(w.getframerate())
            if i % 10000 == 0:
                print(f"  durations {i}/{len(missing)}", flush=True)
        DUR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DUR_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def load_vote_dists() -> dict[str, list[float]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    with DETAILED.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parts = row["EmoDetail"].split(";")
            prim = parts[1].strip() if len(parts) > 1 else ""
            mapped = VOTE_ALIASES.get(prim.lower())
            if mapped:
                counts[row["FileName"]][mapped] += 1
    out: dict[str, list[float]] = {}
    for fn, c in counts.items():
        tot = float(sum(c[k] for k in EMO_CAT))
        if tot <= 0:
            out[fn] = [0.0] * 8
        else:
            out[fn] = [c[k] / tot for k in EMO_CAT]
    return out


def pick_file(candidates: list[tuple[tuple[int, ...], str]]) -> str:
    candidates.sort(key=lambda x: (len(x[0]), x[0], x[1]))
    return candidates[0][1]


def gold_intervals(durs: list[float]) -> list[list[float]]:
    out = []
    t = 0.0
    for d in durs:
        out.append([round(t, 6), round(t + d, 6)])
        t += d
    return out


def one_hot(code: str) -> list[float]:
    name = emotion_name(code)
    # sequence label "Surprised" vs IS2025 "Surprise"
    key = "Surprise" if name == "Surprised" else name
    vec = [0.0] * 8
    if key in EMO_CAT:
        vec[EMO_CAT.index(key)] = 1.0
    return vec


def chunk_run(run: list[int], max_len: int) -> list[list[int]]:
    if len(run) <= max_len:
        return [run]
    parts = []
    for i in range(0, len(run), max_len):
        part = run[i : i + max_len]
        if len(part) >= MIN_LEN:
            parts.append(part)
    return parts


def bin_table(rows: list[dict]) -> dict:
    by = Counter(r["bin"] for r in rows)
    lens = [int(r["seq_len"]) for r in rows]
    durs = [float(r["audio_duration"]) for r in rows]
    by_len = Counter(lens)
    return {
        "2": by.get("2", 0),
        "3": by.get("3", 0),
        "4-6": by.get("4-6", 0),
        "7+": by.get("7+", 0),
        "total": len(rows),
        "max_length": max(lens) if lens else 0,
        "n_segments": sum(lens),
        "n_gt_100s": sum(1 for d in durs if d > 100.0),
        "max_audio_sec": max(durs) if durs else 0.0,
        "by_length": {str(k): by_len[k] for k in sorted(by_len)},
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    max_len = int(args.max_len)
    print("loading consensus (8-class, drop X/O)", flush=True)
    files: list[dict] = []
    n_xo = 0
    with CONSENSUS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row["FileName"]
            split = row["Split_Set"]
            if split not in SPLITS:
                continue
            code = (row.get("EmoClass") or "").strip()
            if code not in KEEP_CODES:
                n_xo += 1
                continue
            try:
                pod, suffixes, uid = parse_podcast_name(fn)
            except ValueError:
                continue
            files.append(
                {
                    "file_name": fn,
                    "split": split,
                    "podcast_id": pod,
                    "unique_id": int(uid),
                    "suffixes": suffixes,
                    "EmoClass": code,
                    "SpkrID": row.get("SpkrID"),
                }
            )
    print(f"kept={len(files)} dropped_xo={n_xo}", flush=True)
    print("loading vote dists", flush=True)
    votes = load_vote_dists()
    durs = load_durations([r["file_name"] for r in files])

    by_split_pod: dict[str, dict[str, dict[int, list[tuple[tuple[int, ...], str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    meta = {r["file_name"]: r for r in files}
    for r in files:
        by_split_pod[r["split"]][r["podcast_id"]][r["unique_id"]].append((r["suffixes"], r["file_name"]))

    n_dup = 0
    n_chunked = 0
    manifests: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for split in SPLITS:
        for pod, id_map in sorted(by_split_pod[split].items()):
            chosen: dict[int, str] = {}
            for uid, cands in id_map.items():
                if len(cands) > 1:
                    n_dup += 1
                chosen[uid] = pick_file(cands)
            for run in unique_id_runs(list(chosen), max_skip=int(args.max_skip)):
                if len(run) < MIN_LEN:
                    continue
                parts = chunk_run(run, max_len)
                if len(parts) > 1:
                    n_chunked += 1
                for part in parts:
                    names = [chosen[u] for u in part]
                    recs = [meta[n] for n in names]
                    durations = [float(durs[n]) for n in names]
                    codes = [r["EmoClass"] for r in recs]
                    labels = [emotion_name(c) for c in codes]
                    dists = []
                    for n, code in zip(names, codes):
                        dist = votes.get(n)
                        if dist is None or sum(dist) <= 0:
                            dist = one_hot(code)
                        dists.append(dist)
                    seq_len = len(part)
                    seq_id = f"{split.lower()}_{pod}_{part[0]:04d}_{part[-1]:04d}_L{seq_len}"
                    manifests[split].append(
                        {
                            "seq_id": seq_id,
                            "partition": split,
                            "bin": length_bin(seq_len),
                            "length_bin": length_bin(seq_len),
                            "seq_len": seq_len,
                            "podcast_id": pod,
                            "unique_ids": part,
                            "file_names": names,
                            "gold_intervals": gold_intervals(durations),
                            "primary_labels": labels,
                            "emotion_codes": codes,
                            "vote_dists": dists,
                            "durations": durations,
                            "audio_duration": round(sum(durations), 6),
                            "loss_mask": [True] * seq_len,
                            "SpkrIDs": [r["SpkrID"] for r in recs],
                            "max_skip": int(args.max_skip),
                            "max_len": max_len,
                            "audio_source": "podcast_concat",
                            "exclude_emotions": ["X", "O"],
                        }
                    )
        manifests[split].sort(key=lambda r: (r["podcast_id"], r["unique_ids"][0], r["seq_id"]))

    summary = {
        "population": "8class_strict",
        "exclude_emotions": ["X", "O"],
        "max_skip": int(args.max_skip),
        "max_len": max_len,
        "min_len": MIN_LEN,
        "audio_source": "podcast_concat",
        "splits": SPLITS,
        "n_duplicate_unique_ids_collapsed": n_dup,
        "n_runs_chunked_at_max_len": n_chunked,
        "n_xo_dropped_files": n_xo,
        "test3": "omitted; packed challenge IDs, not original unique-ID runs",
        "by_split": {SPLIT_SLUG[s]: bin_table(manifests[s]) for s in SPLITS},
        "manifests": {SPLIT_SLUG[s]: str(out_dir / f"{SPLIT_SLUG[s]}.jsonl") for s in SPLITS},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        dest = out_dir / f"{SPLIT_SLUG[split]}.jsonl"
        write_jsonl(dest, manifests[split])
        print(f"wrote {dest} n={len(manifests[split])}", flush=True)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_split"}, indent=2))
    print(json.dumps(summary["by_split"], indent=2))


if __name__ == "__main__":
    main()
