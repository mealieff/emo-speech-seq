"""Concatenated-audio dataset for WavLM-TimeAug."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from emoseq.bins import BIN_NAMES
from emoseq.config import MSP_AUDIO as WAV_DIR
from emoseq.io import load_jsonl

SR = 16000
MAX_AUDIO_SEC = 100.0
N_EMO = 8


def read_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR).astype(np.float32)
    return wav


def concat_files(names: list[str]) -> np.ndarray:
    parts = [read_wav(WAV_DIR / n) for n in names]
    return np.concatenate(parts, axis=0) if parts else np.zeros(1, dtype=np.float32)


def crop_indices(durations: list[float], max_sec: float, rng: random.Random) -> tuple[int, int]:
    n = len(durations)
    if n <= 0:
        return 0, 0
    if sum(durations) <= max_sec + 1e-6:
        return 0, n
    cands: list[tuple[int, int]] = []
    for i in range(n):
        acc = 0.0
        for j in range(i, n):
            acc += durations[j]
            if acc > max_sec:
                break
            if j - i + 1 >= 2:
                cands.append((i, j + 1))
            elif j - i + 1 == 1:
                cands.append((i, j + 1))
    if not cands:
        return 0, n
    return cands[rng.randrange(len(cands))]


def slice_record(rec: dict, i0: int, i1: int) -> dict:
    durs = rec["durations"][i0:i1]
    t0 = sum(rec["durations"][:i0])
    gold = [[round(a - t0, 6), round(b - t0, 6)] for a, b in rec["gold_intervals"][i0:i1]]
    return {
        **{k: rec[k] for k in rec if k not in {
            "file_names", "gold_intervals", "primary_labels", "emotion_codes",
            "vote_dists", "durations", "audio_duration", "loss_mask", "unique_ids",
            "seq_len", "bin", "length_bin",
        }},
        "file_names": rec["file_names"][i0:i1],
        "gold_intervals": gold,
        "primary_labels": rec["primary_labels"][i0:i1],
        "emotion_codes": rec["emotion_codes"][i0:i1],
        "vote_dists": rec["vote_dists"][i0:i1],
        "durations": durs,
        "audio_duration": round(sum(durs), 6),
        "loss_mask": rec["loss_mask"][i0:i1],
        "unique_ids": rec["unique_ids"][i0:i1],
        "seq_len": i1 - i0,
        "bin": rec.get("bin") or rec.get("length_bin"),
        "length_bin": rec.get("length_bin") or rec.get("bin"),
        "seq_id": rec["seq_id"],
    }


class SeqDataset(Dataset):
    def __init__(
        self,
        manifest: Path | None,
        wav_mean: float,
        wav_std: float,
        crop_sec: float | None = None,
        seed: int = 7,
        rows: list[dict] | None = None,
    ):
        if rows is not None:
            self.rows = list(rows)
        else:
            self.rows = load_jsonl(manifest)
        self.wav_mean = float(wav_mean)
        self.wav_std = float(wav_std)
        self.crop_sec = crop_sec
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        rec = self.rows[idx]
        if self.crop_sec is not None:
            i0, i1 = crop_indices(rec["durations"], self.crop_sec, self.rng)
            rec = slice_record(rec, i0, i1)
        wav = concat_files(rec["file_names"])
        wav = (wav - self.wav_mean) / (self.wav_std + 1e-6)
        votes = np.asarray(rec["vote_dists"], dtype=np.float32)
        if votes.ndim != 2:
            votes = votes.reshape(-1, N_EMO)
        return {
            "wav": torch.from_numpy(np.ascontiguousarray(wav)),
            "votes": torch.from_numpy(votes),
            "intervals": [(float(a), float(b)) for a, b in rec["gold_intervals"]],
            "duration": float(rec["audio_duration"]),
            "bin": rec.get("length_bin") or rec.get("bin"),
            "rec": rec,
        }


class BinBalancedSampler(Sampler[int]):
    def __init__(self, rows: list[dict], seed: int = 7):
        self.by_bin: dict[str, list[int]] = {b: [] for b in BIN_NAMES}
        for i, r in enumerate(rows):
            b = r.get("length_bin") or r.get("bin")
            self.by_bin.setdefault(b, []).append(i)
        self.bins = [b for b in BIN_NAMES if self.by_bin.get(b)]
        self.n = len(rows)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        bags = {b: self.by_bin[b][:] for b in self.bins}
        for b in bags:
            self.rng.shuffle(bags[b])
        ptr = {b: 0 for b in self.bins}
        out = []
        i = 0
        while len(out) < self.n:
            b = self.bins[i % len(self.bins)]
            if ptr[b] >= len(bags[b]):
                bags[b] = self.by_bin[b][:]
                self.rng.shuffle(bags[b])
                ptr[b] = 0
            out.append(bags[b][ptr[b]])
            ptr[b] += 1
            i += 1
        yield from out


def class_inv_weights(rows: list[dict]) -> torch.Tensor:
    counts = np.zeros(N_EMO, dtype=np.float64)
    for r in rows:
        for lab, dist in zip(r["primary_labels"], r["vote_dists"]):
            d = np.asarray(dist, dtype=np.float64)
            if d.sum() > 0:
                counts += d
            else:
                idx = ["Angry", "Sad", "Happy", "Surprised", "Fear", "Disgust", "Contempt", "Neutral"].index(lab)
                counts[idx] += 1.0
    counts = np.maximum(counts, 1.0)
    inv = counts.sum() / (N_EMO * counts)
    return torch.tensor(inv, dtype=torch.float32)


def reweight_votes(votes: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    x = votes * inv.unsqueeze(0)
    s = x.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return x / s


def stratified_subset(rows: list[dict], n: int = 32, seed: int = 7) -> list[dict]:
    """n sequences, as even across {2,3,4-6,7+} as possible."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r.get("length_bin") or r.get("bin")].append(r)
    rng = random.Random(seed)
    bins = [b for b in BIN_NAMES if by.get(b)]
    if not bins:
        return rows[:n]
    per = max(1, n // len(bins))
    out: list[dict] = []
    for b in bins:
        bag = by[b][:]
        rng.shuffle(bag)
        out.extend(bag[:per])
    rng.shuffle(out)
    return out[:n]
