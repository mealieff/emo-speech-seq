#!/usr/bin/env python3
"""Run the official IS2025 WavLM categorical baseline (seed 7) on a labelled MSP-Podcast split.

Streams audio from disk. Matches the baseline's 12 s cap, train-set waveform normalisation,
attentive-statistics pooling and the seed-7 checkpoints released at lab-msp.com
(``scripts/setup/download_models.sh`` puts them in ``WAVLM_IS2025_DIR``).

Writes to ``--out-dir``: ``cat_<split>_preds.csv`` (8-way scores per file, the input to
``scripts/evaluate/utterance_on_sequences.py``), ``cat_<split>_preds.json`` (per-file
predictions with gold) and ``cat_<split>.json`` (accuracy, macro / micro / per-class F1).

    python scripts/inference/wavlm_utterance.py --split test1
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

from emoseq import config

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

from emoseq.wavlm import WAVLM_ID, WAVLM_REVISION
from emoseq.wavlm import is2025_net as net

EMO_CAT = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
SPLIT_MAP = {"train": "Train", "dev": "Development", "test1": "Test1", "test2": "Test2"}
MAX_DUR_SEC = 12.0
SR = config.SR


def _load_pt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _read_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
    return np.asarray(wav, dtype=np.float32)


class StreamWavSet(Dataset):
    def __init__(self, names: list[str], labs, wav_mean: float, wav_std: float):
        self.names = list(names)
        self.labs = labs
        self.wav_mean = float(wav_mean)
        self.wav_std = float(wav_std)
        self.max_dur = int(MAX_DUR_SEC * SR)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        wav = _read_wav(config.MSP_AUDIO / name)[: self.max_dur]
        wav = (wav - self.wav_mean) / (self.wav_std + 1e-6)
        return wav, len(wav), self.labs[idx], name


def collate(batch):
    wavs, durs, labs, names = zip(*batch)
    wav_t = nn.utils.rnn.pad_sequence([torch.tensor(w) for w in wavs], batch_first=True)
    mask = torch.zeros(len(batch), int(max(durs)))
    for i, d in enumerate(durs):
        mask[i, :d] = 1
    labs_t = torch.tensor(np.asarray(labs), dtype=torch.float32)
    return wav_t, labs_t, mask, list(names)


def load_backbone(model_dir: Path, device: str):
    model = AutoModel.from_pretrained(WAVLM_ID, revision=WAVLM_REVISION)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    model.load_state_dict(_load_pt(model_dir / "final_ssl.pt"))
    return model.eval().to(device)


def load_heads(model_dir: Path, feat_dim: int, out_dim: int, device: str):
    pool = net.AttentiveStatisticsPooling(feat_dim)
    pool.load_state_dict(_load_pt(model_dir / "final_pool.pt"))
    ser = net.EmotionRegression(feat_dim * 2, 1024, 1, out_dim, dropout=0.5)
    ser.load_state_dict(_load_pt(model_dir / "final_ser.pt"))
    return pool.eval().to(device), ser.eval().to(device)


@torch.no_grad()
def predict(ssl, pool, ser, loader, device: str):
    preds, labs, names = [], [], []
    for x, y, mask, utt in tqdm(loader, desc="eval"):
        x = x.to(device, non_blocking=True).float()
        mask = mask.to(device, non_blocking=True).float()
        h = ssl(x, attention_mask=mask).last_hidden_state
        h = pool(h, mask)
        preds.append(ser(h).cpu())
        labs.append(y)
        names.extend(utt)
    return torch.cat(preds, 0).numpy(), torch.cat(labs, 0).numpy(), names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=list(SPLIT_MAP), default="test1")
    p.add_argument("--model-dir", default=str(config.WAVLM_IS2025_DIR))
    p.add_argument("--labels", default=str(config.WAVLM_LABELS_CSV),
                   help="processed_labels.csv from scripts/data/wavlm_labels.py")
    p.add_argument("--out-dir", default=str(config.RUNS_DIR / "test1" / "utterance" / "wavlm_is2025"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.labels)
    df = df[df["Split_Set"] == SPLIT_MAP[args.split]]
    names = df["FileName"].tolist()
    labs = df[EMO_CAT].to_numpy(dtype=np.float32)
    with open(model_dir / "train_norm_stat.pkl", "rb") as f:
        mean, std = pickle.load(f)
    loader = DataLoader(StreamWavSet(names, labs, mean, std), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
    ssl = load_backbone(model_dir, args.device)
    pool, ser = load_heads(model_dir, ssl.config.hidden_size, 8, args.device)
    pred, gold, utt = predict(ssl, pool, ser, loader, args.device)
    y_true, y_hat = gold.argmax(1), pred.argmax(1)

    with (out / f"cat_{args.split}_preds.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FileName", "pred", "pred_name", *EMO_CAT])
        for name, k, row in zip(utt, y_hat, pred):
            w.writerow([name, int(k), EMO_CAT[int(k)], *(f"{v:.5f}" for v in row.tolist())])
    recs = []
    for name, s, g in zip(utt, pred, gold):
        k, gk = int(np.argmax(s)), int(np.argmax(g))
        recs.append({"FileName": name, "gold": gk, "gold_name": EMO_CAT[gk],
                     "gold_dist": [round(float(v), 5) for v in g], "pred": k, "pred_name": EMO_CAT[k],
                     "correct": k == gk, "scores": {c: round(float(v), 5) for c, v in zip(EMO_CAT, s)}})
    (out / f"cat_{args.split}_preds.json").write_text(json.dumps(
        {"split": args.split, "labels": EMO_CAT, "model": str(model_dir), "n": len(recs), "records": recs},
        indent=2) + "\n")
    payload = {
        "task": "categorical",
        "split": args.split,
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_hat, average="micro", zero_division=0)),
        "accuracy": float((y_true == y_hat).mean()),
        "per_class_f1": {
            name: float(v)
            for name, v in zip(
                EMO_CAT,
                f1_score(y_true, y_hat, average=None, labels=list(range(len(EMO_CAT))), zero_division=0),
            )
        },
        "paper_test3": {"macro_f1": 0.329, "micro_f1": 0.355},
        "model": str(model_dir),
    }
    (out / f"cat_{args.split}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
