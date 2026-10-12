"""WavLM-TimeAug: sliding-window WavLM + timestamp BiLSTM + segment ASP/BiLSTM."""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
FRAME_HZ = 50
FRAME_HOP = SR // FRAME_HZ  # 320
WIN_SEC = 10.0
OVERLAP_SEC = 2.0
STEP_SEC = WIN_SEC - OVERLAP_SEC  # 8 s
EDGE_SEC = OVERLAP_SEC / 2.0  # 1 s
WIN_SAMPLES = int(WIN_SEC * SR)
STEP_SAMPLES = int(STEP_SEC * SR)
BOUND_HZ = 10
POOL_K = FRAME_HZ // BOUND_HZ  # 5
FEAT_DIM = 1024
N_BOTTOM_FROZEN = 12
N_EMO = 8
BOUNDARY_PRIOR = 0.02
TARGET_SIGMA = 0.25
TARGET_SIGMA_FINAL = 0.15


class FrameASP(nn.Module):
    """Attentive statistics pooling over encoder frames (not wav-sample masks)."""

    def __init__(self, input_size: int = FEAT_DIM):
        super().__init__()
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.empty(input_size, 1))
        nn.init.normal_(self.attention, mean=0.0, std=1.0)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or x.size(0) == 0:
            return x.new_zeros(x.size(-1) * 2)
        h = torch.tanh(self.sap_linear(x.unsqueeze(0)))
        w = torch.matmul(h, self.attention).squeeze(2)
        w = torch.softmax(w, dim=1).view(1, x.size(0), 1)
        xx = x.unsqueeze(0)
        mu = torch.sum(xx * w, dim=1)
        rh = torch.sqrt((torch.sum((xx**2) * w, dim=1) - mu**2).clamp(min=1e-5))
        return torch.cat((mu, rh), 1).squeeze(0)


class WavLMSeq(nn.Module):
    def __init__(self, ssl: nn.Module, asp: FrameASP, boundary_prior: float = BOUNDARY_PRIOR):
        super().__init__()
        self.ssl = ssl
        self.asp = asp
        hid = 256
        n_hidden = int(ssl.config.num_hidden_layers) + 1
        self.layer_weights = nn.Parameter(torch.zeros(n_hidden))
        self.bound_lstm = nn.LSTM(
            input_size=FEAT_DIM,
            hidden_size=hid,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.bound_head = nn.Linear(hid * 2, 1)
        self.seg_lstm = nn.LSTM(
            input_size=FEAT_DIM * 2,
            hidden_size=hid,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.emo_head = nn.Linear(hid * 2, N_EMO)
        init_boundary_bias(self.bound_head, boundary_prior)

    def mix_layers(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """SUPERB-style softmax mix over embeddings + encoder layers. (B, T, D)."""
        w = torch.softmax(self.layer_weights, dim=0)
        if len(hidden_states) != w.numel():
            raise RuntimeError(f"hidden_states={len(hidden_states)} layer_weights={w.numel()}")
        stacked = torch.stack(list(hidden_states), dim=0)
        return (w.view(-1, 1, 1, 1) * stacked).sum(dim=0)

    def _ssl_window(self, chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones(1, chunk.numel(), device=chunk.device, dtype=torch.long)
        out = self.ssl(chunk.unsqueeze(0), attention_mask=mask, output_hidden_states=True)
        last = out.last_hidden_state[0]
        mixed = self.mix_layers(out.hidden_states)[0]
        return last, mixed

    def encode_frames(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wav: (T,) → (F_emo, F_bound) each (T_f, 1024) at 50 Hz.

        Emotion uses the last WavLM layer. The boundary branch uses a learned
        weighted sum over all layers.
        """
        n = int(wav.numel())
        if n <= 0:
            z = wav.new_zeros(1, FEAT_DIM)
            return z, z
        duration = n / float(SR)
        if n <= WIN_SAMPLES:
            last, mixed = self._ssl_window(wav)
            return last, mixed

        starts = list(range(0, max(1, n - WIN_SAMPLES + 1), STEP_SAMPLES))
        last_s = n - WIN_SAMPLES
        if starts[-1] != last_s:
            starts.append(last_s)

        emo_parts, bound_parts = [], []
        cursor = 0.0
        n_win = len(starts)
        for i, s in enumerate(starts):
            last, mixed = self._ssl_window(wav[s : s + WIN_SAMPLES])
            n_f = last.size(0)
            start_sec = s / float(SR)
            t = start_sec + (torch.arange(n_f, device=last.device, dtype=last.dtype) + 0.5) / FRAME_HZ
            keep_end = duration if i == n_win - 1 else min(duration, start_sec + WIN_SEC - EDGE_SEC)
            sel = (t >= cursor) & (t < keep_end + 1e-6)
            if sel.any():
                emo_parts.append(last[sel])
                bound_parts.append(mixed[sel])
                cursor = float(t[sel][-1].item() + (1.0 / FRAME_HZ))
            elif i == n_win - 1 and emo_parts:
                emo_parts.append(last[-1:])
                bound_parts.append(mixed[-1:])
        if not emo_parts:
            last, mixed = self._ssl_window(wav[:WIN_SAMPLES] if n > WIN_SAMPLES else wav)
            return last, mixed
        return torch.cat(emo_parts, dim=0), torch.cat(bound_parts, dim=0)

    def boundary_logits(self, frames: torch.Tensor) -> torch.Tensor:
        pooled = avg_pool_to_10hz(frames)
        out, _ = self.bound_lstm(pooled.unsqueeze(0))
        return self.bound_head(out).squeeze(0).squeeze(-1)

    def emotion_logits(self, frames: torch.Tensor, intervals: list[tuple[float, float]]) -> torch.Tensor:
        hs = []
        t_max = frames.size(0) / float(FRAME_HZ)
        for on, off in intervals:
            i0, i1 = span_to_frames(on, off, frames.size(0), t_max)
            hs.append(self.asp.forward_one(frames[i0:i1]))
        h = torch.stack(hs, dim=0).unsqueeze(0)
        out, _ = self.seg_lstm(h)
        return self.emo_head(out).squeeze(0)


def init_boundary_bias(head: nn.Linear, pi: float = BOUNDARY_PRIOR) -> None:
    pi = min(max(float(pi), 1e-4), 1.0 - 1e-4)
    nn.init.constant_(head.bias, math.log(pi / (1.0 - pi)))


def heatmap_focal(
    logits: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalty-reduced focal loss (CenterNet) for Gaussian heatmap targets."""
    logits = logits.float()
    y = y.float()
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    pos = (y >= 0.99).float()
    neg = 1.0 - pos
    pos_loss = -((1.0 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1.0 - y) ** beta) * (p ** alpha) * torch.log(1.0 - p) * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def avg_pool_to_10hz(frames: torch.Tensor) -> torch.Tensor:
    t, d = frames.shape
    n = (t // POOL_K) * POOL_K
    if n <= 0:
        return frames.mean(dim=0, keepdim=True)
    x = frames[:n].view(n // POOL_K, POOL_K, d).mean(dim=1)
    if t > n:
        tail = frames[n:].mean(dim=0, keepdim=True)
        x = torch.cat([x, tail], dim=0)
    return x


def span_to_frames(on: float, off: float, n_frames: int, t_max: float | None = None) -> tuple[int, int]:
    dur = (n_frames / float(FRAME_HZ)) if t_max is None else float(t_max)
    i0 = int(max(0.0, min(dur, on)) * FRAME_HZ)
    i1 = int(max(0.0, min(dur, off)) * FRAME_HZ)
    i0 = min(max(0, i0), n_frames - 1)
    i1 = min(max(i0 + 1, i1), n_frames)
    return i0, i1


def freeze_bottom_layers(ssl: nn.Module, n_bottom: int = N_BOTTOM_FROZEN) -> None:
    if hasattr(ssl, "freeze_feature_encoder"):
        ssl.freeze_feature_encoder()
    layers = ssl.encoder.layers
    for i, layer in enumerate(layers):
        req = i >= n_bottom
        for p in layer.parameters():
            p.requires_grad = req


def load_norm_stat(path: Path) -> tuple[float, float]:
    mean, std = pickle.load(open(path, "rb"))
    return float(mean), float(std)


def gaussian_boundary_target(
    n_10hz: int,
    internal_bounds: list[float],
    sigma: float = TARGET_SIGMA,
) -> torch.Tensor:
    """Gaussian heatmap; nearest 10 Hz frame is snapped to 1 so focal positives exist."""
    t = (torch.arange(n_10hz, dtype=torch.float32) + 0.5) / BOUND_HZ
    g = torch.zeros(n_10hz, dtype=torch.float32)
    for b in internal_bounds:
        b = float(b)
        g = torch.maximum(g, torch.exp(-0.5 * ((t - b) / sigma) ** 2))
        i = int(round(b * BOUND_HZ - 0.5))
        i = min(max(i, 0), n_10hz - 1)
        g[i] = 1.0
    return g.clamp(0.0, 1.0)


def anneal_sigma(epoch: int, epochs: int, start: float = TARGET_SIGMA, end: float = TARGET_SIGMA_FINAL) -> float:
    if epochs <= 1:
        return float(end)
    frac = min(1.0, max(0.0, (epoch - 1) / (epochs - 1)))
    return float(start + (end - start) * frac)
