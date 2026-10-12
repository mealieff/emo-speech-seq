"""Batch-invariance fixes for Qwen3-Omni audio in the pinned transformers build.

1. Encoder attention (``apply``). The encoder splits audio into windows (``cu_seqlens``,
   104 tokens = 8 s) that must not attend to each other. Only the FlashAttention path uses
   ``cu_seqlens``; under SDPA/eager every encoder layer gets ``attention_mask=None``, so each
   token attends across all windows of its clip and, in a batch, across other clips.
   Each layer now receives the block-diagonal mask over ``cu_seqlens``.
2. Feature mask (``fix_processor``). In a padded batch WhisperFeatureExtractor marks
   ceil(samples / hop) frames as valid, but a clip has floor(samples / hop) real frames, so
   every clip except the longest absorbs one padding frame (and sometimes an extra audio token).
   The mask is cut back to floor(samples / hop) per clip, matching batch size 1.

On GPU the model runs on FlashAttention-2 (``attention_backend``), the upstream windowed path, so
fix 1 is inactive there; it is what the CPU tests and an SDPA fallback rely on. Fix 2 applies to
every backend (still present upstream as of transformers 5.19).
"""
from __future__ import annotations

import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

_PATCHED = False
ORIGINAL_LAYER_FORWARD = m.Qwen3OmniMoeAudioEncoderLayer.forward


def _block_mask(hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    n = hidden_states.shape[0]
    mask = torch.full((1, 1, n, n), torch.finfo(hidden_states.dtype).min,
                      device=hidden_states.device, dtype=hidden_states.dtype)
    bounds = cu_seqlens.tolist()
    for a, b in zip(bounds[:-1], bounds[1:]):
        mask[..., a:b, a:b] = 0
    return mask


def attention_backend() -> str:
    if not torch.cuda.is_available():
        return "sdpa"
    from transformers.utils import is_flash_attn_2_available

    if not is_flash_attn_2_available():
        raise RuntimeError("Qwen3-Omni runs on FlashAttention-2 on GPU: install flash-attn (scripts/setup/install.sh)")
    return "flash_attention_2"


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    original = ORIGINAL_LAYER_FORWARD

    def forward(self, hidden_states, cu_seqlens, attention_mask=None, **kwargs):
        if attention_mask is None and not m.is_flash_attention_requested(self.self_attn.config):
            key = (hidden_states.shape[0], hidden_states.dtype, hidden_states.device, tuple(cu_seqlens.tolist()))
            cache = getattr(forward, "_cache", None)
            if cache is None or cache[0] != key:
                mask = _block_mask(hidden_states, cu_seqlens)
                forward._cache = cache = (key, mask)
            attention_mask = cache[1]
        return original(self, hidden_states, cu_seqlens, attention_mask=attention_mask, **kwargs)

    m.Qwen3OmniMoeAudioEncoderLayer.forward = forward
    _PATCHED = True


def fix_processor(processor):
    fe = processor.feature_extractor
    base = type(fe)
    if getattr(base, "_floor_frame_mask", False):
        return processor

    class FloorFrameMask(base):
        _floor_frame_mask = True

        def __call__(self, raw_speech, *args, **kwargs):
            out = super().__call__(raw_speech, *args, **kwargs)
            mask = out.get("attention_mask")
            if mask is not None:
                clips = [raw_speech] if getattr(raw_speech, "ndim", 0) == 1 else raw_speech
                for i, clip in enumerate(clips):
                    mask[i, len(clip) // self.hop_length:] = 0
            return out

    FloorFrameMask.__name__ = base.__name__
    fe.__class__ = FloorFrameMask
    return processor
