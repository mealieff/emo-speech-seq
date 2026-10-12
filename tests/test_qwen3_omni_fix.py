"""Batch invariance of Qwen3-Omni with emoseq.models.qwen3_omni_fix, on a tiny random-weight thinker (CPU, fp32).

The real-weight version (bf16, GPU, MSP-Podcast clips) is scripts/evaluate/qwen3_omni_batch_check.py.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

m = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
from transformers import WhisperFeatureExtractor  # noqa: E402
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig  # noqa: E402

from emoseq.models import qwen3_omni_fix  # noqa: E402

AUDIO_START, AUDIO_TOKEN, AUDIO_END, PAD = 151669, 151675, 151670, 151643
FRAMES = [1250, 430, 905]  # 12.5 s spans two 8 s windows; none is a multiple of the 100-frame chunk
TOL = 1e-3


def tiny_thinker():
    torch.manual_seed(0)
    cfg = Qwen3OmniMoeThinkerConfig(
        audio_config=dict(d_model=32, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=64,
                          num_mel_bins=128, downsample_hidden_size=16, output_dim=32, n_window=50,
                          n_window_infer=800, max_source_positions=1500),
        text_config=dict(vocab_size=151680, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                         num_experts=4, num_experts_per_tok=2, rope_scaling={"rope_type": "default",
                         "mrope_section": [2, 3, 3], "interleaved": True, "mrope_interleaved": True}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32,
                           deepstack_visual_indexes=[0]),
        audio_token_id=AUDIO_TOKEN, audio_start_token_id=AUDIO_START, audio_end_token_id=AUDIO_END,
        position_id_per_seconds=13, vision_start_token_id=151652, vision_end_token_id=151653,
    )
    cfg._attn_implementation = "sdpa"
    cfg.audio_config._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = "sdpa"
    model = m.Qwen3OmniMoeThinkerForConditionalGeneration(cfg).float().eval()
    # At the default init the loss barely depends on the audio and the unfixed model passes too;
    # at 20x the unfixed per-example losses differ by ~0.2 between batch 1 and batch 3.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2 and "embed_tokens" not in name:
                p.mul_(20)
    return model


def example(n_frames, seed):
    g = torch.Generator().manual_seed(seed)
    n_audio = int(m._get_feat_extract_output_lengths(torch.tensor(n_frames)))
    text = torch.randint(1000, 2000, (5,), generator=g).tolist()
    label = torch.randint(1000, 2000, (3,), generator=g).tolist()
    ids = text[:2] + [AUDIO_START] + [AUDIO_TOKEN] * n_audio + [AUDIO_END] + text[2:] + label
    return {"ids": ids, "n_label": len(label), "feats": torch.randn(128, n_frames, generator=g)}


def collate(examples):
    width = max(len(e["ids"]) for e in examples)
    frames = max(e["feats"].shape[1] for e in examples)
    batch = {
        "input_ids": torch.full((len(examples), width), PAD),
        "attention_mask": torch.zeros(len(examples), width, dtype=torch.long),
        "labels": torch.full((len(examples), width), -100),
        "input_features": torch.zeros(len(examples), 128, frames),
        "feature_attention_mask": torch.zeros(len(examples), frames, dtype=torch.long),
    }
    for i, e in enumerate(examples):
        n, f = len(e["ids"]), e["feats"].shape[1]
        batch["input_ids"][i, width - n:] = torch.tensor(e["ids"])
        batch["attention_mask"][i, width - n:] = 1
        batch["labels"][i, width - e["n_label"]:] = torch.tensor(e["ids"][-e["n_label"]:])
        batch["input_features"][i, :, :f] = e["feats"]
        batch["feature_attention_mask"][i, :f] = 1
    return batch


@torch.no_grad()
def per_example_loss(model, examples):
    batch = collate(examples)
    labels = batch.pop("labels")
    logits = model(**batch).logits
    ce = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none")
    keep = (labels[:, 1:] != -100).float()
    return (ce * keep).sum(1) / keep.sum(1)


def test_per_example_loss_batch1_equals_batchN():
    qwen3_omni_fix.apply()
    model = tiny_thinker()
    examples = [example(n, seed) for seed, n in enumerate(FRAMES)]
    single = torch.cat([per_example_loss(model, [e]) for e in examples])
    batched = per_example_loss(model, examples)
    assert (single - batched).abs().max().item() <= TOL, (single, batched)


def test_encoder_windows_do_not_attend_to_each_other():
    qwen3_omni_fix.apply()
    encoder = tiny_thinker().audio_tower
    feats = torch.randn(128, FRAMES[0], generator=torch.Generator().manual_seed(1))
    changed = feats.clone()
    changed[:, 800:] = torch.randn(128, FRAMES[0] - 800, generator=torch.Generator().manual_seed(2))
    lens = torch.tensor([FRAMES[0]])
    with torch.no_grad():
        a = encoder(feats, feature_lens=lens).last_hidden_state
        b = encoder(changed, feature_lens=lens).last_hidden_state
    window = 13 * (800 // 100)
    scale = a.abs().max()
    assert (a[:window] - b[:window]).abs().max() <= 1e-5 * scale
    assert (a[window:] - b[window:]).abs().max() > 1e-2 * scale


def test_feature_mask_counts_real_frames_only():
    processor = qwen3_omni_fix.fix_processor(SimpleNamespace(feature_extractor=WhisperFeatureExtractor(feature_size=128)))
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(n).astype(np.float32) for n in (167814, 103133, 48288)]
    fe = processor.feature_extractor
    batched = fe(clips, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
    for i, clip in enumerate(clips):
        alone = fe([clip], sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
        assert int(batched["attention_mask"][i].sum()) == int(alone["attention_mask"][0].sum()) == len(clip) // 160
