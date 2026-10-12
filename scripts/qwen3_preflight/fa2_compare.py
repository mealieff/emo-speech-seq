"""Qwen3-Omni audio encoder: patched SDPA vs FlashAttention-2 (the upstream windowed path), real weights.

    PYTHONPATH=<flash-attn dir> python fa2_compare.py <snapshot> <out.json>

Batch 1 on Test1 sequences of increasing length (bf16 FA2, bf16 SDPA patched/unpatched, fp32 SDPA
patched as a precision reference), then FA2 batch-of-6 vs single with the fixed feature mask.
"""
import copy, json, sys
from pathlib import Path

import soundfile as sf
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

sys.path.insert(0, "/media/volume/AudioEmotionSequenceVol/emo-seq")
from emoseq.models import qwen3_omni_fix  # noqa: E402

torch.set_grad_enabled(False)
snap, out = Path(sys.argv[1]), Path(sys.argv[2])
base_cfg = AutoConfig.from_pretrained(snap).thinker_config.audio_config
sd = {}
for f in snap.glob("*.safetensors"):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("thinker.audio_tower."):
                sd[k[len("thinker.audio_tower."):]] = h.get_tensor(k)


def encoder(attn, dtype):
    cfg = copy.deepcopy(base_cfg)
    cfg._attn_implementation = attn
    enc = m.Qwen3OmniMoeAudioEncoder(cfg)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert not unexpected and all("positional" in k for k in missing), (missing, unexpected)
    return enc.to("cuda", dtype).eval()


proc = qwen3_omni_fix.fix_processor(AutoProcessor.from_pretrained(snap))
fe = proc.feature_extractor


def encode(enc, clips, dtype):
    f = fe(clips, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
    x, mk = f["input_features"], f["attention_mask"]
    L = mk.sum(-1)
    packed = x.permute(0, 2, 1)[mk.bool()].permute(1, 0).to("cuda", dtype)
    o = enc(packed, feature_lens=L.to("cuda")).last_hidden_state.float()
    return list(o.split(m._get_feat_extract_output_lengths(L).tolist()))


def cmp(a, b):
    return {"rel_max": ((a - b).abs().max() / b.abs().max()).item(),
            "cos_mean": torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item(),
            "cos_min": torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()}


rows = sorted((json.loads(l) for l in open("/media/volume/AudioEmotionSequenceVol/emo-seq/data/sequences/test1.jsonl")),
              key=lambda r: r["audio_duration"])
picks = [rows[round(q * (len(rows) - 1))] for q in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
clips = [sf.read(r["audio_path"], dtype="float32")[0] for r in picks]

fa2 = encoder("flash_attention_2", torch.bfloat16)
sdpa_bf16 = encoder("sdpa", torch.bfloat16)
sdpa_fp32 = encoder("sdpa", torch.float32)

unpatched = [encode(sdpa_bf16, [c], torch.bfloat16)[0] for c in clips]
qwen3_omni_fix.apply()
report = {"batch1": []}
for r, c, un in zip(picks, clips, unpatched):
    ref = encode(fa2, [c], torch.bfloat16)[0]
    pat = encode(sdpa_bf16, [c], torch.bfloat16)[0]
    f32 = encode(sdpa_fp32, [c], torch.float32)[0]
    row = {"seq_id": r["seq_id"], "sec": round(r["audio_duration"], 1),
           "patched_sdpa_bf16_vs_fa2": cmp(pat, ref), "unpatched_sdpa_bf16_vs_fa2": cmp(un, ref),
           "fa2_vs_fp32_patched": cmp(ref, f32), "patched_bf16_vs_fp32_patched": cmp(pat, f32)}
    report["batch1"].append(row)
    print(f"{row['sec']:6.1f}s  patched-vs-FA2 cos {row['patched_sdpa_bf16_vs_fa2']['cos_mean']:.5f} "
          f"(min {row['patched_sdpa_bf16_vs_fa2']['cos_min']:.4f})  unpatched-vs-FA2 cos "
          f"{row['unpatched_sdpa_bf16_vs_fa2']['cos_mean']:.4f}  | vs fp32 ref: FA2 cos "
          f"{row['fa2_vs_fp32_patched']['cos_mean']:.5f}, patched bf16 cos {row['patched_bf16_vs_fp32_patched']['cos_mean']:.5f}",
          flush=True)

batched = encode(fa2, clips, torch.bfloat16)
singles = [encode(fa2, [c], torch.bfloat16)[0] for c in clips]
report["fa2_batch6_vs_single"] = [cmp(b, s) for b, s in zip(batched, singles)]
print("FA2 batch-of-6 vs single, cos min per clip:",
      [round(x["cos_min"], 5) for x in report["fa2_batch6_vs_single"]])
out.write_text(json.dumps(report, indent=2))
