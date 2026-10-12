"""CPU fp32 check of the Qwen3-Omni audio encoder: batch-vs-single and long-clip windowing, before/after the mask fix."""
import json, sys
import numpy as np, soundfile as sf, torch
from safetensors import safe_open
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m

torch.set_grad_enabled(False)
snap = Path(sys.argv[1])
cfg = AutoConfig.from_pretrained(snap).thinker_config.audio_config
cfg._attn_implementation = "sdpa"
enc = m.Qwen3OmniMoeAudioEncoder(cfg).float().eval()
sd = {}
for f in snap.glob("*.safetensors"):
    with safe_open(f, "pt") as h:
        for k in h.keys():
            if k.startswith("thinker.audio_tower."):
                sd[k[len("thinker.audio_tower."):]] = h.get_tensor(k).float()
missing, unexpected = enc.load_state_dict(sd, strict=False)
assert not unexpected and not [k for k in missing if "positional" not in k], (missing, unexpected)
fe = AutoProcessor.from_pretrained(snap).feature_extractor

def load(p):
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    assert sr == 16000, sr
    return a

def encode(clips, fix_len=False):
    f = fe(clips, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
    x, mk = f["input_features"], f["attention_mask"]
    if fix_len:
        for i, c in enumerate(clips):
            mk[i, len(c) // 160:] = 0
    L = mk.sum(-1)
    packed = x.permute(0, 2, 1)[mk.bool()].permute(1, 0)
    out = enc(packed, feature_lens=L).last_hidden_state
    sizes = m._get_feat_extract_output_lengths(L).tolist()
    assert sum(sizes) == out.shape[0], (sizes, out.shape)
    return list(out.split(sizes)), L.tolist()

def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()

utts = [json.loads(l) for l, _ in zip(open("data/utterances/msp_podcast_test1.jsonl"), range(8))]
clips = [load(r["audio_path"]) for r in utts]
seq = json.loads(next(open("data/sequences/test1.jsonl")))
long_clip = load(seq["audio_path"])

def run(tag, fix_len=False):
    batch, Lb = encode(clips, fix_len)
    worst = 0.0
    for i, c in enumerate(clips):
        single, Ls = encode([c], fix_len)
        print(f"  [{tag}] clip {i}: {len(c)} samples, frames single {Ls[0]} batch {Lb[i]}")
        if Ls[0] != Lb[i] or single[0].shape != batch[i].shape:
            print(f"  [{tag}] clip {i}: frames single {Ls[0]} vs batch {Lb[i]}, tokens {single[0].shape[0]} vs {batch[i].shape[0]}")
            n = min(single[0].shape[0], batch[i].shape[0]) - 2
            worst = max(worst, rel(batch[i][:n], single[0][:n]))
            continue
        worst = max(worst, rel(batch[i], single[0]))
    print(f"[{tag}] batch-of-8 vs single, worst relative max diff: {worst:.2e}")
    return encode([long_clip])[0][0]

sys.path.insert(0, ".")
from emoseq.models import qwen3_omni_fix
qwen3_omni_fix.apply()
run("patched+len", fix_len=True)
