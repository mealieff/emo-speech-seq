"""Per-example eval-mode losses of Qwen3-Omni (real weights, bf16, one GPU), with and without
emoseq/models/qwen3_omni_fix.py. Run before any batched Qwen3-Omni training.

    python scripts/evaluate/qwen3_omni_batch_check.py data/utterances/msp_podcast_train.jsonl 32 \\
        data/sequences/train.jsonl 8 runs/qwen3_batch_check.json

Utterances (collator of scripts/finetune/utterance_lora.py): batch 1 vs batch 8 on a seed-7 sample,
batches of 8 in sample order, so clip lengths are mixed. Pass: max |diff| <= 1e-3 with the fix.
Sequences (collator of scripts/finetune/sequence_lora.py, user layout): batch 1 only, n_seq Train
sequences spread over the duration range; fixed vs unfixed shows whether batch-1 training changed.
LoRA is attached with B = 0, so all numbers are the base model's. Loss = mean token CE on labels.
"""
import json, random, sys
from pathlib import Path

import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model

from emoseq import config

sys.path.insert(0, str(config.REPO / "scripts" / "finetune"))
from utterance_lora import qwen3_omni_collate  # noqa: E402
from sequence_lora import make_qwen2_collate  # noqa: E402
from emoseq.models import qwen3_omni_fix  # noqa: E402
from emoseq.models.adapters import get_adapter, resolve_snapshot  # noqa: E402
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as m  # noqa: E402

utt_src, n, seq_src, n_seq, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
rows = random.Random(7).sample([json.loads(l) for l in open(utt_src) if l.strip()], n)
seqs = sorted((json.loads(l) for l in open(seq_src) if l.strip()), key=lambda r: r["audio_duration"])
seqs = [seqs[round(i * (len(seqs) - 1) / (n_seq - 1))] for i in range(n_seq)]
utt_dur = [sf.info(r["audio_path"]).duration for r in rows]

ad = get_adapter("qwen3_omni")
snap = resolve_snapshot("qwen3_omni")
processor = ad.load_processor(snap)
model = ad.load_model(snap, torch.bfloat16, device_map={"": 0})
tcfg = json.loads((config.REPO / "configs/finetune/utterance_lora.json").read_text())
model = get_peft_model(model, LoraConfig(**tcfg["lora"])).eval()
utt_collate = qwen3_omni_collate(processor, config.UTTERANCE_PROMPT.read_text().strip())
seq_collate = make_qwen2_collate(processor, config.SEQUENCE_PROMPT.read_text().strip(),
                                 ad.sampling_rate(processor), "user", model_key="qwen3_omni")
fixed_fe_class = type(processor.feature_extractor)
plain_fe_class = fixed_fe_class.__mro__[1]


@torch.no_grad()
def per_example(collate, batch_rows):
    inputs = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in collate(batch_rows).items()}
    labels = inputs.pop("labels")
    for k in ("sample_weight", "token_type_weight"):
        inputs.pop(k, None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits.float()
    lp = torch.nn.functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="none")
    keep = (labels[:, 1:] != -100).float()
    return ((lp * keep).sum(1) / keep.sum(1)).tolist()


def run(tag, with_seq):
    single = [per_example(utt_collate, [r])[0] for r in rows]
    batched = [x for i in range(0, n, 8) for x in per_example(utt_collate, rows[i:i + 8])]
    diffs = [abs(a - b) for a, b in zip(single, batched)]
    res = {"utt_bs1": single, "utt_bs8": batched, "utt_max_abs_diff": max(diffs),
           "utt_n_over_1e-3": sum(d > 1e-3 for d in diffs)}
    print(f"[{tag}] utterances batch 1 vs 8: max|d|={max(diffs):.2e} over 1e-3: {res['utt_n_over_1e-3']}/{n}", flush=True)
    if with_seq:
        res["seq_bs1"] = [per_example(seq_collate, [r])[0] for r in seqs]
    return res


report = {"utt_ids": [r["audio_path"] for r in rows], "utt_duration": utt_dur,
          "seq_ids": [r["seq_id"] for r in seqs], "seq_duration": [r["audio_duration"] for r in seqs]}
report["fixed"] = run("fixed", True)
m.Qwen3OmniMoeAudioEncoderLayer.forward = qwen3_omni_fix.ORIGINAL_LAYER_FORWARD
processor.feature_extractor.__class__ = plain_fe_class
report["unfixed"] = run("unfixed", True)
qwen3_omni_fix._PATCHED = False
qwen3_omni_fix.apply()
report["encoder_fix_only"] = run("encoder fix only", False)

print("batch-1 losses, unfixed vs fixed:")
for kind, durs in (("utt", utt_dur), ("seq", report["seq_duration"])):
    a, b = report["unfixed"][f"{kind}_bs1"], report["fixed"][f"{kind}_bs1"]
    for cond, name in ((lambda d: d <= 8.0, "<=8 s"), (lambda d: d > 8.0, ">8 s")):
        idx = [i for i, d in enumerate(durs) if cond(d)]
        if idx:
            d = [abs(a[i] - b[i]) for i in idx]
            print(f"  {kind} {name}: n={len(idx)} max|d|={max(d):.2e} mean|d|={sum(d)/len(d):.2e} "
                  f"mean loss unfixed {sum(a[i] for i in idx)/len(idx):.4f} fixed {sum(b[i] for i in idx)/len(idx):.4f}")
for i, r in enumerate(seqs):
    print(f"  seq {r['seq_id']} {r['audio_duration']:.1f}s loss unfixed {report['unfixed']['seq_bs1'][i]:.4f} "
          f"fixed {report['fixed']['seq_bs1'][i]:.4f}")
out.write_text(json.dumps(report, indent=2))
print(f"PASS={report['fixed']['utt_max_abs_diff'] <= 1e-3}  -> {out}")
