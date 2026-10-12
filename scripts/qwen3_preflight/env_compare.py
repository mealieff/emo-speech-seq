"""Batch-1 per-example losses of Qwen3-Omni (base model, LoRA B = 0) on the examples of
runs/qwen3_seq_speed/batch_check.json, in the current environment; compared against that file's
"fixed" batch-1 losses (torch 2.6 env, FA2, eager-loop MoE experts).

    .venv-t28/bin/python env_compare.py runs/qwen3_seq_speed/batch_check.json out.json
"""
import json, random, sys, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from emoseq import config

sys.path.insert(0, str(config.REPO / "scripts" / "finetune"))
from utterance_lora import qwen3_omni_collate  # noqa: E402
from sequence_lora import make_qwen2_collate  # noqa: E402
from emoseq.models.adapters import get_adapter, resolve_snapshot  # noqa: E402
from transformers.integrations import moe  # noqa: E402

ref = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
utt_rows = [json.loads(l) for l in open(config.utterance_jsonl("train")) if l.strip()]
rows = random.Random(7).sample(utt_rows, len(ref["utt_ids"]))
assert [r["audio_path"] for r in rows] == ref["utt_ids"]
by_id = {json.loads(l)["seq_id"]: json.loads(l) for l in open(config.REPO / "data/sequences/train.jsonl") if l.strip()}
seqs = [by_id[s] for s in ref["seq_ids"]]

ad = get_adapter("qwen3_omni")
snap = resolve_snapshot("qwen3_omni")
processor = ad.load_processor(snap)
model = ad.load_model(snap, torch.bfloat16, device_map={"": 0})
tcfg = json.loads((config.REPO / "configs/finetune/utterance_lora.json").read_text())
model = get_peft_model(model, LoraConfig(**tcfg["lora"])).eval()
text_cfg = model.base_model.model.config.thinker_config.text_config if hasattr(model.base_model.model.config, "thinker_config") \
    else model.base_model.model.config.get_text_config()
experts = getattr(text_cfg, "_experts_implementation", None)
probe = torch.zeros(4, 8, device="cuda:0", dtype=torch.bfloat16)
native = moe._can_use_grouped_mm(probe, torch.zeros(2, 8, 8, device="cuda:0", dtype=torch.bfloat16),
                                 torch.tensor([2, 4], device="cuda:0", dtype=torch.int32))
print(f"torch {torch.__version__} experts_implementation={experts} native_grouped_mm={native} "
      f"attn={model.base_model.model.config._attn_implementation}", flush=True)
utt_collate = qwen3_omni_collate(processor, config.UTTERANCE_PROMPT.read_text().strip())
seq_collate = make_qwen2_collate(processor, config.SEQUENCE_PROMPT.read_text().strip(),
                                 ad.sampling_rate(processor), "user", model_key="qwen3_omni")


@torch.no_grad()
def loss1(collate, row):
    inputs = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in collate([row]).items()}
    labels = inputs.pop("labels")
    for k in ("sample_weight", "token_type_weight"):
        inputs.pop(k, None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits.float()
    lp = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="none")
    keep = (labels[:, 1:] != -100).float()
    return ((lp * keep).sum(1) / keep.sum(1)).item()


loss1(utt_collate, rows[0])
torch.cuda.synchronize(); t0 = time.time()
utt = [loss1(utt_collate, r) for r in rows]
torch.cuda.synchronize(); t_utt = (time.time() - t0) / len(rows)
seq = [loss1(seq_collate, r) for r in seqs]
du = [abs(a - b) for a, b in zip(utt, ref["fixed"]["utt_bs1"])]
ds = [abs(a - b) for a, b in zip(seq, ref["fixed"]["seq_bs1"])]
res = {"torch": torch.__version__, "experts_implementation": experts, "native_grouped_mm": native,
       "utt_bs1": utt, "seq_bs1": seq, "utt_max_abs_diff": max(du), "utt_mean_abs_diff": sum(du) / len(du),
       "seq_max_abs_diff": max(ds), "seq_mean_abs_diff": sum(ds) / len(ds), "forward_sec_per_utt": t_utt}
print(f"utterances vs torch 2.6: max|d|={max(du):.2e} mean|d|={sum(du)/len(du):.2e}; "
      f"sequences: max|d|={max(ds):.2e} mean|d|={sum(ds)/len(ds):.2e}; forward {t_utt:.3f} s/utt", flush=True)
for r, a, b in zip(seqs, seq, ref["fixed"]["seq_bs1"]):
    print(f"  seq {r['seq_id']} {r['audio_duration']:.1f}s torch2.6 {b:.4f} now {a:.4f}")
out.write_text(json.dumps(res, indent=2))
