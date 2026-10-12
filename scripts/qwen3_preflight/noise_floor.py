"""Is Qwen3-Omni's remaining batch-1 vs batch-8 loss difference bf16 noise or a leak? (FA2 + fix)

    noise_floor.py <utt_train.jsonl> <out.json>

Same 32 seed-7 Train utterances as qwen3_omni_batch_check.py. Per-example loss of each clip:
  bs1         alone
  copies      in a batch of 8 identical copies (batch shape changes; no padding, no other clips)
  mixed       in its mixed batch of 8 (padding + other clips)
each with bf16 logits (as trained) and with fp32 logits (final hidden state x lm_head in fp32).
If |mixed - bs1| is of the same size as |copies - bs1|, the difference is batch-shape numerics.
"""
import json, random, sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from emoseq import config
from emoseq.models.adapters import get_adapter, resolve_snapshot

sys.path.insert(0, str(config.REPO / "scripts" / "finetune"))
from utterance_lora import qwen3_omni_collate  # noqa: E402

rows = random.Random(7).sample([json.loads(l) for l in open(sys.argv[1]) if l.strip()], 32)
out = Path(sys.argv[2])
ad = get_adapter("qwen3_omni")
snap = resolve_snapshot("qwen3_omni")
processor = ad.load_processor(snap)
model = ad.load_model(snap, torch.bfloat16, device_map={"": 0})
tcfg = json.loads((config.REPO / "configs/finetune/utterance_lora.json").read_text())
model = get_peft_model(model, LoraConfig(**tcfg["lora"])).eval()
collate = qwen3_omni_collate(processor, config.UTTERANCE_PROMPT.read_text().strip())
lm_head = model.get_output_embeddings()
w32 = lm_head.weight.float()
captured = {}
lm_head.register_forward_hook(lambda mod, inp, outp: captured.__setitem__("h", inp[0]))


def ce(logits, labels):
    lp = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none")
    keep = (labels[:, 1:] != -100).float()
    return ((lp * keep).sum(1) / keep.sum(1)).tolist()


@torch.no_grad()
def losses(batch_rows):
    inputs = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in collate(batch_rows).items()}
    labels = inputs.pop("labels")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits
    return ce(logits.float(), labels), ce(captured["h"].float() @ w32.T, labels)


res = {k: {"bf16": [], "fp32": []} for k in ("bs1", "copies", "mixed")}
for r in rows:
    for kind, (a, b) in (("bs1", losses([r])), ("copies", losses([r] * 8))):
        res[kind]["bf16"].append(a[0])
        res[kind]["fp32"].append(b[0])
for i in range(0, 32, 8):
    a, b = losses(rows[i:i + 8])
    res["mixed"]["bf16"] += a
    res["mixed"]["fp32"] += b

summary = {}
for prec in ("bf16", "fp32"):
    for kind in ("copies", "mixed"):
        d = [abs(x - y) for x, y in zip(res[kind][prec], res["bs1"][prec])]
        summary[f"{kind}_{prec}"] = {"max": max(d), "mean": sum(d) / len(d), "n_over_1e-3": sum(x > 1e-3 for x in d)}
        print(f"{prec} logits, {kind:6s} vs batch 1: max|d| {max(d):.2e} mean|d| {sum(d)/len(d):.2e} "
              f"over 1e-3: {sum(x > 1e-3 for x in d)}/32", flush=True)
out.write_text(json.dumps({"summary": summary, "losses": res}, indent=2))
