"""Gradient equivalence of Qwen3-Omni sequence SFT: batch 1 x accumulation 8 vs batch 8 x accumulation 1.

    python scripts/evaluate/qwen3_omni_grad_check.py data/sequences/train.jsonl runs/qwen3_grad_check.json

Same 8 Train sequences (seed 7), same LoRA/connector setup and the same LengthBinWeightedTrainer
training_step (loss scaling, accumulation, bin weights) as scripts/finetune/sequence_lora.py.
Dropout is zeroed and LoRA B gets small seeded values (at its zero init LoRA A has no gradient).
Pass: relative L2 difference of the full trainable-gradient vector < 1e-3. A batch-8 run in shuffled
order is reported as the bf16 batching noise floor.
"""
import json, random, sys
from pathlib import Path

import torch
from transformers import TrainingArguments

from emoseq import config
from emoseq.bins import apply_bin_weights, sklearn_balanced_bin_weights
from emoseq.models.adapters import get_adapter, resolve_snapshot
from emoseq.records import load_jsonl

sys.path.insert(0, str(config.REPO / "scripts" / "finetune"))
from sequence_lora import (LengthBinWeightedTrainer, freeze_audio_tower, lora_config,  # noqa: E402
                           make_qwen2_collate)
from peft import get_peft_model  # noqa: E402

TOL = 1e-3
src, out = sys.argv[1], Path(sys.argv[2])
rows = load_jsonl(src)
apply_bin_weights(rows, sklearn_balanced_bin_weights(rows))
batch_rows = random.Random(7).sample(rows, 8)

ad = get_adapter("qwen3_omni")
snap = resolve_snapshot("qwen3_omni")
processor = ad.load_processor(snap)
model = ad.load_model(snap, torch.bfloat16, device_map={"": 0})
collate = make_qwen2_collate(processor, config.SEQUENCE_PROMPT.read_text().strip(), ad.sampling_rate(processor),
                             "user", model_key="qwen3_omni")
freeze_audio_tower(model)
model = get_peft_model(model, lora_config("qwen3_omni"))
model.enable_input_require_grads()
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.copy_((torch.randn(p.shape, generator=g) * 1e-3).to(p.dtype))
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.0
params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def gradient(bs, order):
    args = TrainingArguments(output_dir=str(out.parent / "_grad_check_tmp"), per_device_train_batch_size=bs,
                             gradient_accumulation_steps=8 // bs, bf16=True, gradient_checkpointing=True,
                             remove_unused_columns=False, report_to=[])
    trainer = LengthBinWeightedTrainer(model=model, args=args, data_collator=collate, apply_bin_weight=True)
    model.zero_grad(set_to_none=True)
    micro = [collate([batch_rows[i] for i in order[k:k + bs]]) for k in range(0, 8, bs)]
    n_items = sum(int(b["labels"].ne(-100).sum()) for b in micro)
    trainer.current_gradient_accumulation_steps = len(micro)
    losses = []
    for b in micro:
        b = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):  # what Trainer(bf16=True) wraps forward in
            losses.append(float(trainer.training_step(model, b, torch.tensor(n_items, device="cuda:0"))))
    flat = torch.cat([p.grad.detach().float().flatten() for _, p in params])
    per_tensor = {n: p.grad.detach().float().clone() for n, p in params}
    return flat, per_tensor, losses, trainer.model_accepts_loss_kwargs


g1, t1, l1, accepts = gradient(1, list(range(8)))
g8, t8, l8, _ = gradient(8, list(range(8)))
shuffled = random.Random(1).sample(range(8), 8)
g8s, _, _, _ = gradient(8, shuffled)


def rel(a, b):
    return float((a - b).norm() / b.norm())


report = {
    "seq_ids": [r["seq_id"] for r in batch_rows], "durations": [r["audio_duration"] for r in batch_rows],
    "model_accepts_loss_kwargs": accepts,
    "step_loss_bs1x8": sum(l1), "step_loss_bs8x1": sum(l8),
    "rel_l2_bs1x8_vs_bs8x1": rel(g8, g1), "rel_l2_bs8_vs_bs8_shuffled": rel(g8s, g8),
    "grad_norm_bs1x8": float(g1.norm()), "grad_norm_bs8x1": float(g8.norm()),
    "worst_tensors": sorted(((rel(t8[n], t1[n]), n) for n in t1 if t1[n].norm() > 0), reverse=True)[:5],
}
report["pass"] = report["rel_l2_bs1x8_vs_bs8x1"] < TOL
out.write_text(json.dumps(report, indent=2))
print(f"durations {[round(d, 1) for d in report['durations']]}")
print(f"step loss bs1x8 {report['step_loss_bs1x8']:.5f}  bs8x1 {report['step_loss_bs8x1']:.5f}")
print(f"grad norm bs1x8 {report['grad_norm_bs1x8']:.5f}  bs8x1 {report['grad_norm_bs8x1']:.5f}")
print(f"relative L2 bs1x8 vs bs8x1: {report['rel_l2_bs1x8_vs_bs8x1']:.2e}  "
      f"(bs8 vs bs8 shuffled: {report['rel_l2_bs8_vs_bs8_shuffled']:.2e})")
print(f"PASS={report['pass']}  -> {out}")
