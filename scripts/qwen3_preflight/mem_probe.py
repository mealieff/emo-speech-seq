"""Peak memory of one Qwen3-Omni sequence-SFT forward+backward without gradient checkpointing (FA2, bf16, LoRA).

    mem_probe.py <train.jsonl> <out.json>

Batch 1 at decreasing durations (longest Train sequence at or under each target), then batch 8 at
padded totals (8 x longest clip) near the batch-1 limit. Prints peak GiB or OOM per case.
"""
import json, sys
from pathlib import Path

import torch

from emoseq import config
from emoseq.models.adapters import get_adapter, resolve_snapshot
from emoseq.records import load_jsonl

sys.path.insert(0, str(config.REPO / "scripts" / "finetune"))
from sequence_lora import freeze_audio_tower, lora_config, make_qwen2_collate, weighted_token_ce  # noqa: E402
from peft import get_peft_model  # noqa: E402

rows = sorted(load_jsonl(sys.argv[1]), key=lambda r: r["audio_duration"])
out = Path(sys.argv[2])
ad = get_adapter("qwen3_omni")
snap = resolve_snapshot("qwen3_omni")
processor = ad.load_processor(snap)
model = ad.load_model(snap, torch.bfloat16, device_map={"": 0})
collate = make_qwen2_collate(processor, config.SEQUENCE_PROMPT.read_text().strip(), ad.sampling_rate(processor),
                             "user", model_key="qwen3_omni")
freeze_audio_tower(model)
model = get_peft_model(model, lora_config("qwen3_omni"))
model.train()
base = torch.cuda.memory_allocated() / 2**30
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"weights+LoRA resident {base:.1f} GiB of {total:.1f}", flush=True)


def probe(batch):
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        inputs = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in collate(batch).items()}
        w = inputs.pop("sample_weight")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**inputs)
            loss = weighted_token_ce(o.logits, inputs["labels"], w)
        loss.backward()
        del o, loss, inputs
        return torch.cuda.max_memory_allocated() / 2**30
    except torch.OutOfMemoryError:
        return None
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def at_most(sec):
    return max((r for r in rows if r["audio_duration"] <= sec), key=lambda r: r["audio_duration"])


results = {"resident_gib": base, "total_gib": total, "batch1": [], "batch8": []}
for target in (210, 180, 150, 120, 90, 60):
    r = at_most(target)
    peak = probe([r])
    results["batch1"].append({"sec": r["audio_duration"], "peak_gib": peak})
    print(f"batch 1  {r['audio_duration']:6.1f} s  peak {'OOM' if peak is None else f'{peak:.1f} GiB'}", flush=True)
for per_clip in (26, 20, 15, 10):
    clip = [r for r in rows if r["audio_duration"] <= per_clip][-8:]
    padded = 8 * max(r["audio_duration"] for r in clip)
    peak = probe(clip)
    results["batch8"].append({"padded_sec": padded, "peak_gib": peak})
    print(f"batch 8  padded {padded:6.1f} s  peak {'OOM' if peak is None else f'{peak:.1f} GiB'}", flush=True)
out.write_text(json.dumps(results, indent=2))
