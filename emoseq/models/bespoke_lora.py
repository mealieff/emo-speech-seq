"""LoRA structure for the bespoke ALM stacks (SALMONN, Kimi-Audio), shared by training
(scripts/finetune/bespoke_lora.py) and inference (scripts/inference/bespoke.py --sft-dir).

Same adapter as scripts/finetune/sequence_lora.py (Qwen2-Audio / AF3): r=16, alpha=32,
dropout 0.05 on the LLM attention projections; audio encoders frozen; modality connector trained.
Only trainable tensors are saved (``trainable.pt``), keyed by parameter name, so a checkpoint is
restored by rebuilding the same structure and loading that file.
"""
from __future__ import annotations

from pathlib import Path

import torch

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
ATTN_PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]
TRAINABLE_FILE = "trainable.pt"

# SALMONN connector: window-level Q-Former, its query tokens, the LLM projection and the two
# input LayerNorms (all trained in SALMONN's own recipe). Whisper and BEATs stay frozen.
SALMONN_CONNECTOR = ("speech_Qformer.", "speech_query_tokens", "speech_llama_proj.", "ln_speech.", "ln_audio.")
# Kimi-Audio: LoRA on the main 28-layer stack only; text logits come from lm_head on that stack
# (the 6 mimo layers feed the audio stream). vq_adaptor maps Whisper features into the LLM.
KIMI_TARGET = r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def salmonn_add_lora(model) -> None:
    """Merge SALMONN's released r=8 LoRA into Vicuna, then add a fresh r=16 adapter.

    ``model.lora`` stays True: SALMONN reaches embeddings via ``llama_model.model.model``, which
    the new PeftModel wrapper provides.
    """
    from peft import LoraConfig, get_peft_model

    model.llama_model = model.llama_model.merge_and_unload()
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     target_modules=ATTN_PROJ, bias="none", task_type="CAUSAL_LM")
    model.llama_model = get_peft_model(model.llama_model, cfg)
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name or name.startswith(SALMONN_CONNECTOR)


def kimi_add_lora(alm):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     target_modules=KIMI_TARGET, modules_to_save=["vq_adaptor"], bias="none")
    return get_peft_model(alm, cfg)


def trainable_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in module.named_parameters() if p.requires_grad}


def save_trainable(module: torch.nn.Module, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state(module), ckpt_dir / TRAINABLE_FILE)


def load_trainable(module: torch.nn.Module, ckpt_dir: Path) -> None:
    state = torch.load(Path(ckpt_dir) / TRAINABLE_FILE, map_location="cpu")
    params = dict(module.named_parameters())
    missing = [k for k in state if k not in params]
    if missing:
        raise RuntimeError(f"{len(missing)} saved tensors have no parameter in the model, e.g. {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(device=params[k].device, dtype=params[k].dtype))
    expected = [n for n, p in params.items() if p.requires_grad and n not in state]
    if expected:
        raise RuntimeError(f"{len(expected)} trainable parameters absent from {ckpt_dir}, e.g. {expected[:3]}")
