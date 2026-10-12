"""Greedy generation for the Hugging Face ALMs (Qwen2-Audio, Audio-Flamingo-3, Qwen3-Omni).

``generate_pretrained`` / ``generate_qwen2_chunked`` use the adapter's chat template on base
weights; ``generate_sft*`` use the same chat template as ``scripts/finetune/sequence_lora.py``.
Audio longer than Qwen2-Audio's 30 s encoder window is decoded in non-overlapping windows whose
spans are shifted back to sequence time and seam-merged (``emoseq.windows.stitch_windows``).
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import torch

from emoseq.parsing.sequence import parse_prediction
from emoseq.windows import QWEN2_MAX_SEC, split_audio, stitch_windows  # noqa: F401


def load_done(jsonl: Path) -> set[str]:
    done = set()
    if not jsonl.is_file():
        return done
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["seq_id"])
    return done


def _tensors_to_device(inputs, device, dtype):
    casted = {}
    for k, v in dict(inputs).items():
        if hasattr(v, "to"):
            if v.ndim == 1 and k in {"input_ids", "attention_mask"}:
                v = v.unsqueeze(0)
            casted[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        else:
            casted[k] = v
    return casted


def generate_from_audio(
    model, processor, adapter, instruction: str, audio, sr: int, max_new: int, device, *, audio_path: str | None,
    prompt_layout: str = "user",
):
    inputs, _mode = adapter.build_generation_inputs(
        processor,
        instruction,
        audio,
        sr,
        audio_path=audio_path or "sequence",
        base_only=True,
        prompt_layout=prompt_layout,
    )
    dtype = next(model.parameters()).dtype
    casted = _tensors_to_device(inputs, device, dtype)
    with torch.no_grad():
        gen = model.generate(**casted, max_new_tokens=max_new, do_sample=False)
    out_ids = gen[:, casted["input_ids"].shape[1] :]
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def generate_pretrained(
    model, processor, adapter, instruction: str, audio_path: str, sr: int, max_new: int, device,
    prompt_layout: str = "user",
):
    audio = librosa.load(audio_path, sr=sr)[0]
    return generate_from_audio(
        model, processor, adapter, instruction, audio, sr, max_new, device, audio_path=audio_path,
        prompt_layout=prompt_layout,
    )


def generate_qwen2_chunked(
    model, processor, adapter, instruction: str, audio_path: str, sr: int, max_new: int, device, max_sec: float,
    prompt_layout: str = "user",
) -> tuple[str, list[dict], int]:
    audio = librosa.load(audio_path, sr=sr)[0]
    windows = split_audio(audio, sr, max_sec)
    texts: list[str] = []
    parsed: list[tuple[float, list[dict]]] = []
    for offset, wav in windows:
        text = generate_from_audio(
            model, processor, adapter, instruction, wav, sr, max_new, device, audio_path="sequence",
            prompt_layout=prompt_layout,
        )
        texts.append(f"# chunk offset={offset:.2f}s dur={len(wav)/sr:.2f}s\n{text}")
        parsed.append((offset, parse_prediction(text)))
    return "\n\n".join(texts), stitch_windows(parsed), len(windows)


def _sft_conversation(model_key: str, instruction: str, audio_path: str, prompt_layout: str = "user") -> list[dict]:
    """Default ``user`` matches train_lora.py (instruction folded into the user turn)."""
    if model_key == "audio_flamingo3":
        audio = {"type": "audio", "path": audio_path}
        system = {"role": "system", "content": [{"type": "text", "text": instruction}]}
    else:
        audio = {"type": "audio", "audio_url": "sequence"}
        system = {"role": "system", "content": instruction}
    if prompt_layout == "system":
        return [system, {"role": "user", "content": [audio]}]
    if prompt_layout != "user":
        raise ValueError(f"unknown prompt_layout {prompt_layout!r}")
    return [{"role": "user", "content": [audio, {"type": "text", "text": instruction}]}]


def generate_sft(
    model, processor, model_key: str, instruction: str, audio_path: str, sr: int, max_new: int, device,
    prompt_layout: str = "user",
):
    """Same chat template as scripts/finetune/sequence_lora.py unless prompt_layout=system."""
    dtype = next(model.parameters()).dtype
    conversation = _sft_conversation(model_key, instruction, audio_path, prompt_layout)
    if model_key == "audio_flamingo3":
        inputs = processor.apply_chat_template(
            [conversation], tokenize=True, add_generation_prompt=True, return_dict=True,
        )
        casted = _tensors_to_device(inputs, device, dtype)
    else:
        audio = librosa.load(audio_path, sr=sr)[0]
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
        casted = _tensors_to_device(inputs, device, dtype)
    with torch.no_grad():
        gen = model.generate(**casted, max_new_tokens=max_new, do_sample=False)
    out_ids = gen[:, casted["input_ids"].shape[1] :]
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def generate_sft_batch(
    model, processor, model_key: str, instruction: str, audio_paths: list[str], sr: int, max_new: int, device,
    prompt_layout: str = "user",
) -> list[str]:
    """Batched ``generate_sft``: same template and greedy decoding, prompts left-padded."""
    if len(audio_paths) == 1:
        return [generate_sft(model, processor, model_key, instruction, audio_paths[0], sr, max_new, device,
                             prompt_layout=prompt_layout)]
    dtype = next(model.parameters()).dtype
    conversations = [_sft_conversation(model_key, instruction, p, prompt_layout) for p in audio_paths]
    tok = processor.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        if model_key == "audio_flamingo3":
            # Text-only padding: top-level padding=True would also pad mel features to the
            # longest clip instead of the fixed 30 s window the AF3 encoder expects.
            inputs = processor.apply_chat_template(
                conversations, tokenize=True, add_generation_prompt=True, return_dict=True,
                processor_kwargs={"text_kwargs": {"padding": True}},
            )
        else:
            audios = [librosa.load(p, sr=sr)[0] for p in audio_paths]
            prompts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                       for c in conversations]
            inputs = processor(text=prompts, audio=audios, sampling_rate=sr, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = prev_side
    casted = _tensors_to_device(inputs, device, dtype)
    with torch.no_grad():
        gen = model.generate(**casted, max_new_tokens=max_new, do_sample=False)
    out_ids = gen[:, casted["input_ids"].shape[1] :]
    return [t.strip() for t in processor.batch_decode(out_ids, skip_special_tokens=True)]


def generate_sft_chunked(
    model, processor, instruction: str, audio_path: str, sr: int, max_new: int, device, max_sec: float,
    prompt_layout: str = "user",
) -> tuple[str, list[dict], int]:
    """Qwen2-Audio SFT on audio longer than its 30 s encoder window: same windows and seam merge as
    ``generate_qwen2_chunked``, but each window uses the SFT chat template. Windows of one clip are
    decoded as one left-padded greedy batch."""
    audio = librosa.load(audio_path, sr=sr)[0]
    windows = split_audio(audio, sr, max_sec)
    conversation = _sft_conversation("qwen2_audio", instruction, audio_path, prompt_layout)
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    dtype = next(model.parameters()).dtype
    tok = processor.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        inputs = processor(
            text=[prompt] * len(windows), audio=[w for _, w in windows], sampling_rate=sr,
            return_tensors="pt", padding=True,
        )
    finally:
        tok.padding_side = prev_side
    casted = _tensors_to_device(inputs, device, dtype)
    with torch.no_grad():
        gen = model.generate(**casted, max_new_tokens=max_new, do_sample=False)
    out_ids = gen[:, casted["input_ids"].shape[1] :]
    outs = [t.strip() for t in processor.batch_decode(out_ids, skip_special_tokens=True)]
    texts: list[str] = []
    parsed: list[tuple[float, list[dict]]] = []
    for (offset, wav), text in zip(windows, outs):
        texts.append(f"# chunk offset={offset:.2f}s dur={len(wav)/sr:.2f}s\n{text}")
        parsed.append((offset, parse_prediction(text)))
    return "\n\n".join(texts), stitch_windows(parsed), len(windows)


def bitsandbytes_config(dtype):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
