#!/usr/bin/env python3
"""LoRA SFT for the bespoke ALM stacks (SALMONN, Kimi-Audio): discrete SER (--task utt) or
sequential SER (--task seq).

--task utt matches scripts/finetune/utterance_lora.py: utterance prompt
(configs/prompts/utterance.txt), target = the emotion name, whole clip, 3 epochs, no Dev
evaluation; the last epoch checkpoint (final_checkpoint.json) is the one evaluated.

--task seq matches scripts/finetune/sequence_lora.py (Qwen2-Audio / AF3) wherever the stacks allow:
  * LoRA r16 / alpha 32 / dropout 0.05 on the LLM q,k,v,o projections, audio encoders frozen,
    modality connector trained (emoseq/models/bespoke_lora.py)
  * same instruction (configs/prompts/sequence.txt) and "(start, end, emotion)" target lines
  * AdamW lr 2e-4, linear decay without warmup, grad-norm clip 1.0, micro-batch 1 x 8
    accumulation, 3 epochs, one checkpoint per epoch
  * train CE = per-example token-mean CE x sklearn-balanced length-bin weight (--length-bin-weight);
    Dev loss = unweighted per-example token-mean CE averaged over examples
  * Dev 256 during training; full-Dev selection afterwards (--eval-only-checkpoint, then
    scripts/finetune/select_checkpoint.py)

SALMONN: Whisper/BEATs take <= 30 s, so every sequence is cut into the non-overlapping 30 s
windows used at inference (scripts/inference/bespoke.py). A window's target is the sequence's
segments clipped to the window and re-timed from its start; each window carries
bin_weight / n_windows so a sequence keeps its total weight.
Kimi-Audio: whole sequence (its Whisper path slices 30 s internally); loss on the text stream
only, as in the released finetune code for understanding tasks.

Runs in the legacy environment (.venv-legacy):
  python scripts/finetune/bespoke_lora.py --model salmonn --task utt --output-dir runs/sft/utterance/salmonn
  python scripts/finetune/bespoke_lora.py --model salmonn --task seq --length-bin-weight \\
      --output-dir runs/sft/sequence/salmonn
  python scripts/finetune/bespoke_lora.py --model salmonn --task seq --output-dir runs/sft/sequence/salmonn \\
      --eval-limit 0 --eval-only-checkpoint runs/sft/sequence/salmonn/checkpoint-* \\
      --eval-output-dir runs/sft/sequence/salmonn/full_dev
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import random
import time
from pathlib import Path

from emoseq import config

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from emoseq.bins import apply_bin_weights, bin_of_record, sklearn_balanced_bin_weights
from emoseq.models import bespoke
from emoseq.models.bespoke_lora import kimi_add_lora, load_trainable, salmonn_add_lora, save_trainable
from emoseq.scoring.utterance import EMOTION_NAMES
from emoseq.targets import format_target, target_string_from_record
from emoseq.windows import QWEN2_MAX_SEC

SR = 16000
MIN_WINDOW_SEC = 1.0
MIN_PIECE_SEC = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["salmonn", "kimi_audio"])
    p.add_argument("--task", default="seq", choices=["utt", "seq"])
    p.add_argument("--train-jsonl", default=None,
                   help="Default: data/utterances/msp_podcast_train.jsonl (utt), data/sequences/train.jsonl (seq).")
    p.add_argument("--eval-jsonl", default=None,
                   help="Default: none (utt), data/sequences/dev.jsonl (seq).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prompt", default=None,
                   help="Default: configs/prompts/utterance.txt (utt), configs/prompts/sequence.txt (seq).")
    p.add_argument("--limit", type=int, default=None, help="First N training rows (smoke tests).")
    p.add_argument("--eval-limit", type=int, default=256, help="First N Dev rows; 0 = all.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (smoke tests).")
    p.add_argument("--length-bin-weight", action="store_true")
    p.add_argument("--window-sec", type=float, default=QWEN2_MAX_SEC, help="SALMONN window (encoder limit).")
    p.add_argument("--no-gradient-checkpointing", action="store_true", help="Kimi-Audio only.")
    p.add_argument("--resume", action="store_true", help="Continue from the latest epoch checkpoint.")
    p.add_argument("--eval-only-checkpoint", nargs="+", default=None)
    p.add_argument("--eval-output-dir", default=None, help="One <checkpoint>.json per --eval-only-checkpoint.")
    args = p.parse_args()
    utt = args.task == "utt"
    if args.train_jsonl is None:
        args.train_jsonl = str(config.utterance_jsonl("train") if utt else config.sequence_jsonl("train"))
    if args.eval_jsonl is None and not utt:
        args.eval_jsonl = str(config.sequence_jsonl("dev"))
    if args.prompt is None:
        args.prompt = str(config.UTTERANCE_PROMPT if utt else config.SEQUENCE_PROMPT)
    if utt and args.length_bin_weight:
        p.error("--length-bin-weight applies to --task seq only")
    return args


def load_rows(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def utterance_items(rows: list[dict]) -> list[dict]:
    return [{"seq_id": r.get("file_name") or str(i), "audio_path": r["audio_path"], "offset": 0.0,
             "duration": None, "target": EMOTION_NAMES[int(r["label"])], "weight": 1.0}
            for i, r in enumerate(rows)]


def whole_items(rows: list[dict]) -> list[dict]:
    return [{"seq_id": r["seq_id"], "audio_path": r["audio_path"], "offset": 0.0, "duration": None,
             "target": target_string_from_record(r), "weight": float(r.get("bin_weight") or 1.0),
             "length_bin": r.get("length_bin") or bin_of_record(r)} for r in rows]


def window_items(rows: list[dict], win: float) -> list[dict]:
    """Same window grid as emoseq.windows.split_audio (non-overlapping, from 0)."""
    out = []
    for r in rows:
        dur = float(r["audio_duration"])
        segs = r["segments"]
        spans = [(0.0, dur)] if dur <= win else [
            (k * win, min(dur, (k + 1) * win)) for k in range(math.ceil(dur / win))
        ]
        wins = []
        for a, b in spans:
            if len(spans) > 1 and b - a < MIN_WINDOW_SEC:
                continue
            clipped = [
                {"start": max(float(s["start"]), a) - a, "end": min(float(s["end"]), b) - a, "emotion": s["emotion"]}
                for s in segs
                if min(float(s["end"]), b) - max(float(s["start"]), a) >= MIN_PIECE_SEC
            ]
            if clipped:
                wins.append((a, b, clipped))
        w = float(r.get("bin_weight") or 1.0)
        for a, b, clipped in wins:
            out.append({
                "seq_id": r["seq_id"] if len(spans) == 1 else f"{r['seq_id']}#w{a:.0f}",
                "audio_path": r["audio_path"], "offset": a, "duration": None if len(spans) == 1 else b - a,
                "target": target_string_from_record(r) if len(spans) == 1 else format_target(clipped),
                "weight": w / len(wins), "length_bin": r.get("length_bin") or bin_of_record(r),
            })
    return out


def load_audio(item: dict) -> np.ndarray:
    return librosa.load(item["audio_path"], sr=SR, offset=item["offset"], duration=item["duration"])[0]


# ---------------------------------------------------------------- model wrappers
class SalmonnSFT:
    """SALMONN with its released LoRA merged and a fresh r16 adapter; fp16 autocast + GradScaler."""

    fp16 = True

    def __init__(self, instruction: str) -> None:
        self.backend = bespoke.Salmonn()
        m = self.backend.model
        salmonn_add_lora(m)
        self.module = m
        self.prompt = self.backend.cfg.config.model.prompt_template.format(
            "<Speech><SpeechHere></Speech>\n" + instruction.strip())
        self.tok = m.llama_tokenizer
        self.embed = m.llama_model.model.model.embed_tokens
        for p in m.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    def train_mode(self, on: bool) -> None:
        m = self.module
        m.train(on)
        m.speech_encoder.eval()
        if getattr(m, "beats", None) is not None:
            m.beats.eval()

    def loss(self, item: dict) -> torch.Tensor:
        m = self.module
        samples = self.backend._samples(load_audio(item))  # noqa: SLF001
        speech, atts = m.encode_speech(samples["spectrogram"], raw_wav=samples["raw_wav"],
                                       audio_padding_mask=samples["padding_mask"])
        speech, atts = m.prompt_wrap(speech, atts, [self.prompt], multi_prompt=True)
        tgt = self.tok([item["target"] + m.end_sym], return_tensors="pt", add_special_tokens=False).to(speech.device)
        ids = tgt.input_ids
        bos = torch.full((1, 1), self.tok.bos_token_id, dtype=ids.dtype, device=ids.device)
        bos_emb, tgt_emb = self.embed(bos), self.embed(ids)
        embeds = torch.cat([bos_emb, speech.to(bos_emb.dtype), tgt_emb], dim=1)
        att = torch.cat([atts[:, :1], atts, tgt.attention_mask], dim=1)
        with m.maybe_autocast():
            logits = m.llama_model(inputs_embeds=embeds, attention_mask=att, return_dict=True).logits
        n = ids.shape[1]
        return F.cross_entropy(logits[0, -n - 1:-1].float(), ids[0])


def _checkpointed(fn, *args, **kwargs):
    if torch.is_grad_enabled():
        return torch_ckpt.checkpoint(fn, *args, use_reentrant=False, **kwargs)
    return fn(*args, **kwargs)


class KimiSFT:
    """Kimi-Audio-7B-Instruct (bf16) + r16 adapter; inputs built by the inference prompt manager."""

    fp16 = False

    def __init__(self, instruction: str, grad_ckpt: bool = True) -> None:
        self.backend = bespoke.KimiAudioBackend()
        api = self.backend.model
        self.pm = api.prompt_manager
        # Semantic tokens of a file never change; cache them across epochs.
        self.pm._tokenize_audio = functools.lru_cache(maxsize=None)(self.pm._tokenize_audio)  # noqa: SLF001
        self.module = kimi_add_lora(api.alm)
        self.instruction = instruction.strip()
        if grad_ckpt:
            for layer in self.module.base_model.model.model.layers:
                layer.forward = functools.partial(_checkpointed, layer.forward)
        for p in self.module.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.device = torch.device("cuda", torch.cuda.current_device())

    def train_mode(self, on: bool) -> None:
        self.module.train(on)

    def loss(self, item: dict) -> torch.Tensor:
        msgs = [
            {"role": "user", "message_type": "text", "content": self.instruction},
            {"role": "user", "message_type": "audio", "content": item["audio_path"]},
            {"role": "assistant", "message_type": "text", "content": item["target"]},
        ]
        content = self.pm.get_prompt(msgs, output_type="text", add_assistant_start_msg=False)
        a_ids, t_ids, cont_mask, _, t_loss_mask = content.to_tensor()
        dev = self.device
        feats = [f.to(dev) for f in content.continuous_feature]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.module(input_ids=a_ids.to(dev), text_input_ids=t_ids.to(dev), whisper_input_feature=feats,
                              is_continuous_mask=cont_mask.to(dev), use_cache=False, return_dict=False)
        text_logits = out[1]
        labels = t_ids[0, 1:].to(dev)
        mask = t_loss_mask[0, 1:].to(dev)
        return F.cross_entropy(text_logits[0, :-1][mask].float(), labels[mask])


# ---------------------------------------------------------------- loops
@torch.no_grad()
def evaluate(sft, items: list[dict]) -> float:
    sft.train_mode(False)
    losses = [sft.loss(it).item() for it in items]
    sft.train_mode(True)
    return float(np.mean(losses))


def latest_checkpoint(out_dir: Path) -> Path | None:
    cks = [c for c in out_dir.glob("checkpoint-*") if (c / "optim.pt").is_file()]
    return max(cks, key=lambda c: int(c.name.split("-")[1])) if cks else None


def train(args, sft, train_items: list[dict], eval_items: list[dict], out_dir: Path) -> None:
    params = [p for p in sft.module.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_items) / args.grad_accum)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total))
    scaler = torch.cuda.amp.GradScaler(enabled=sft.fp16)
    state = {"log_history": [], "global_step": 0, "epoch": 0, "steps_per_epoch": steps_per_epoch,
             "total_steps": total}
    if args.resume and (ck := latest_checkpoint(out_dir)) is not None:
        load_trainable(sft.module, ck)
        saved = torch.load(ck / "optim.pt", map_location="cpu")
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = json.loads((ck / "trainer_state.json").read_text())
        print(f"resumed from {ck} (epoch {state['epoch']}, step {state['global_step']})")

    done = False
    for epoch in range(state["epoch"], args.epochs):
        order = list(range(len(train_items)))
        random.Random(args.seed + epoch).shuffle(order)
        sft.train_mode(True)
        opt.zero_grad(set_to_none=True)
        acc, acc_n, window, t0 = 0.0, 0, [], time.time()
        for i, idx in enumerate(order):
            item = train_items[idx]
            loss = sft.loss(item) * (item["weight"] if args.length_bin_weight else 1.0)
            scaler.scale(loss / args.grad_accum).backward()
            acc += loss.item()
            acc_n += 1
            if (i + 1) % args.grad_accum and i + 1 < len(order):
                continue
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            state["global_step"] += 1
            window.append(acc / acc_n)
            acc, acc_n = 0.0, 0
            step = state["global_step"]
            if step % args.logging_steps == 0:
                entry = {"loss": round(float(np.mean(window)), 4), "grad_norm": round(float(grad_norm), 4),
                         "learning_rate": sched.get_last_lr()[0], "epoch": round(epoch + (i + 1) / len(order), 4),
                         "step": step, "sec_per_step": round((time.time() - t0) / len(window), 2)}
                state["log_history"].append(entry)
                print(json.dumps(entry), flush=True)
                window, t0 = [], time.time()
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        ck = out_dir / f"checkpoint-{state['global_step']}"
        save_trainable(sft.module, ck)
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict()},
                   ck / "optim.pt")
        state["epoch"] = epoch + 1
        if eval_items:
            ev = evaluate(sft, eval_items)
            entry = {"eval_loss": ev, "epoch": epoch + 1, "step": state["global_step"], "n_eval": len(eval_items)}
            state["log_history"].append(entry)
            print(json.dumps(entry), flush=True)
        for path in (ck / "trainer_state.json", out_dir / "trainer_state.json"):
            path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        if done:
            break

    (out_dir / "final_checkpoint.json").write_text(json.dumps({
        "final_checkpoint": str(out_dir / f"checkpoint-{state['global_step']}"),
        "global_step": state["global_step"], "epoch": state["epoch"],
    }, indent=2) + "\n", encoding="utf-8")
    evals = [e for e in state["log_history"] if "eval_loss" in e]
    if evals:
        best = min(evals, key=lambda e: e["eval_loss"])
        (out_dir / "best_checkpoint.json").write_text(json.dumps({
            "best_model_checkpoint": str(out_dir / f"checkpoint-{best['step']}"),
            "best_metric": best["eval_loss"], "metric_for_best_model": "eval_loss",
        }, indent=2) + "\n", encoding="utf-8")


def eval_only(args, sft, eval_items: list[dict]) -> None:
    out_dir = Path(args.eval_output_dir or Path(args.output_dir) / "full_dev")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ck in args.eval_only_checkpoint:
        ck = Path(ck)
        load_trainable(sft.module, ck)
        loss = evaluate(sft, eval_items)
        rec = {"checkpoint": str(ck), "model": args.model, "eval_jsonl": args.eval_jsonl,
               "n_eval": len(eval_items), "eval_loss": loss}
        (out_dir / f"{ck.name}.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"eval_only {ck.name} n={len(eval_items)} eval_loss={loss:.4f}", flush=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    instruction = Path(args.prompt).read_text(encoding="utf-8").strip()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(args.train_jsonl, args.limit)
    eval_rows = load_rows(args.eval_jsonl, args.eval_limit or None) if args.eval_jsonl else []
    if args.length_bin_weight:
        wmap = sklearn_balanced_bin_weights(train_rows)
        apply_bin_weights(train_rows, wmap)
        print(f"length-bin weights from Train: {wmap}")
    if args.task == "utt":
        to_items = utterance_items
    elif args.model == "salmonn":
        to_items = lambda rows: window_items(rows, args.window_sec)  # noqa: E731
    else:
        to_items = whole_items
    train_items, eval_items = to_items(train_rows), to_items(eval_rows)
    print(f"{args.model} {args.task} | train {len(train_rows)} rows -> {len(train_items)} items | "
          f"dev {len(eval_rows)} -> {len(eval_items)} | out {out_dir}")

    if args.model == "salmonn":
        sft = SalmonnSFT(instruction)
    else:
        sft = KimiSFT(instruction, grad_ckpt=not args.no_gradient_checkpointing)

    if args.eval_only_checkpoint:
        eval_only(args, sft, eval_items)
        return

    (out_dir / "prompt_train.txt").write_text(instruction + "\n", encoding="utf-8")
    (out_dir / "train_config.json").write_text(json.dumps({
        **{k: v for k, v in vars(args).items()}, "n_train_sequences": len(train_rows),
        "n_train_items": len(train_items), "n_eval_items": len(eval_items),
        "bin_weights": sklearn_balanced_bin_weights(train_rows) if args.length_bin_weight else None,
    }, indent=2) + "\n", encoding="utf-8")
    train(args, sft, train_items, eval_items, out_dir)


if __name__ == "__main__":
    main()
