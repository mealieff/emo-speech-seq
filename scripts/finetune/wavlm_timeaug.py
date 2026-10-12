#!/usr/bin/env python3
"""Train WavLM-TimeAug on the sequence manifests (Train; Dev for early stopping / threshold).

Initialised from the IS2025 seed-7 categorical checkpoint (``WAVLM_IS2025_DIR``). Saves
``best.pt`` (best 0.5 * (Dev macro-F1 + boundary AP)), ``best_ap.pt``, ``best_f1.pt``,
``last.pt``, ``history.json`` and per-epoch Dev reports to ``--ckpt-dir``.

    python scripts/finetune/wavlm_timeaug.py --seed 7 --overfit 32   # sanity check
    python scripts/finetune/wavlm_timeaug.py --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from emoseq import config

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from emoseq.bins import BIN_NAMES
from emoseq.io import load_jsonl, write_json
from emoseq.scoring.report import format_quartile_table, quartile_report
from emoseq.wavlm import WAVLM_ID, WAVLM_REVISION
from emoseq.wavlm.dataset import (
    MAX_AUDIO_SEC,
    BinBalancedSampler,
    SeqDataset,
    class_inv_weights,
    reweight_votes,
    stratified_subset,
)
from emoseq.wavlm.decode import (
    boundary_ap,
    boundary_f1,
    bounds_to_intervals,
    gold_to_e5,
    local_maxima,
    peak_pick,
    sweep_theta,
    to_pred_segments,
)
from emoseq.wavlm.model import (
    TARGET_SIGMA,
    FrameASP,
    WavLMSeq,
    anneal_sigma,
    avg_pool_to_10hz,
    freeze_bottom_layers,
    gaussian_boundary_target,
    heatmap_focal,
    load_norm_stat,
)

CAT_SER = config.WAVLM_IS2025_DIR
MANIFESTS = config.MANIFEST_DIR
CKPT_ROOT = config.RUNS_DIR / "wavlm_timeaug"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr-enc", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lambda-b", type=float, default=0.0, help="0 = set so focal ≈ KL at init")
    p.add_argument("--theta-b", type=float, default=0.05, help="fallback until Dev sweep")
    p.add_argument("--sigma", type=float, default=TARGET_SIGMA)
    p.add_argument("--max-audio-sec", type=float, default=MAX_AUDIO_SEC)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0, help=">0 = cap optimizer-scale steps")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--overfit", type=int, default=0, help="if >0, train/eval on N stratified Train sequences")
    p.add_argument("--ckpt-dir", default="", help="default: $EMOSEQ_RUNS/wavlm_timeaug/seed<seed>")
    p.add_argument("--resume", default="", help="resume from a checkpoint (model+opt+theta+lambda_b)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model() -> WavLMSeq:
    ssl = AutoModel.from_pretrained(WAVLM_ID, revision=WAVLM_REVISION)
    ssl.load_state_dict(load_pt(CAT_SER / "final_ssl.pt"))
    freeze_bottom_layers(ssl)
    asp = FrameASP(ssl.config.hidden_size)
    asp.load_state_dict(load_pt(CAT_SER / "final_pool.pt"))
    return WavLMSeq(ssl, asp)


def jitter_intervals(intervals: list[tuple[float, float]], rng: random.Random, p: float = 0.5, max_shift: float = 0.3):
    if len(intervals) < 2 or rng.random() > p:
        return intervals
    bounds = [intervals[0][0]] + [iv[1] for iv in intervals]
    start, end = bounds[0], bounds[-1]
    new_b = [start]
    for b in bounds[1:-1]:
        x = b + rng.uniform(-max_shift, max_shift)
        x = min(max(x, new_b[-1] + 0.05), end - 0.05)
        new_b.append(x)
    new_b.append(end)
    return [(float(a), float(b)) for a, b in zip(new_b, new_b[1:]) if b > a + 1e-4]


def overlap_assign(pred_spans, gold_spans, gold_votes, frac_thr=0.7):
    votes_out = []
    mask = []
    g_votes = [np.asarray(v, dtype=np.float64) for v in gold_votes]
    for on, off in pred_spans:
        w = np.array(
            [max(0.0, min(off, g1) - max(on, g0)) for (g0, g1) in gold_spans],
            dtype=np.float64,
        )
        tot = float(w.sum())
        if tot <= 1e-6:
            votes_out.append(np.zeros(8, dtype=np.float32))
            mask.append(False)
            continue
        soft = sum(w[i] * g_votes[i] for i in range(len(g_votes))) / tot
        mixed = (w.max() / tot) < frac_thr
        votes_out.append(soft.astype(np.float32))
        mask.append(not mixed)
    return torch.tensor(np.stack(votes_out)), mask


@torch.no_grad()
def infer_record(model: WavLMSeq, wav: torch.Tensor, duration: float, theta: float, oracle=None):
    frames_emo, frames_bound = model.encode_frames(wav)
    b_logits = model.boundary_logits(frames_bound)
    if oracle is not None:
        spans = list(oracle)
    else:
        peaks = peak_pick(b_logits, theta, duration)
        spans = bounds_to_intervals(peaks, duration)
    e_logits = model.emotion_logits(frames_emo, spans)
    return b_logits, e_logits, spans, frames_emo


@torch.no_grad()
def eval_split(model, loader, theta, device, max_items=0, tune_theta=True):
    """One encode pass. Emotion/F1 use ``theta``; AP is threshold-free; sweep updates θ for the next epoch."""
    model.eval()
    rows = []
    packed = []
    aps = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            wav = batch["wav"].to(device)
            rec = batch["rec"]
            dur = float(batch["duration"])
            gold = batch["intervals"]
            internals = [iv[1] for iv in gold[:-1]]
            b_logits, e_logits, spans, _ = infer_record(model, wav, dur, theta)
            times, scores = local_maxima(b_logits, dur)
            ap_i = boundary_ap(times, scores, internals)
            aps.append(ap_i)
            packed.append((b_logits.detach().cpu(), dur, internals))
            rows.append(
                {
                    "seq_id": rec["seq_id"],
                    "seq_len": rec["seq_len"],
                    "length_bin": rec.get("length_bin") or rec.get("bin"),
                    "audio_duration": dur,
                    "ground_truth": gold_to_e5(rec),
                    "pred": to_pred_segments(spans, e_logits),
                    "boundary_f1": boundary_f1(rec["gold_intervals"], spans),
                    "boundary_ap": ap_i,
                }
            )
            if max_items and len(rows) >= max_items:
                break
    theta_info = {"best_theta": float(theta), "best_f1": float(np.mean([r["boundary_f1"] for r in rows]) if rows else 0.0)}
    if tune_theta and packed:
        theta_info = sweep_theta(packed)
    report = quartile_report(rows)
    bf1 = float(np.mean([r["boundary_f1"] for r in rows])) if rows else 0.0
    ap = float(np.mean(aps)) if aps else 0.0
    ap_by_bin: dict[str, list[float]] = defaultdict(list)
    for r, a in zip(rows, aps):
        ap_by_bin[str(r.get("length_bin") or "")].append(float(a))
    report["boundary_ap"] = ap
    report["boundary_ap_by_bin"] = {b: float(np.mean(ap_by_bin[b])) for b in BIN_NAMES if ap_by_bin.get(b)}
    macro = float((report.get("additional") or {}).get("overall", {}).get("macro_f1") or 0.0)
    combined = 0.5 * (macro + ap)
    return report, bf1, ap, combined, theta_info, rows


def layer_mix_report(model: WavLMSeq) -> dict:
    w = torch.softmax(model.layer_weights.detach().float().cpu(), dim=0).numpy()
    # hidden_states[0]=embed; [1:13]=encoder 0–11 (frozen); [13:25]=encoder 12–23
    return {
        "n": int(w.size),
        "embed": float(w[0]) if w.size else 0.0,
        "bottom12": float(w[1:13].sum()) if w.size >= 13 else 0.0,
        "top12": float(w[13:].sum()) if w.size > 13 else 0.0,
        "argmax": int(w.argmax()) if w.size else -1,
        "weights": [round(float(x), 6) for x in w],
    }


def collate_one(batch):
    assert len(batch) == 1
    return batch[0]


def losses_one(model, batch, inv, rng, device, lambda_b, sigma, use_pred, theta):
    wav = batch["wav"].to(device)
    gold = batch["intervals"]
    votes = batch["votes"].to(device)
    dur = float(batch["duration"])
    frames_emo, frames_bound = model.encode_frames(wav)
    b_logits = model.boundary_logits(frames_bound)
    n_10 = avg_pool_to_10hz(frames_bound).size(0)
    g_b = gaussian_boundary_target(n_10, [iv[1] for iv in gold[:-1]], sigma=sigma).to(device)
    n_b = min(b_logits.numel(), g_b.numel())
    loss_b = heatmap_focal(b_logits[:n_b], g_b[:n_b])

    if use_pred:
        peaks = peak_pick(b_logits.detach(), theta, dur)
        spans = bounds_to_intervals(peaks, dur)
        soft, mask = overlap_assign(spans, gold, votes.detach().cpu().numpy())
        soft = reweight_votes(soft.to(device), inv)
        use_mask = mask
    else:
        spans = jitter_intervals(gold, rng, p=0.5)
        soft = reweight_votes(votes, inv)
        use_mask = [True] * len(spans)
        if len(spans) != votes.size(0):
            soft, use_mask = overlap_assign(spans, gold, votes.detach().cpu().numpy())
            soft = reweight_votes(soft.to(device), inv)

    e_logits = model.emotion_logits(frames_emo, spans)
    k = min(e_logits.size(0), soft.size(0), len(use_mask))
    loss_e = frames_emo.new_zeros(())
    n_k = 0
    for i in range(k):
        if not use_mask[i]:
            continue
        loss_e = loss_e + F.kl_div(
            F.log_softmax(e_logits[i].float(), dim=-1),
            soft[i].float().clamp_min(1e-8),
            reduction="sum",
        )
        n_k += 1
    if n_k:
        loss_e = loss_e / n_k
    loss = loss_e + lambda_b * loss_b
    return loss, loss_e.detach(), loss_b.detach()


@torch.no_grad()
def estimate_lambda_b(model, loader, inv, rng, device, sigma, n=8) -> float:
    model.eval()
    ks, bs = [], []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        _, le, lb = losses_one(model, batch, inv, rng, device, 1.0, sigma, False, 0.05)
        ks.append(float(le))
        bs.append(float(lb))
    kl = float(np.mean(ks)) if ks else 1.0
    foc = float(np.mean(bs)) if bs else 1.0
    lam = kl / max(foc, 1e-6)
    lam = float(min(max(lam, 0.1), 20.0))
    print(f"lambda_b auto: kl={kl:.4f} focal={foc:.4f} lambda_b={lam:.4f}", flush=True)
    model.train()
    return lam


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"overfit{args.overfit}" if args.overfit else f"seed{args.seed}"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_ROOT / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wav_mean, wav_std = load_norm_stat(CAT_SER / "train_norm_stat.pkl")
    train_rows = load_jsonl(MANIFESTS / "train.jsonl")
    if args.overfit:
        train_rows = stratified_subset(train_rows, n=int(args.overfit), seed=args.seed)
        print(f"overfit n={len(train_rows)} bins={ {r.get('length_bin') or r.get('bin') for r in train_rows} }", flush=True)
        train_ds = SeqDataset(None, wav_mean, wav_std, crop_sec=None, seed=args.seed, rows=train_rows)
        dev_ds = SeqDataset(None, wav_mean, wav_std, crop_sec=None, seed=args.seed, rows=train_rows)
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_one, pin_memory=True
        )
        dev_loader = DataLoader(
            dev_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_one
        )
    else:
        train_ds = SeqDataset(MANIFESTS / "train.jsonl", wav_mean, wav_std, crop_sec=args.max_audio_sec, seed=args.seed)
        dev_ds = SeqDataset(MANIFESTS / "dev.jsonl", wav_mean, wav_std, crop_sec=None, seed=args.seed)
        sampler = BinBalancedSampler(train_ds.rows, seed=args.seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_one,
            pin_memory=True,
        )
        dev_loader = DataLoader(
            dev_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_one,
            pin_memory=True,
        )

    inv = class_inv_weights(train_ds.rows).to(device)
    print("building WavLM-TimeAug from the IS2025 cat_ser checkpoint", flush=True)
    model = build_model().to(device)
    enc_params = [p for p in model.ssl.parameters() if p.requires_grad]
    head_params = list(model.bound_lstm.parameters()) + list(model.bound_head.parameters())
    head_params += list(model.seg_lstm.parameters()) + list(model.emo_head.parameters())
    head_params += list(model.asp.parameters()) + [model.layer_weights]
    opt = torch.optim.Adam(
        [{"params": enc_params, "lr": args.lr_enc}, {"params": head_params, "lr": args.lr_head}]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = random.Random(args.seed)
    if args.overfit:
        args.accum = 1
        print(f"overfit: accum=1 (one update per sequence)", flush=True)

    start_epoch = 1
    best = -1.0
    best_ap = -1.0
    best_f1 = -1.0
    bad = 0
    global_step = 0
    theta = float(args.theta_b)
    history: list[dict] = []
    lambda_b = float(args.lambda_b)

    resume_path = Path(args.resume) if args.resume else None
    if resume_path and resume_path.is_file():
        blob = load_pt(resume_path)
        model.load_state_dict(blob["model"])
        if blob.get("opt"):
            opt.load_state_dict(blob["opt"])
        start_epoch = int(blob.get("epoch") or 0) + 1
        theta = float(blob.get("theta") or theta)
        lambda_b = float(blob.get("lambda_b") or lambda_b)
        best = float(blob.get("best") or -1.0)
        hist_path = ckpt_dir / "history.json"
        if hist_path.is_file():
            history = [h for h in json.loads(hist_path.read_text()) if int(h.get("epoch") or 0) < start_epoch]
            if history:
                best_ap = max(float(h.get("dev_boundary_ap") or -1) for h in history)
                best_f1 = max(float(h.get("dev_macro_f1") or -1) for h in history)
        print(
            f"resume {resume_path} next_epoch={start_epoch} theta={theta:.3f} "
            f"lambda_b={lambda_b:.4f} best={best:.4f}",
            flush=True,
        )

    if lambda_b <= 0:
        lambda_b = estimate_lambda_b(model, train_loader, inv, rng, device, args.sigma, n=8)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss = run_e = run_b = 0.0
        n_seen = 0
        sigma = anneal_sigma(epoch, args.epochs, start=args.sigma)
        sched = (not args.overfit) and epoch >= 11
        for step, batch in enumerate(train_loader, 1):
            use_pred = sched and rng.random() < 0.5
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, loss_e, loss_b = losses_one(
                    model, batch, inv, rng, device, lambda_b, sigma, use_pred, theta
                )
                loss = loss / args.accum
            scaler.scale(loss).backward()
            run_loss += float(loss.item()) * args.accum
            run_e += float(loss_e)
            run_b += float(loss_b)
            n_seen += 1
            global_step += 1
            if step % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={run_loss / max(n_seen, 1):.4f} "
                    f"kl={run_e / max(n_seen, 1):.4f} "
                    f"focal={run_b / max(n_seen, 1):.4f} "
                    f"lam*focal={lambda_b * run_b / max(n_seen, 1):.4f} "
                    f"sigma={sigma:.3f}",
                    flush=True,
                )
            if args.max_steps and global_step >= args.max_steps:
                break
        if n_seen and step % args.accum != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        mix = layer_mix_report(model)
        rec = {
            "epoch": epoch,
            "train_loss": run_loss / max(n_seen, 1),
            "train_kl": run_e / max(n_seen, 1),
            "train_focal": run_b / max(n_seen, 1),
            "train_lam_focal": lambda_b * run_b / max(n_seen, 1),
            "focal_over_kl": (run_b / max(run_e, 1e-8)),
            "lambda_b": lambda_b,
            "sigma": sigma,
            "steps": n_seen,
            "layer_mix": {k: mix[k] for k in ("embed", "bottom12", "top12", "argmax")},
        }
        do_eval = (epoch % args.eval_every == 0) or (args.max_steps and global_step >= args.max_steps)
        if do_eval:
            max_dev = 0
            if args.max_steps and not args.overfit:
                max_dev = 32
            report, bf1, ap, combined, theta_info, _ = eval_split(
                model, dev_loader, theta, device, max_items=max_dev, tune_theta=True
            )
            theta = float(theta_info.get("best_theta") or theta)
            rec.update(
                {
                    "dev_macro_f1": float((report.get("additional") or {}).get("overall", {}).get("macro_f1") or 0),
                    "dev_boundary_f1": bf1,
                    "dev_boundary_ap": ap,
                    "dev_boundary_ap_by_bin": report.get("boundary_ap_by_bin") or {},
                    "dev_combined": combined,
                    "best_theta": theta,
                    "theta_sweep_best_f1": theta_info.get("best_f1"),
                }
            )
            print(format_quartile_table(report), flush=True)
            print(
                f"epoch {epoch} combined={combined:.4f} macroF1={rec['dev_macro_f1']:.4f} "
                f"AP={ap:.4f} ap_bins={rec['dev_boundary_ap_by_bin']} "
                f"bf1={bf1:.4f} theta={theta:.3f} "
                f"mix embed={mix['embed']:.3f} bot={mix['bottom12']:.3f} top={mix['top12']:.3f}",
                flush=True,
            )
            write_json(
                ckpt_dir / f"dev_epoch{epoch}.json",
                {"report": report, "theta": theta_info, "rec": rec, "layer_mix": mix},
            )
            payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best": best,
                "theta": theta,
                "lambda_b": lambda_b,
                "args": vars(args),
            }
            if combined > best:
                best = combined
                payload["best"] = best
                bad = 0
                torch.save(payload, ckpt_dir / "best.pt")
                print(f"saved {ckpt_dir / 'best.pt'}", flush=True)
            else:
                bad += 1
            if ap > best_ap:
                best_ap = ap
                torch.save({**payload, "best": best_ap}, ckpt_dir / "best_ap.pt")
            if rec["dev_macro_f1"] > best_f1:
                best_f1 = rec["dev_macro_f1"]
                torch.save({**payload, "best": best_f1}, ckpt_dir / "best_f1.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best": best,
                "theta": theta,
                "lambda_b": lambda_b,
                "args": vars(args),
            },
            ckpt_dir / "last.pt",
        )
        history.append(rec)
        write_json(ckpt_dir / "history.json", history)
        if args.max_steps and global_step >= args.max_steps:
            break
        if (not args.overfit) and bad >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break
    print(json.dumps({"best": best, "theta": theta, "ckpt": str(ckpt_dir / "best.pt")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
