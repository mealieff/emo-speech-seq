#!/usr/bin/env python3
"""Pick the sequence-SFT checkpoint with the lowest full-Dev eval loss.

Reads ``<sft_dir>/full_dev/checkpoint-*.json`` (written by ``sequence_lora.py
--eval-only-checkpoint``), writes ``<sft_dir>/selection_full_dev.json`` and prints the
selected checkpoint path.

    python scripts/finetune/select_checkpoint.py runs/sft/sequence/qwen2_audio
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def select(sft_dir: str) -> str:
    d = Path(sft_dir)
    rows = []
    for f in sorted((d / "full_dev").glob("checkpoint-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        r = json.loads(f.read_text())
        rows.append({"checkpoint": r["checkpoint"], "n_eval": r["n_eval"], "eval_loss": r["eval_loss"]})
    if not rows:
        raise SystemExit(f"no full-Dev evals under {d / 'full_dev'}")
    best = min(rows, key=lambda r: r["eval_loss"])
    dev256 = json.loads((d / "best_checkpoint.json").read_text()) if (d / "best_checkpoint.json").is_file() else {}
    state = json.loads((d / "trainer_state.json").read_text()) if (d / "trainer_state.json").is_file() else {}
    dev256_losses = [
        {"epoch": h.get("epoch"), "step": h.get("step"), "eval_loss": h["eval_loss"]}
        for h in state.get("log_history", []) if "eval_loss" in h
    ]
    same = Path(best["checkpoint"]).name == Path(dev256.get("best_model_checkpoint") or "").name
    out = {
        "selection": "lowest full-Dev eval_loss (unweighted token CE, Trainer eval path)",
        "full_dev": rows,
        "best_checkpoint": best["checkpoint"],
        "best_full_dev_eval_loss": best["eval_loss"],
        "dev256_best_checkpoint": dev256.get("best_model_checkpoint"),
        "dev256_eval_losses": dev256_losses,
        "selection_unchanged_vs_dev256": same,
    }
    (d / "selection_full_dev.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return best["checkpoint"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sft_dir")
    print(select(p.parse_args().sft_dir))


if __name__ == "__main__":
    main()
