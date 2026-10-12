#!/usr/bin/env python3
"""Write discrete-SER Test1 outputs as one label per utterance.

The archived prediction is the whole model string when that string is exactly one
emotion name. A glued repetition is left unparsed (pred null). It is not read as the
first name inside the string.

    python scripts/evaluate/export_utterance_one_label.py runs/test1/utterance_one_label
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from emoseq.config import RUNS_DIR
from emoseq.scoring.utterance import LABEL_TO_INT, task_a_metrics

SOURCES = [
    ("qwen2_audio_sft", RUNS_DIR / "test1/utterance/qwen2_audio_sft/predictions_test1.json"),
    ("qwen2_audio_pt", RUNS_DIR / "test1/utterance/qwen2_audio_pt/predictions_test1.json"),
    ("audio_flamingo3_sft", RUNS_DIR / "test1/utterance/audio_flamingo3_sft/predictions_test1.json"),
    ("audio_flamingo3_pt", RUNS_DIR / "test1/utterance/audio_flamingo3_pt/predictions_test1.json"),
    ("wavlm_is2025", RUNS_DIR / "test1/utterance/wavlm_is2025/cat_test1_preds.json"),
]


def _key(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def label_table(records: list[dict]) -> dict[str, tuple[int, str]]:
    table: dict[str, tuple[int, str]] = {}
    for rec in records:
        if "label_name" in rec:
            table[_key(rec["label_name"])] = (int(rec["label"]), rec["label_name"])
        elif "gold_name" in rec:
            table[_key(rec["gold_name"])] = (int(rec["gold"]), rec["gold_name"])
        if rec.get("pred_name") is not None and rec.get("pred") is not None:
            table.setdefault(_key(rec["pred_name"]), (int(rec["pred"]), rec["pred_name"]))
    for alias, canon in (("surprise", "surprised"), ("surprised", "surprise")):
        if alias not in table and canon in table:
            table[alias] = table[canon]
        if alias not in table and canon in LABEL_TO_INT:
            table[alias] = (LABEL_TO_INT[canon], canon[:1].upper() + canon[1:])
    return table


def row_fields(rec: dict) -> tuple[int, str, str, str | None]:
    if "FileName" in rec:
        return int(rec["gold"]), rec["gold_name"], rec["FileName"], rec.get("pred_name")
    name = Path(rec.get("audio_path") or "").name
    return int(rec["label"]), rec["label_name"], name, rec.get("raw_text")


def export_one(dest_root: Path, system: str, src: Path) -> None:
    records = json.loads(src.read_text(encoding="utf-8"))["records"]
    table = label_table(records)
    out_dir = dest_root / system
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true: list[int] = []
    y_pred: list[int] = []
    n_one = 0
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fout:
        for idx, rec in enumerate(records):
            gold, gold_name, file_name, text = row_fields(rec)
            got = table.get(_key(text)) if text and str(text).strip() else None
            if got is None:
                pred, pred_name = None, None
            else:
                pred, pred_name = got
                n_one += 1
                y_true.append(gold)
                y_pred.append(pred)
            fout.write(json.dumps({
                "idx": idx, "file_name": file_name, "label": gold, "label_name": gold_name,
                "pred": pred, "pred_name": pred_name,
            }) + "\n")
    n = len(records)
    metrics = task_a_metrics(y_true, y_pred) if y_true else {"n": 0}
    summary = {
        "task": "utt", "model": system,
        "rule": "The whole output is exactly one emotion label. A repeated or glued label is unparsed.",
        "source": str(src), "n": n, "n_one_label": n_one,
        "parse_failure_rate": round((n - n_one) / n, 4) if n else 0.0,
        "metrics_on_one_label": {k: metrics.get(k) for k in ("n", "WA", "UA", "macro_F1")},
    }
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{system}: {n_one}/{n} one-label")


def main() -> None:
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else RUNS_DIR / "test1" / "utterance_one_label")
    for system, src in SOURCES:
        if src.is_file():
            export_one(dest, system, src)
        else:
            print(f"missing {src}", file=sys.stderr)


if __name__ == "__main__":
    main()
