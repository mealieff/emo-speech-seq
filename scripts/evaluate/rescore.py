#!/usr/bin/env python3
"""Re-score saved sequence predictions from their raw model text.

For each run directory, re-parses every ``pred_text`` in ``predictions.jsonl`` with the
current parser (timestamps in seconds or m:ss / h:mm:ss) and with the seconds-only grammar
(strict-format parse rate), and re-computes every metric. Writes
``<dir>/rescored/{predictions.jsonl, quartile_report.json, metrics.json, diff.json}``;
the original files are left untouched.

    python scripts/evaluate/rescore.py runs/test1/sequence/qwen2_audio_pt [more dirs ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from emoseq.bins import BIN_NAMES, bin_of_record
from emoseq.parsing.sequence import parse_saved_output
from emoseq.provenance import git_commit
from emoseq.scoring.report import pool_metrics, quartile_report
from emoseq.scoring.sequence import pack_scores, score_sequence

_DIFF_KEYS = ("accuracy", "macro_f1", "mean_tiou", "parse_rate")


def rescore(*dirs: str) -> None:
    """Re-parse every saved pred_text with the current parser (timestamps in seconds or
    m:ss / h:mm:ss) and with the seconds-only grammar (strict-format parse rate).
    Writes <dir>/rescored/{predictions.jsonl, quartile_report.json, metrics.json, diff.json};
    the original files are left untouched."""
    commit = git_commit()
    for d in map(Path, dirs):
        rows = [json.loads(l) for l in (d / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        scored = [r for r in rows if not r.get("over_native_context_limit")]
        old_report = quartile_report(scored)
        new_rows, changed = [], []
        n_strict_vs_old = 0
        for r in rows:
            r = dict(r)
            if r.get("over_native_context_limit"):
                new_rows.append(r)
                continue
            gold = [{"start": float(s["start"]), "end": float(s["end"]), "emotion": s["emotion"]}
                    for s in r["ground_truth"]]
            pred = parse_saved_output(r.get("pred_text") or "")
            strict = 1.0 if parse_saved_output(r.get("pred_text") or "", clock=False) else 0.0
            old_parsed = 1.0 if r.get("pred") else 0.0
            n_strict_vs_old += strict != old_parsed
            if pred != (r.get("pred") or []):
                changed.append({"seq_id": r["seq_id"], "length_bin": r.get("length_bin"),
                                "old_pred": r.get("pred"), "new_pred": pred,
                                "pred_text": (r.get("pred_text") or "")[:400]})
            scores = score_sequence(gold, pred)
            r.update({
                "pred": pred, "parsed": scores["parsed"], "parsed_strict": strict,
                "macro_f1": scores["macro_f1"], "micro_f1": scores["micro_f1"], "mean_tiou": scores["mean_tiou"],
                "_scores": pack_scores(scores),
            })
            new_rows.append(r)
        new_scored = [r for r in new_rows if not r.get("over_native_context_limit")]
        report = quartile_report(new_scored)
        strict_by_bin: dict[str, list[float]] = {b: [] for b in BIN_NAMES}
        for r in new_scored:
            strict_by_bin.setdefault(r.get("length_bin") or bin_of_record(r), []).append(r["parsed_strict"])
        for b, v in strict_by_bin.items():
            report["by_quartile"].setdefault(b, {})["parse_rate_strict"] = sum(v) / len(v) if v else 0.0
        all_strict = [r["parsed_strict"] for r in new_scored]
        report["additional"]["overall"]["parse_rate_strict"] = sum(all_strict) / len(all_strict) if all_strict else 0.0
        report["parser"] = {"git_commit": commit, "timestamps": "seconds | m:ss[.ff] | h:mm:ss[.ff]",
                            "parse_rate_strict": "seconds-only timestamps"}
        summary = pool_metrics([r["_scores"] for r in new_scored])
        summary.update({"n_records": len(new_scored), "n_records_written": len(new_rows),
                        "parse_rate_strict": report["additional"]["overall"]["parse_rate_strict"],
                        "parser_commit": commit, "source": str(d / "predictions.jsonl")})

        def cells(rep: dict) -> dict:
            out = {b: {k: rep["by_quartile"][b].get(k) for k in _DIFF_KEYS} for b in BIN_NAMES}
            out["ALL"] = {k: rep["additional"]["overall"].get(k) for k in _DIFF_KEYS}
            return out

        old_c, new_c = cells(old_report), cells(report)
        delta = {b: {k: round(float(new_c[b][k] or 0) - float(old_c[b][k] or 0), 6) for k in _DIFF_KEYS}
                 for b in new_c}
        diff = {"n_scored": len(new_scored), "n_pred_changed": len(changed),
                "n_strict_parsed_differs_from_old_parsed": n_strict_vs_old,
                "old": old_c, "new": new_c, "delta_new_minus_old": delta, "changed_rows": changed[:200]}
        out = d / "rescored"
        out.mkdir(exist_ok=True)
        (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in new_rows), encoding="utf-8")
        (out / "quartile_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "diff.json").write_text(json.dumps(diff, indent=2) + "\n", encoding="utf-8")
        max_d = max(abs(v) for row in delta.values() for v in row.values())
        print(f"{d}: n={len(new_scored)} pred_changed={len(changed)} strict!=old={n_strict_vs_old} "
              f"max|delta|={max_d:.4f} strict_rate={summary['parse_rate_strict']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    rescore(*p.parse_args().dirs)


if __name__ == "__main__":
    main()
