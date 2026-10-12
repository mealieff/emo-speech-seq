#!/usr/bin/env python3
"""Batch-size-1 vs batched spot-check. Prints ``ok`` (use batched decoding) or ``fallback``.

Sequence runs: identical text, or |delta macro-F1| <= tol. Writes ``<dir_bsN>/spotcheck.json``.
Utterance runs: identical first-N labels, or |delta UA| <= tol. Writes ``<res_bsN>_spotcheck.json``.

    python scripts/evaluate/spotcheck.py seq <dir_bs1> <dir_bsN> 0.01
    python scripts/evaluate/spotcheck.py utt <pred_bs1> <pred_bsN> <res_bs1> <res_bsN> 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _seq_preds(d: str) -> dict[str, str]:
    lines = (Path(d) / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(l)["seq_id"]: json.loads(l)["pred_text"] for l in lines if l.strip()}


def cmp_seq(d1: str, dn: str, tol: str) -> bool:
    a, b = _seq_preds(d1), _seq_preds(dn)
    n_diff = sum(a[k] != b.get(k) for k in a)
    m1 = json.loads((Path(d1) / "metrics.json").read_text())
    mn = json.loads((Path(dn) / "metrics.json").read_text())
    delta = {k: round(float(mn[k]) - float(m1[k]), 4) for k in ("macro_f1", "micro_f1", "mean_tiou", "parse_rate")}
    ok = n_diff == 0 or abs(delta["macro_f1"]) <= float(tol)
    report = {"n": len(a), "n_text_differs": n_diff, "delta_bsN_minus_bs1": delta,
              "macro_f1_bs1": m1["macro_f1"], "macro_f1_bsN": mn["macro_f1"], "tol": float(tol),
              "decision": "batched" if ok else "batch_size_1"}
    Path(dn, "spotcheck.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), file=sys.stderr)
    return ok


def cmp_utt(p1: str, pn: str, r1: str, rn: str, tol: str) -> bool:
    a = json.loads(Path(p1).read_text())["records"]
    b = json.loads(Path(pn).read_text())["records"]
    n_text = sum(x["raw_text"] != y["raw_text"] for x, y in zip(a, b))
    n_label = sum(x["pred_for_metrics"] != y["pred_for_metrics"] for x, y in zip(a, b))
    m1 = json.loads(Path(r1).read_text())["metrics"]
    mn = json.loads(Path(rn).read_text())["metrics"]
    delta = {k: round(float(mn[k]) - float(m1[k]), 4) for k in ("WA", "UA", "macro_F1", "strict_parse_failure_rate")}
    ok = n_label == 0 or abs(delta["UA"]) <= float(tol)
    report = {"n": len(a), "n_text_differs": n_text, "n_first_n_label_differs": n_label,
              "delta_bsN_minus_bs1": delta, "tol": float(tol),
              "decision": "batched" if ok else "batch_size_1"}
    Path(rn).with_name(Path(rn).stem + "_spotcheck.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), file=sys.stderr)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind", required=True)
    s = sub.add_parser("seq")
    s.add_argument("dir_bs1")
    s.add_argument("dir_bsN")
    s.add_argument("tol")
    u = sub.add_parser("utt")
    for name in ("pred_bs1", "pred_bsN", "res_bs1", "res_bsN", "tol"):
        u.add_argument(name)
    a = p.parse_args()
    if a.kind == "seq":
        ok = cmp_seq(a.dir_bs1, a.dir_bsN, a.tol)
    else:
        ok = cmp_utt(a.pred_bs1, a.pred_bsN, a.res_bs1, a.res_bsN, a.tol)
    print("ok" if ok else "fallback")


if __name__ == "__main__":
    main()
