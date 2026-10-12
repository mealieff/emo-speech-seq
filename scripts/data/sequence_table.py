#!/usr/bin/env python3
"""Print the LaTeX sequence-statistics table from the sequence manifests.

    python scripts/data/sequence_table.py
"""
from __future__ import annotations

import json
from collections import Counter

from emoseq.bins import length_bin
from emoseq.config import manifest_jsonl

SPLITS = (("Train", "train"), ("Development", "dev"), ("Test1", "test1"))
BINS = ("2", "3", "4-6", "7+")


def main() -> None:
    rows, total, total_max = [], Counter(), 0
    for name, key in SPLITS:
        path = manifest_jsonl(key)
        lens = [int(json.loads(line)["seq_len"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        c = Counter(length_bin(n) for n in lens)
        c["all"] = len(lens)
        total.update(c)
        total_max = max(total_max, max(lens) if lens else 0)
        rows.append((name, c, max(lens) if lens else 0))

    def fmt(n: int) -> str:
        return f"{n:,}"

    print(r"""\begin{table}[t]
\centering
\caption{Strict consecutive unique-ID sequence statistics for MSP-Podcast 1.12
(Train, Development, Test1). $2 \le L \le 40$. Test3 is omitted.}
\label{tab:seq-corpus}
\small
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Split} &
\textbf{$L=2$} &
\textbf{$L=3$} &
\textbf{$L=4$--$6$} &
\textbf{$L\geq7$} &
\textbf{Total} &
\textbf{Max $L$} \\
\midrule""")
    for name, c, mx in rows:
        print(f"{name:<12} & " + " & ".join(fmt(c[b]) for b in BINS) + f" & {fmt(c['all'])} & {mx} \\\\")
    print(r"\midrule")
    print(r"\textbf{Total} & " + " & ".join(rf"\textbf{{{fmt(total[b])}}}" for b in BINS)
          + rf" & \textbf{{{fmt(total['all'])}}} & \textbf{{{total_max}}} \\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
