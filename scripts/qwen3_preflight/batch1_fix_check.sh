#!/usr/bin/env bash
# Does the qwen3_omni_fix change Qwen3-Omni outputs at batch 1, and is batching safe with it?
# 1. Re-decode the Test1 spot-check sequences (seed 7, 30) at batch 1 with the fixed code, pretrained and
#    SFT, same flags as scripts/inference/run_test1.sh; compare raw text with the old code's batch-1
#    spot-check outputs, split at 8 s.
# 2. Per-example loss, batch 1 vs batch 8 (utterances) with and without the fix; batch-1 sequence
#    losses fixed vs unfixed.
# Decision rule: any batch-1 output over 8 s differs -> rerun everything for Qwen3-Omni.
# Output: $EMOSEQ_RUNS/qwen3_batch1_fix_check/summary.txt
set -uo pipefail
. /media/volume/AudioEmotionSequenceVol/env.sh
source /media/volume/AudioEmotionSequenceVol/emo-seq/scripts/common.sh
export HF_HUB_OFFLINE=1

P="${RUNS_DIR}/qwen3_batch1_fix_check"
OLD="${RUNS_DIR}/test1/spotcheck"
STATUS="${P}/status.log"
mkdir -p "${P}"
ADAPTER=$("${MAIN_PY}" -c "import json,sys;print(json.load(open(sys.argv[1]))['best_checkpoint'])" \
  "${RUNS_DIR}/sft/sequence/qwen3_omni/selection_full_dev.json")

stage pt_seq "fixed, batch 1"
"${MAIN_PY}" scripts/inference/sequence.py --model qwen3_omni --jsonl "${OLD}/seq.jsonl" --max-new-tokens 800 \
  --batch-size 1 --no-resume --tag spot --out-dir "${P}/pt_seq" > "${P}/pt_seq.log" 2>&1
stage sft_seq "fixed, batch 1, ${ADAPTER}"
"${MAIN_PY}" scripts/inference/sequence.py --model qwen3_omni --adapter "${ADAPTER}" --jsonl "${OLD}/seq.jsonl" \
  --max-new-tokens 800 --batch-size 1 --no-resume --tag spot --out-dir "${P}/sft_seq" > "${P}/sft_seq.log" 2>&1

"${MAIN_PY}" - "${OLD}" "${P}" <<'EOF' | tee "${P}/summary.txt"
import json, sys
from pathlib import Path
old, new = Path(sys.argv[1]), Path(sys.argv[2])

def seq(path):
    return {r["seq_id"]: r for r in map(json.loads, open(path)) if r}

for name, sub in (("pt_seq", "qwen3_omni_pt"), ("sft_seq", "qwen3_omni_sft")):
    a, b = seq(old / sub / "seq_bs1" / "predictions.jsonl"), seq(new / name / "predictions.jsonl")
    pairs = [(a[k]["audio_duration"], a[k]["pred_text"], b[k]["pred_text"]) for k in a if k in b]
    for label, group in (("<=8 s", [p for p in pairs if p[0] <= 8.0]), (">8 s", [p for p in pairs if p[0] > 8.0])):
        print(f"{name} {label}: identical raw text {sum(x == y for _, x, y in group)}/{len(group)}")
    ma, mb = json.load(open(old / sub / "seq_bs1" / "metrics.json")), json.load(open(new / name / "metrics.json"))
    print(f"  {name} macro_f1 old {ma['macro_f1']:.4f} fixed {mb['macro_f1']:.4f}; "
          f"micro_f1 old {ma['micro_f1']:.4f} fixed {mb['micro_f1']:.4f}")
EOF

stage batch_check "per-example loss"
"${MAIN_PY}" scripts/evaluate/qwen3_omni_batch_check.py "${UTT_DIR}/msp_podcast_train.jsonl" 32 \
  "${SEQ_DIR}/train.jsonl" 8 "${P}/batch_check.json" > "${P}/batch_check.log" 2>&1
grep -E '^\[|^PASS|^batch-1|^  (utt|seq)' "${P}/batch_check.log" | tee -a "${P}/summary.txt"
stage done "summary: ${P}/summary.txt"
