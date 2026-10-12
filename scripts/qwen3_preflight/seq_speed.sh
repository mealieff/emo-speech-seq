#!/usr/bin/env bash
# Qwen3-Omni sequence SFT on FlashAttention-2: checks, then 50-step timings.
# 1. per-example loss batch 1 vs 8 (scripts/evaluate/qwen3_omni_batch_check.py)
# 2. gradient equivalence batch 1 x 8 vs batch 8 x 1 (scripts/evaluate/qwen3_omni_grad_check.py)
# 3. memory without checkpointing -> CKPT_SEC (largest padded audio seconds that fit with 4 GiB spare)
# 4. 50 optimizer steps each: T1 checkpointing every step; T2 only above CKPT_SEC;
#    T3 = T2 + batch 8 x 1 grouped by length (only if 2 passed). GPU utilization sampled each second.
# Output: $EMOSEQ_RUNS/qwen3_seq_speed/summary.txt
set -uo pipefail
. /media/volume/AudioEmotionSequenceVol/env.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
source /media/volume/AudioEmotionSequenceVol/emo-seq/scripts/common.sh
export HF_HUB_OFFLINE=1

P="${RUNS_DIR}/qwen3_seq_speed"
mkdir -p "${P}"
STATUS="${P}/status.log"
SUMMARY="${P}/summary.txt"
: > "${SUMMARY}"
TRAIN="${SEQ_DIR}/train.jsonl"
DEV="${SEQ_DIR}/dev.jsonl"
say() { echo "$*" | tee -a "${SUMMARY}"; }

stage batch_check "per-example loss batch 1 vs 8, FA2"
"${MAIN_PY}" scripts/evaluate/qwen3_omni_batch_check.py "${UTT_DIR}/msp_podcast_train.jsonl" 32 "${TRAIN}" 8 \
  "${P}/batch_check.json" > "${P}/batch_check.log" 2>&1
grep -E '^\[|^PASS' "${P}/batch_check.log" | sed 's/^/batch check: /' | tee -a "${SUMMARY}"

stage grad_check "batch 1 x 8 vs batch 8 x 1"
"${MAIN_PY}" scripts/evaluate/qwen3_omni_grad_check.py "${TRAIN}" "${P}/grad_check.json" > "${P}/grad_check.log" 2>&1
grep -E '^(durations|step loss|grad norm|relative|PASS)' "${P}/grad_check.log" | sed 's/^/grad check: /' | tee -a "${SUMMARY}"
GRAD_OK=0; grep -q '^PASS=True' "${P}/grad_check.log" && GRAD_OK=1

stage mem "no checkpointing"
"${MAIN_PY}" "${HERE}/mem_probe.py" "${TRAIN}" "${P}/mem.json" > "${P}/mem.log" 2>&1
grep -E '^(weights|batch)' "${P}/mem.log" | sed 's/^/memory: /' | tee -a "${SUMMARY}"
CKPT_SEC=$("${MAIN_PY}" - "${P}/mem.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
limit = d["total_gib"] - 4
fits = [x["sec"] for x in d["batch1"] if x["peak_gib"] is not None and x["peak_gib"] <= limit]
fits += [x["padded_sec"] for x in d["batch8"] if x["peak_gib"] is not None and x["peak_gib"] <= limit]
over = [x["sec"] for x in d["batch1"] if x["peak_gib"] is None or x["peak_gib"] > limit]
over += [x["padded_sec"] for x in d["batch8"] if x["peak_gib"] is None or x["peak_gib"] > limit]
ok = [s for s in fits if all(o > s for o in over)]
print(f"{max(ok):.0f}" if ok else 0)
EOF
)
say "checkpointing threshold: ${CKPT_SEC} s padded audio"

timed() {  # tag args...
  local tag=$1; shift
  stage "${tag}" "$*"
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 1000 > "${P}/${tag}.gpu" &
  local smi=$!
  "${MAIN_PY}" "${HERE}/peak.py" scripts/finetune/sequence_lora.py --model qwen3_omni --train-jsonl "${TRAIN}" \
    --eval-jsonl "${DEV}" --no-eval --length-bin-weight --max-steps 50 --logging-steps 10 --dataloader-workers 4 \
    --save-total-limit 1 --output-dir "${P}/${tag}" "$@" > "${P}/${tag}.log" 2>&1
  kill "${smi}" 2>/dev/null
  "${MAIN_PY}" - "${P}/${tag}.log" "${P}/${tag}.gpu" "${tag}" <<'EOF' | tee -a "${SUMMARY}"
import ast, re, sys
log, gpu, tag = open(sys.argv[1]).read(), [int(x) for x in open(sys.argv[2]).read().split() if x.isdigit()], sys.argv[3]
m = re.search(r"\{[^{}]*'train_runtime'[^{}]*\}", log)
peak = re.findall(r"^\[peak\].*$", log, re.M)
if not m:
    print(f"{tag}: FAILED ({'OOM' if 'out of memory' in log.lower() else 'see log'})")
    sys.exit()
rt = float(ast.literal_eval(m.group(0))["train_runtime"])
busy = gpu[len(gpu) // 10:]  # skip model loading
low = sum(u < 90 for u in busy) / max(len(busy), 1)
print(f"{tag}: {rt / 50:.2f} s/step -> 3 epochs (3561 steps) {rt / 50 * 3561 / 3600:.1f} h  "
      f"GPU util mean {sum(busy) / max(len(busy), 1):.0f}%, below 90% {low:.0%}  {peak[-1] if peak else ''}")
EOF
}

timed t1_ckpt_always
[[ "${CKPT_SEC}" != 0 ]] && timed t2_ckpt_above_${CKPT_SEC} --checkpoint-above-sec "${CKPT_SEC}"
if [[ "${GRAD_OK}" == 1 && "${CKPT_SEC}" != 0 ]]; then
  timed t3_bs8_ckpt_above_${CKPT_SEC} --checkpoint-above-sec "${CKPT_SEC}" --per-device-batch-size 8 --grad-accum 1 --group-by-length
else
  say "batch 8 not timed: gradient check passed=${GRAD_OK}, threshold=${CKPT_SEC}"
fi
stage done "summary: ${SUMMARY}"
