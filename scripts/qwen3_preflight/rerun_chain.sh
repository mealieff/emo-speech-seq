#!/usr/bin/env bash
# Qwen3-Omni reruns after the speed check (FA2 + fixed audio masking):
# 1. wait for seq_speed.sh; 2. noise-floor check of the batch-1 vs batch-8 loss gap (noise_floor.py);
# 3. sequence SFT, 3 epochs, with the fastest timed setting that succeeded (batch 1 x 8 either way),
#    full-Dev selection; 4. Test1: SFT sequence first, then pretrained sequence + utterance.
set -uo pipefail
. /media/volume/AudioEmotionSequenceVol/env.sh
HERE="$(cd "$(dirname "$0")" && pwd)"
source /media/volume/AudioEmotionSequenceVol/emo-seq/scripts/common.sh
export HF_HUB_OFFLINE=1
P="${RUNS_DIR}/qwen3_seq_speed"
STATUS="${LOG_DIR}/qwen3_rerun.status"

if [[ -n "${SFT_EXTRA:-}" ]]; then EXTRA="${SFT_EXTRA}"; else
while pgrep -f "qwen3_preflight/seq_speed.sh" > /dev/null; do sleep 30; done
stage noise_floor "start"
"${MAIN_PY}" "${HERE}/noise_floor.py" "${UTT_DIR}/msp_podcast_train.jsonl" "${P}/noise_floor.json" > "${P}/noise_floor.log" 2>&1
grep -E 'logits' "${P}/noise_floor.log" | sed 's/^/noise floor: /' | tee -a "${P}/summary.txt"

EXTRA=$("${MAIN_PY}" - "${P}/summary.txt" <<'EOF'
import re, sys
best = None
for line in open(sys.argv[1]):
    m = re.match(r"(t[12]_\S+): ([\d.]+) s/step", line)
    if m and (best is None or float(m.group(2)) < best[1]):
        best = (m.group(1), float(m.group(2)))
extra = "--dataloader-workers 4" if best else ""
if best and best[0].startswith("t2_ckpt_above_"):
    extra = f"--checkpoint-above-sec {best[0].rsplit('_', 1)[1]} " + extra
print(extra)
EOF
)
fi
stage sft "SFT_EXTRA=${EXTRA}"
echo "sequence SFT launched with SFT_EXTRA='${EXTRA}' at $(date -Is)" | tee -a "${P}/summary.txt"
QWEN3_SET=full SFT_EXTRA="${EXTRA}" bash scripts/finetune/run_sequence_sft.sh qwen3_omni
stage sft "exit=$?"
QWEN3_SEQ_BS=4 bash scripts/inference/run_test1.sh qwen3_omni_sft qwen3_omni_pt
stage test1 "exit=$?"
