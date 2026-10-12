#!/usr/bin/env bash
# Rerun qwen3_omni_sft sequence Test1 at batch 1 once rerun_chain.sh is done (the batched run passed
# the macro-F1 spot-check but not the 1e-3 per-example check). The batch-4 output is deleted first.
set -uo pipefail
. /media/volume/AudioEmotionSequenceVol/env.sh
source /media/volume/AudioEmotionSequenceVol/emo-seq/scripts/common.sh
export HF_HUB_OFFLINE=1
STATUS="${LOG_DIR}/qwen3_rerun.status"

while pgrep -f "qwen3_preflight/rerun_chain.sh" > /dev/null; do sleep 60; done
stage sft_seq_bs1 "deleting batch-4 output ${RUNS_DIR}/test1/sequence/qwen3_omni_sft"
rm -rf "${RUNS_DIR}/test1/sequence/qwen3_omni_sft"
QWEN3_SEQ_BS=1 bash scripts/inference/run_test1.sh qwen3_omni_sft
stage sft_seq_bs1 "exit=$?"
