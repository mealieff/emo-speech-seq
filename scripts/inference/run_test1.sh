#!/usr/bin/env bash
# Test1 inference + scoring for every system, run one after another on a single GPU.
#
#   bash scripts/inference/run_test1.sh all
#   bash scripts/inference/run_test1.sh qwen2_audio_pt audio_flamingo3_sft wavlm_timeaug
#
# Systems (sequence task, utterance task, or both):
#   qwen2_audio_pt  audio_flamingo3_pt   pretrained, both tasks
#   qwen2_audio_sft audio_flamingo3_sft  utterance adapter (runs/sft/utterance/<model>) for the utterance
#                                        task; full-Dev-selected sequence checkpoint for the sequence task
#   qwen3_omni_pt qwen3_omni_sft         as above; sequence batch QWEN3_SEQ_BS (default 4, one 80 GB GPU)
#   salmonn_pt kimi_audio_pt             pretrained, both tasks (legacy environment)
#   salmonn_sft kimi_audio_sft           selected sequence checkpoint, plus the last utterance checkpoint
#                                        when runs/sft/utterance/<model>/final_checkpoint.json exists
#   wavlm_is2025                         utterance task (IS2025 seed-7 classifier)
#   wavlm_timeaug                        sequence task
#   references                           content-free reference predictions (no GPU)
#
# Batching rule: a run decodes batched only if batch 1 and batch N agree on a seeded random
# Test1 sample (sequence: identical text or |delta macro-F1| <= 0.01 on 30 sequences, plus 10
# sequences > 30 s for SALMONN; utterance: identical labels or |delta UA| <= 0.01 on 30 clips).
# Otherwise it decodes with batch 1. Pretrained Qwen2-Audio sequence decoding and Kimi-Audio are
# always batch 1. Every utterance run is also scored on the Test1 sequences with gold
# boundaries (runs/test1/utt_on_sequences/<system>).
#
# Output: $EMOSEQ_RUNS/test1/{sequence,utterance,utt_on_sequences,references,spotcheck}/<system>/
set -uo pipefail
source "$(dirname "$0")/../common.sh"

SEQ_MAX_NEW="${SEQ_MAX_NEW:-800}"
SEQ_BS="${SEQ_BS:-8}"
QWEN3_SEQ_BS="${QWEN3_SEQ_BS:-4}"
UTT_BS="${UTT_BS:-16}"
SPOT_N="${SPOT_N:-30}"
SPOT_TOL="${SPOT_TOL:-0.01}"
SEQ_JSONL="${SEQ_DIR}/test1.jsonl"
UTT_JSONL="${UTT_DIR}/msp_podcast_test1.jsonl"
T1="${RUNS_DIR}/test1"
SPOT="${T1}/spotcheck"
STATUS="${LOG_DIR}/test1.status"
mkdir -p "${SPOT}"

ALL=(references wavlm_is2025 wavlm_timeaug qwen2_audio_pt audio_flamingo3_pt qwen2_audio_sft audio_flamingo3_sft
     qwen3_omni_pt qwen3_omni_sft salmonn_pt kimi_audio_pt salmonn_sft kimi_audio_sft)
SYSTEMS=("$@")
[[ ${#SYSTEMS[@]} -eq 0 ]] && { sed -n '2,24p' "$0"; exit 1; }
[[ "${SYSTEMS[0]}" == all ]] && SYSTEMS=("${ALL[@]}")

for f in "${SEQ_JSONL}" "${UTT_JSONL}"; do
  [[ -s "${f}" ]] || { echo "missing ${f}; run bash scripts/data/prepare_all.sh" >&2; exit 1; }
done

# ---------------------------------------------------------------- spot-check samples (seed 7)
"${MAIN_PY}" scripts/data/sample_jsonl.py "${SEQ_JSONL}" "${SPOT_N}" 7 "${SPOT}/seq.jsonl"
"${MAIN_PY}" scripts/data/sample_jsonl.py "${UTT_JSONL}" "${SPOT_N}" 7 "${SPOT}/utt.jsonl"
"${MAIN_PY}" - "${SEQ_JSONL}" "${SPOT}/seq.jsonl" "${SPOT}/seq_long.jsonl" <<'EOF'
import json, random, sys
allr = [l for l in open(sys.argv[1]) if l.strip()]
base = [l for l in open(sys.argv[2]) if l.strip()]
ids = {json.loads(l)["seq_id"] for l in base}
long = [l for l in allr if json.loads(l)["audio_duration"] > 30 and json.loads(l)["seq_id"] not in ids]
open(sys.argv[3], "w").write("".join(base + random.Random(7).sample(long, min(10, len(long)))))
EOF

best_seq_ckpt() {  # model -> selected sequence checkpoint
  local sel="${RUNS_DIR}/sft/sequence/$1/selection_full_dev.json"
  [[ -f "${sel}" ]] || { echo "no ${sel}; run bash scripts/finetune/run_sequence_sft.sh $1" >&2; return 1; }
  "${MAIN_PY}" -c "import json,sys;print(json.load(open(sys.argv[1]))['best_checkpoint'])" "${sel}"
}

decide() {  # verdict-from-spotcheck default_bs -> batch size
  [[ "$1" == ok ]] && echo "$2" || echo 1
}

# ---------------------------------------------------------------- Hugging Face ALMs (main env)
hf_seq() {  # system model [adapter]
  local sys=$1 model=$2 adapter=${3:-} out="${T1}/sequence/$1" bs=1 s="${SPOT}/$1" nbs="${SEQ_BS}"
  local extra=(); [[ -n "${adapter}" ]] && extra=(--adapter "${adapter}")
  [[ "${model}" == qwen3_omni ]] && nbs="${QWEN3_SEQ_BS}"
  if [[ -n "${adapter}" || "${model}" == audio_flamingo3 || "${model}" == qwen3_omni ]]; then
    for b in 1 "${nbs}"; do
      "${MAIN_PY}" scripts/inference/sequence.py --model "${model}" "${extra[@]}" --jsonl "${SPOT}/seq.jsonl" \
        --max-new-tokens "${SEQ_MAX_NEW}" --batch-size "${b}" --no-resume --tag spot \
        --out-dir "${s}/seq_bs${b}" > "${s}_seq_bs${b}.log" 2>&1
    done
    bs=$(decide "$("${MAIN_PY}" scripts/evaluate/spotcheck.py seq "${s}/seq_bs1" "${s}/seq_bs${nbs}" "${SPOT_TOL}" 2>>"${STATUS}")" "${nbs}")
  fi
  stage "${sys}_seq" "start batch=${bs} adapter=${adapter:-none}"
  provenance "${out}/provenance.json" model="${model}" adapter="${adapter:-none}" batch_size="${bs}" \
    max_new_tokens="${SEQ_MAX_NEW}" jsonl="${SEQ_JSONL}" decoding=greedy
  "${MAIN_PY}" scripts/inference/sequence.py --model "${model}" "${extra[@]}" --jsonl "${SEQ_JSONL}" \
    --max-new-tokens "${SEQ_MAX_NEW}" --batch-size "${bs}" --no-resume --out-dir "${out}"
  stage "${sys}_seq" "exit=$?"
}

hf_utt() {  # system model [model_dir]
  local sys=$1 model=$2 mdir=${3:-} out="${T1}/utterance/$1" s="${SPOT}/$1"
  local extra=(--base-only); [[ -n "${mdir}" ]] && extra=(--model-dir "${mdir}")
  for b in 1 "${UTT_BS}"; do
    "${MAIN_PY}" scripts/inference/utterance.py --model "${model}" "${extra[@]}" --eval-jsonl "${SPOT}/utt.jsonl" \
      --batch-size "${b}" --out-dir "${s}/utt_bs${b}" > "${s}_utt_bs${b}.log" 2>&1
  done
  local bs
  bs=$(decide "$("${MAIN_PY}" scripts/evaluate/spotcheck.py utt "${s}/utt_bs1/predictions_utt.json" \
      "${s}/utt_bs${UTT_BS}/predictions_utt.json" "${s}/utt_bs1/eval_utt.json" "${s}/utt_bs${UTT_BS}/eval_utt.json" \
      "${SPOT_TOL}" 2>>"${STATUS}")" "${UTT_BS}")
  stage "${sys}_utt" "start batch=${bs} model_dir=${mdir:-base}"
  provenance "${out}/provenance.json" model="${model}" model_dir="${mdir:-none}" batch_size="${bs}" \
    parse_mode=first_n eval_jsonl="${UTT_JSONL}"
  "${MAIN_PY}" scripts/inference/utterance.py --model "${model}" "${extra[@]}" --eval-jsonl "${UTT_JSONL}" \
    --batch-size "${bs}" --out-dir "${out}"
  stage "${sys}_utt" "exit=$?"
  utt_on_sequences "${sys}" "${out}/predictions_test1.json"
}

# ---------------------------------------------------------------- SALMONN / Kimi-Audio (legacy env)
bespoke_seq() {  # system model [sft_dir]
  require_legacy
  local sys=$1 model=$2 sft=${3:-} out="${T1}/sequence/$1" bs=1 s="${SPOT}/$1"
  local extra=(); [[ -n "${sft}" ]] && extra=(--sft-dir "${sft}")
  if [[ "${model}" == salmonn ]]; then
    for b in 1 "${SEQ_BS}"; do
      "${LEGACY_PY}" scripts/inference/bespoke.py --model salmonn --task seq "${extra[@]}" --jsonl "${SPOT}/seq_long.jsonl" \
        --max-new-tokens "${SEQ_MAX_NEW}" --batch-size "${b}" --no-resume --out "${s}/seq_bs${b}" > "${s}_seq_bs${b}.log" 2>&1
    done
    bs=$(decide "$("${MAIN_PY}" scripts/evaluate/spotcheck.py seq "${s}/seq_bs1" "${s}/seq_bs${SEQ_BS}" "${SPOT_TOL}" 2>>"${STATUS}")" "${SEQ_BS}")
  fi
  stage "${sys}_seq" "start batch=${bs} sft=${sft:-none}"
  provenance "${out}/provenance.json" model="${model}" adapter="${sft:-none}" batch_size="${bs}" \
    max_new_tokens="${SEQ_MAX_NEW}" jsonl="${SEQ_JSONL}" decoding=greedy
  TQDM_DISABLE=1 "${LEGACY_PY}" scripts/inference/bespoke.py --model "${model}" --task seq "${extra[@]}" \
    --jsonl "${SEQ_JSONL}" --max-new-tokens "${SEQ_MAX_NEW}" --batch-size "${bs}" --no-resume --out "${out}"
  stage "${sys}_seq" "exit=$?"
}

bespoke_utt() {  # system model [sft_dir]
  require_legacy
  local sys=$1 model=$2 sft=${3:-} out="${T1}/utterance/$1" bs=1 s="${SPOT}/$1"
  local extra=(); [[ -n "${sft}" ]] && extra=(--sft-dir "${sft}")
  if [[ "${model}" == salmonn ]]; then
    for b in 1 "${UTT_BS}"; do
      "${LEGACY_PY}" scripts/inference/bespoke.py --model salmonn --task utt "${extra[@]}" --jsonl "${SPOT}/utt.jsonl" \
        --batch-size "${b}" --no-resume --out "${s}/utt_bs${b}" > "${s}_utt_bs${b}.log" 2>&1
    done
    bs=$(decide "$("${MAIN_PY}" scripts/evaluate/spotcheck.py utt "${s}/utt_bs1/predictions_utt.json" \
        "${s}/utt_bs${UTT_BS}/predictions_utt.json" "${s}/utt_bs1/eval_utt.json" "${s}/utt_bs${UTT_BS}/eval_utt.json" \
        "${SPOT_TOL}" 2>>"${STATUS}")" "${UTT_BS}")
  fi
  stage "${sys}_utt" "start batch=${bs} sft=${sft:-none}"
  provenance "${out}/provenance.json" model="${model}" adapter="${sft:-none}" batch_size="${bs}" parse_mode=first_n \
    eval_jsonl="${UTT_JSONL}" decoding=greedy
  TQDM_DISABLE=1 "${LEGACY_PY}" scripts/inference/bespoke.py --model "${model}" --task utt "${extra[@]}" \
    --jsonl "${UTT_JSONL}" --batch-size "${bs}" --no-resume --out "${out}"
  stage "${sys}_utt" "exit=$?"
  utt_on_sequences "${sys}" "${out}/predictions_test1.json"
}

# ---------------------------------------------------------------- scoring helpers
utt_on_sequences() {  # system predictions_file
  local o="${T1}/utt_on_sequences/$1"
  [[ -s "$2" ]] || { stage utt_on_sequences "$1: missing $2"; return; }
  "${MAIN_PY}" scripts/evaluate/utterance_on_sequences.py --preds "$2" --jsonl "${SEQ_JSONL}" --out-dir "${o}" --model "$1" > /dev/null
  provenance "${o}/provenance.json" preds="$2" jsonl="${SEQ_JSONL}" label=pred_for_metrics boundaries=gold
  stage utt_on_sequences "$1 exit=$?"
}

run_system() {
  case "$1" in
    references)
      "${MAIN_PY}" scripts/evaluate/references.py --jsonl "${SEQ_JSONL}" --out "${T1}/references" ;;
    wavlm_is2025)
      local o="${T1}/utterance/wavlm_is2025"
      provenance "${o}/provenance.json" model=wavlm_is2025_cat_ser_seed7 split=test1
      "${MAIN_PY}" scripts/inference/wavlm_utterance.py --split test1 --out-dir "${o}"
      utt_on_sequences wavlm_is2025 "${o}/cat_test1_preds.csv" ;;
    wavlm_timeaug)
      local o="${T1}/sequence/wavlm_timeaug"
      provenance "${o}/provenance.json" ckpt="${WAVLM_TIMEAUG_CKPT}" split=test1
      "${MAIN_PY}" scripts/inference/wavlm_timeaug.py --split test1 --out-dir "${o}" ;;
    qwen2_audio_pt|audio_flamingo3_pt|qwen3_omni_pt)
      local m=${1%_pt}; hf_seq "$1" "${m}"; hf_utt "$1" "${m}" ;;
    qwen2_audio_sft|audio_flamingo3_sft|qwen3_omni_sft)
      local m=${1%_sft} ck
      ck=$(best_seq_ckpt "${m}") && hf_seq "$1" "${m}" "${ck}"
      if [[ -f "${RUNS_DIR}/sft/utterance/${m}/adapter_config.json" ]]; then
        hf_utt "$1" "${m}" "${RUNS_DIR}/sft/utterance/${m}"
      else
        stage "$1_utt" "skipped: no utterance adapter at ${RUNS_DIR}/sft/utterance/${m}"
      fi ;;
    salmonn_pt|kimi_audio_pt)
      local m=${1%_pt}; bespoke_seq "$1" "${m}"; bespoke_utt "$1" "${m}" ;;
    salmonn_sft|kimi_audio_sft)
      local m=${1%_sft} ck final
      ck=$(best_seq_ckpt "${m}") && bespoke_seq "$1" "${m}" "${ck}"
      final="${RUNS_DIR}/sft/utterance/${m}/final_checkpoint.json"
      if [[ -s ${final} ]]; then
        ck=$("${MAIN_PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['final_checkpoint'])" "${final}")
        bespoke_utt "$1" "${m}" "${ck}"
      else
        stage "$1_utt" "skipped: no ${final}"
      fi ;;
    *) echo "unknown system $1" >&2; return 1 ;;
  esac
}

stage start "systems=${SYSTEMS[*]} commit=$(git rev-parse --short HEAD 2>/dev/null || echo none)"
for sys in "${SYSTEMS[@]}"; do
  mkdir -p "${SPOT}/${sys}"
  run_system "${sys}"
done
stage done "results under ${T1}"
