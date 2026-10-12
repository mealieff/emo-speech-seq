# Sourced by the shell entry points. Resolves the interpreters and the paths from
# emoseq.config (configs/paths.env + environment), and defines a small logger.
#
#   MAIN_PY    main environment (Qwen2-Audio, Audio-Flamingo-3, WavLM, scoring); default .venv
#   LEGACY_PY  legacy environment (SALMONN, Kimi-Audio);                       default .venv-legacy

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MAIN_PY="${MAIN_PY:-${REPO_ROOT}/.venv/bin/python}"
LEGACY_PY="${LEGACY_PY:-${REPO_ROOT}/.venv-legacy/bin/python}"
if [[ ! -x "${MAIN_PY}" ]]; then
  echo "main environment not found at ${MAIN_PY}; run bash scripts/setup/install.sh (or set MAIN_PY)" >&2
  exit 1
fi

eval "$("${MAIN_PY}" -m emoseq.config)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${MODEL_CACHE}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${MODEL_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

LOG_DIR="${RUNS_DIR}/logs"
mkdir -p "${LOG_DIR}"

# stage <tag> <message>: timestamped line to stdout and ${STATUS} (if set).
stage() {
  local line
  line="$(date -Is) [$1] $2"
  echo "${line}"
  [[ -n "${STATUS:-}" ]] && echo "${line}" >> "${STATUS}"
  return 0
}

# provenance <out.json> key=value ...: git commit, time and settings for a run directory.
provenance() { "${MAIN_PY}" -m emoseq.provenance "$@"; }

require_legacy() {
  if [[ ! -x "${LEGACY_PY}" ]]; then
    echo "legacy environment not found at ${LEGACY_PY}; run bash scripts/setup/install_legacy.sh (or set LEGACY_PY)" >&2
    exit 1
  fi
}
