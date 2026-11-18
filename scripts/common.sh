REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MAIN_PY="${MAIN_PY:-${REPO_ROOT}/.venv/bin/python}"
LEGACY_PY="${LEGACY_PY:-${REPO_ROOT}/.venv-legacy/bin/python}"
if [[ ! -x "${MAIN_PY}" ]]; then
  echo "run bash scripts/setup/install.sh (or set MAIN_PY)" >&2
  exit 1
fi

eval "$("${MAIN_PY}" -m emoseq.config)"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${MODEL_CACHE}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${MODEL_CACHE}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

LOG_DIR="${RUNS_DIR}/logs"
mkdir -p "${LOG_DIR}"

stage() {
  local line
  line="$(date -Is) [$1] $2"
  echo "${line}"
  [[ -n "${STATUS:-}" ]] && echo "${line}" >> "${STATUS}"
  return 0
}

provenance() { "${MAIN_PY}" -m emoseq.provenance "$@"; }

require_legacy() {
  if [[ ! -x "${LEGACY_PY}" ]]; then
    echo "run bash scripts/setup/install_legacy.sh (or set LEGACY_PY)" >&2
    exit 1
  fi
}
