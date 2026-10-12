#!/usr/bin/env bash
# Create the legacy environment (.venv-legacy) for SALMONN and Kimi-Audio, and fetch the
# Kimi-Audio source at the commit used for the reported runs.
#
# SALMONN's vendored LLaMA needs tuple KV caches and Kimi-Audio pins transformers 4.x, so these
# two models cannot share the main environment. Linux + NVIDIA GPU (CUDA 12.x driver) only:
# the flash-attn wheel is built for CUDA 12 / torch 2.6 / CPython 3.12, and Kimi-Audio moves
# tensors to the current CUDA device unconditionally.
#
#   bash scripts/setup/install_legacy.sh
#
# Env: PYTHON (default python3.12), VENV (default .venv-legacy), KIMI_AUDIO_ROOT (default third_party/Kimi-Audio).
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-$(command -v python3.12 || true)}"
VENV="${VENV:-.venv-legacy}"
KIMI_AUDIO_ROOT="${KIMI_AUDIO_ROOT:-third_party/Kimi-Audio}"
KIMI_COMMIT=349251e1d8f4f98d58fda59246381faecd7392e0
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

[[ "$(uname -s)" == Linux ]] || { echo "legacy environment is Linux + CUDA only" >&2; exit 1; }
[[ -n "${PYTHON}" ]] || { echo "python3.12 not found (the flash-attn wheel is cp312); set PYTHON=..." >&2; exit 1; }

if [[ ! -d "${KIMI_AUDIO_ROOT}/.git" ]]; then
  git clone https://github.com/MoonshotAI/Kimi-Audio.git "${KIMI_AUDIO_ROOT}"
fi
git -C "${KIMI_AUDIO_ROOT}" fetch --quiet origin
git -C "${KIMI_AUDIO_ROOT}" checkout --quiet "${KIMI_COMMIT}"
git -C "${KIMI_AUDIO_ROOT}" submodule update --init --recursive
echo "Kimi-Audio at $(git -C "${KIMI_AUDIO_ROOT}" rev-parse HEAD) in ${KIMI_AUDIO_ROOT}"

"${PYTHON}" -m venv "${VENV}"
PIP="${VENV}/bin/python -m pip"
${PIP} install --upgrade pip wheel
${PIP} install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
${PIP} install "${FLASH_ATTN_WHL}"
${PIP} install -r requirements/legacy.txt
${PIP} install -e . --no-deps

"${VENV}/bin/python" scripts/setup/check_env.py --legacy || true
echo "legacy environment ready: ${VENV}/bin/python (scripts pick it up as LEGACY_PY)"
