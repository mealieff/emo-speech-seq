#!/usr/bin/env bash
# Create the main environment (.venv): Qwen2-Audio, Audio-Flamingo-3, WavLM, data prep, scoring.
#
#   bash scripts/setup/install.sh            # auto: CUDA 12.4 wheels if nvidia-smi is present, else default wheels
#   TORCH_VARIANT=cpu bash scripts/setup/install.sh
#   LOCKED=1 bash scripts/setup/install.sh   # exact package set of the reported runs (requirements/main.lock.txt)
#   TORCH_STACK=2.8 VENV=.venv-t28 LOCKED=1 bash scripts/setup/install.sh
#       # torch 2.8 (CUDA 12.8) + flash-attn 2.8.3: native torch._grouped_mm for Qwen3-Omni's MoE experts
#       # (2.7x faster LoRA steps on H100); used for the Qwen3-Omni utterance adapter. sympy follows torch.
#
# Env: PYTHON (default python3.12, else python3), VENV (default .venv), TORCH_VARIANT (cu124|cpu|default),
#      TORCH_STACK (2.6 default | 2.8).
# Data prep, scoring and the tests run on CPU (Linux / macOS / WSL). Model inference and
# fine-tuning need an NVIDIA GPU (see README).
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"
VENV="${VENV:-.venv}"
TORCH_STACK="${TORCH_STACK:-2.6}"
case "${TORCH_STACK}" in
  2.6) TORCH_VERSION=2.6.0 TORCHAUDIO_VERSION=2.6.0 TORCHVISION_VERSION=0.21.0 ;;
  2.8) TORCH_VERSION=2.8.0 TORCHAUDIO_VERSION=2.8.0 TORCHVISION_VERSION=0.23.0 ;;
  *) echo "TORCH_STACK must be 2.6 or 2.8" >&2; exit 1 ;;
esac

if [[ -z "${TORCH_VARIANT:-}" ]]; then
  if [[ "$(uname -s)" == Linux ]] && command -v nvidia-smi >/dev/null 2>&1; then TORCH_VARIANT=cu124; else TORCH_VARIANT=default; fi
fi
# PyPI's Linux x86_64 torch 2.8.0 wheels are the CUDA 12.8 builds
[[ "${TORCH_STACK}" == 2.8 && "${TORCH_VARIANT}" == cu124 ]] && TORCH_VARIANT=cu128

"${PYTHON}" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python >= 3.10 required (3.12 used for the reported runs); found {sys.version.split()[0]}")
EOF

echo "python=${PYTHON} venv=${VENV} torch=${TORCH_VERSION} variant=${TORCH_VARIANT}"
"${PYTHON}" -m venv "${VENV}"
PIP="${VENV}/bin/python -m pip"
${PIP} install --upgrade pip wheel

case "${TORCH_VARIANT}" in
  cu124) INDEX=(--index-url https://download.pytorch.org/whl/cu124) ;;
  cpu)   INDEX=(--index-url https://download.pytorch.org/whl/cpu) ;;
  *)     INDEX=() ;;
esac
TORCH_PINS=("torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" "torchvision==${TORCHVISION_VERSION}")
${PIP} install "${INDEX[@]}" "${TORCH_PINS[@]}"
# Qwen3-Omni runs on FlashAttention-2, the path on which its audio encoder applies its 8 s windows.
PYTAG=$("${VENV}/bin/python" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
FA=https://github.com/Dao-AILab/flash-attention/releases/download
case "${TORCH_VARIANT}" in
  cu124) ${PIP} install einops==0.8.2 "${FA}/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-${PYTAG}-${PYTAG}-linux_x86_64.whl" ;;
  cu128) ${PIP} install einops==0.8.2 "${FA}/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-${PYTAG}-${PYTAG}-linux_x86_64.whl" ;;
esac

CONSTRAINTS="$(mktemp)"
printf '%s\n' "${TORCH_PINS[@]}" > "${CONSTRAINTS}"
if [[ "${LOCKED:-0}" == 1 ]]; then
  LOCK=requirements/main.lock.txt
  if [[ "${TORCH_STACK}" == 2.8 ]]; then
    LOCK="$(mktemp)"
    grep -v '^sympy==' requirements/main.lock.txt > "${LOCK}"  # torch 2.8 needs sympy>=1.13.3
  fi
  ${PIP} install -r "${LOCK}" -c "${CONSTRAINTS}" pytest
else
  ${PIP} install -r requirements/main.txt -c "${CONSTRAINTS}"
fi
${PIP} install -e . --no-deps

[[ -f configs/paths.env ]] || { cp configs/paths.env.example configs/paths.env; echo "wrote configs/paths.env (edit MSP_ROOT / MODEL_CACHE)"; }
"${VENV}/bin/python" scripts/setup/check_env.py --main || true
echo "main environment ready: source ${VENV}/bin/activate"
