#!/usr/bin/env bash
# Second main environment (.venv-t28): torch 2.8 (native torch._grouped_mm for the Qwen3-Omni MoE experts
# on H100), flash-attn 2.8.3 built for torch 2.8, otherwise the locked package set of .venv.
set -euo pipefail
cd /media/volume/AudioEmotionSequenceVol/emo-seq
V=.venv-t28
/home/exouser/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12 -m venv "${V}"
P="${V}/bin/python -m pip --retries 20 --timeout 60"
${P} install --upgrade pip wheel
# PyPI's Linux x86_64 torch 2.8.0 wheels are the CUDA 12.8 builds
${P} install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
${P} install einops==0.8.2 \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
${P} install -r requirements/main.lock.txt pytest
${P} install -e . --no-deps
"${V}/bin/python" - <<'EOF'
import torch, flash_attn, transformers, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "_grouped_mm", hasattr(torch, "_grouped_mm"))
print("flash_attn", flash_attn.__version__, "transformers", transformers.__version__, "peft", peft.__version__)
EOF
echo BUILD_OK
