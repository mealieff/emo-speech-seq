#!/usr/bin/env bash
# Repair .venv-t28: the lock's sympy==1.13.1 made pip fall back to torch 2.6. Hold torch at 2.8.0 and let
# sympy follow torch; every other locked pin is kept.
set -euo pipefail
cd /media/volume/AudioEmotionSequenceVol/emo-seq
V=.venv-t28
P="${V}/bin/python -m pip --retries 20 --timeout 60"
grep -v '^sympy==' requirements/main.lock.txt > /tmp/main.lock.t28.txt
printf 'torch==2.8.0\ntorchaudio==2.8.0\ntorchvision==0.23.0\n' > /tmp/t28.constraints.txt
${P} install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
${P} install -r /tmp/main.lock.t28.txt -c /tmp/t28.constraints.txt pytest
${P} install --force-reinstall --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
${P} install -e . --no-deps
${P} freeze > runs/logs/venv_t28.freeze.txt
"${V}/bin/python" - <<'EOF'
import torch, flash_attn, transformers, peft, sympy
print("torch", torch.__version__, "cuda", torch.version.cuda, "_grouped_mm", hasattr(torch, "_grouped_mm"))
print("flash_attn", flash_attn.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "sympy", sympy.__version__)
EOF
echo BUILD_OK
