"""Run a repo script and report peak CUDA memory: python peak.py <script.py> [args...]"""
import atexit
import runpy
import sys

import torch


@atexit.register
def _report():
    if torch.cuda.is_available():
        print(f"[peak] max_allocated={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  "
              f"max_reserved={torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)


sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
