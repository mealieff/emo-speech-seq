#!/usr/bin/env python3
"""Report what is installed and which data / model paths resolve. Exits non-zero on a missing package.

    python scripts/setup/check_env.py --main     # main environment
    python scripts/setup/check_env.py --legacy   # legacy environment (SALMONN, Kimi-Audio)
"""
from __future__ import annotations

import argparse
import importlib
import sys

from emoseq import config

MAIN = {"torch": "2.6.0", "torchaudio": "2.6.0", "transformers": None, "peft": "0.18.1", "accelerate": "1.13.0",
        "datasets": "4.8.4", "librosa": "0.11.0", "soundfile": "0.13.1", "numpy": "2.4.4", "pandas": "3.0.2",
        "sklearn": "1.8.0", "scipy": "1.17.1"}
LEGACY = {"torch": "2.6.0", "transformers": "4.44.1", "peft": "0.12.0", "tokenizers": "0.19.1", "omegaconf": "2.3.0",
          "whisper": None, "flash_attn": "2.7.4.post1", "librosa": None, "soundfile": None}


def check_packages(wanted: dict[str, str | None]) -> bool:
    ok = True
    for name, want in wanted.items():
        try:
            mod = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            print(f"  MISSING  {name}: {type(e).__name__}: {e}")
            ok = False
            continue
        ver = getattr(mod, "__version__", "?")
        flag = "" if want is None or ver.split("+")[0] == want else f"   (reported runs: {want})"
        print(f"  ok       {name} {ver}{flag}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--main", action="store_true")
    g.add_argument("--legacy", action="store_true")
    args = p.parse_args()

    print(f"python {sys.version.split()[0]}")
    ok = check_packages(LEGACY if args.legacy else MAIN)
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"  cuda     {torch.version.cuda}  {props.name}  {props.total_memory / 2**30:.0f} GiB")
        else:
            print("  cuda     not available (data prep, scoring and tests only)")
    except Exception:  # noqa: BLE001
        pass

    print("paths (configs/paths.env or environment):")
    rows = [
        ("MSP_ROOT", config.MSP_AUDIO),
        ("utterance JSONL", config.utterance_jsonl("test1")),
        ("sequence JSONL", config.sequence_jsonl("test1")),
        ("CONCAT_DIR", config.CONCAT_DIR),
        ("MODEL_CACHE", config.MODEL_CACHE),
        ("SALMONN_DIR", config.SALMONN_DIR / "salmonn" / "salmonn_7b_v0.pth"),
        ("KIMI_AUDIO_ROOT", config.KIMI_AUDIO_ROOT / "kimia_infer"),
        ("WAVLM_IS2025_DIR", config.WAVLM_IS2025_DIR / "final_ssl.pt"),
        ("WAVLM_TIMEAUG_CKPT", config.WAVLM_TIMEAUG_CKPT),
    ]
    for name, path in rows:
        print(f"  {'found  ' if path.exists() else 'missing'}  {name:18s} {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
