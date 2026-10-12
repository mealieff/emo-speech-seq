#!/usr/bin/env python3
"""Download every pretrained model at the revision used for the reported runs.

Hugging Face models go to the hub cache at ``MODEL_CACHE`` (``models--org--name/snapshots/<rev>``),
which is where the scripts look for them. SALMONN's components are laid out under
``SALMONN_DIR/{vicuna,whisper,beats,salmonn}``; the IS2025 WavLM checkpoint is unpacked to
``WAVLM_IS2025_DIR``. Single-file downloads are checked against their SHA-256.

    python scripts/setup/download_models.py                       # main-environment models
    python scripts/setup/download_models.py --models salmonn kimi_audio
    python scripts/setup/download_models.py --models all

Disk: Qwen2-Audio ~16 GB, Audio-Flamingo-3 ~17 GB, Kimi-Audio ~20 GB (+ GLM-4-Voice tokenizer),
SALMONN ~30 GB (Vicuna, Whisper, BEATs, SALMONN), WavLM ~1.3 GB + IS2025 ~1.3 GB,
Qwen3-Omni ~70 GB (optional; not part of the reported tables).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from emoseq import config

HF_MODELS = {
    "qwen2_audio": [("Qwen/Qwen2-Audio-7B-Instruct", "0a095220c30b7b31434169c3086508ef3ea5bf0a")],
    "audio_flamingo3": [("nvidia/audio-flamingo-3-hf", "5c47d2de83c3163162586561f3ddba75b4a1194e")],
    "kimi_audio": [
        ("moonshotai/Kimi-Audio-7B-Instruct", "9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"),
        ("THUDM/glm-4-voice-tokenizer", "a5f2404e63c84e92f5238908e1706316324ebafa"),
    ],
    "wavlm": [("microsoft/wavlm-large", "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c")],
    "qwen3_omni": [("Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695")],
}
SALMONN_PARTS = {
    "vicuna": ("lmsys/vicuna-7b-v1.5", "3321f76e3f527bd14065daf69dad9344000a201d", None),
    "whisper": ("openai/whisper-large-v2", "ae4642769ce2ad8fc292556ccea8e901f1530655",
                ["*.h5", "*.msgpack", "pytorch_model.bin"]),
}
SALMONN_FILES = {
    "beats": ("camenduru/beats", "6e898b79eed92c81c4bb43a21f52fc39f4f849f0",
              "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
              "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c"),
    "salmonn": ("tsinghua-ee/SALMONN-7B", "ed583f2c565c1110b3bf5772ed42d6744f3cc443", "salmonn_7b_v0.pth",
                "2cb2782495b2e3f487222763a30b53b02f727d49059201cc5fa88a7a1fd9dff9"),
}
IS2025_URL = "https://lab-msp.com/MSP-Podcast_Competition/IS2025/models/cat_ser.zip"
IS2025_SHA256 = "63673f21c58adcc7589762e97052c98ec90e8f2060b67e1ff1dbcb517cb0fbc7"
IS2025_FILES = ("final_ssl.pt", "final_pool.pt", "final_ser.pt", "train_norm_stat.pkl")

MAIN = ["qwen2_audio", "audio_flamingo3", "wavlm", "is2025"]
LEGACY = ["salmonn", "kimi_audio"]
CHOICES = MAIN + LEGACY + ["qwen3_omni"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def check_sha(path: Path, expected: str) -> None:
    got = sha256(path)
    if got != expected:
        sys.exit(f"SHA-256 mismatch for {path}: {got} != {expected}")
    print(f"  sha256 ok {path.name}")


def hf_snapshot(repo: str, revision: str, **kw) -> None:
    from huggingface_hub import snapshot_download

    print(f"{repo}@{revision[:10]}", flush=True)
    snapshot_download(repo_id=repo, revision=revision, cache_dir=str(config.MODEL_CACHE), **kw)
    # Code that loads by repo name (Kimi-Audio's GLM tokenizer) resolves "main"; with
    # HF_HUB_OFFLINE=1 this pointer makes that the pinned revision.
    ref = config.MODEL_CACHE / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    if not ref.is_file():
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(revision)


def salmonn() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    root = config.SALMONN_DIR
    for part, (repo, rev, ignore) in SALMONN_PARTS.items():
        print(f"{repo}@{rev[:10]} -> {root / part}", flush=True)
        snapshot_download(repo_id=repo, revision=rev, local_dir=str(root / part), ignore_patterns=ignore)
    for part, (repo, rev, filename, digest) in SALMONN_FILES.items():
        dest = root / part / filename
        if not dest.is_file():
            print(f"{repo}@{rev[:10]}/{filename} -> {dest}", flush=True)
            hf_hub_download(repo_id=repo, revision=rev, filename=filename, local_dir=str(root / part))
        check_sha(dest, digest)


def is2025() -> None:
    dest = config.WAVLM_IS2025_DIR
    if all((dest / f).is_file() for f in IS2025_FILES):
        print(f"IS2025 checkpoint present at {dest}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "cat_ser.zip"
        print(f"{IS2025_URL} -> {dest}", flush=True)
        urllib.request.urlretrieve(IS2025_URL, zpath)
        check_sha(zpath, IS2025_SHA256)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        src = next(p.parent for p in Path(tmp).rglob("final_ssl.pt") if p.parent.name == "7")
        dest.mkdir(parents=True, exist_ok=True)
        for f in IS2025_FILES:
            shutil.copy2(src / f, dest / f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["main"], choices=CHOICES + ["main", "legacy", "all"],
                   help="main = Qwen2-Audio, AF3, WavLM, IS2025; legacy = SALMONN, Kimi-Audio; "
                        "all = main + legacy (Qwen3-Omni only when named)")
    args = p.parse_args()
    todo: list[str] = []
    for m in args.models:
        todo += {"main": MAIN, "legacy": LEGACY, "all": MAIN + LEGACY}.get(m, [m])
    print(f"MODEL_CACHE={config.MODEL_CACHE}  SALMONN_DIR={config.SALMONN_DIR}")
    for m in dict.fromkeys(todo):
        if m == "salmonn":
            salmonn()
        elif m == "is2025":
            is2025()
        else:
            for repo, rev in HF_MODELS[m]:
                hf_snapshot(repo, rev)
    print("done")


if __name__ == "__main__":
    main()
