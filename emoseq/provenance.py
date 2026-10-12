"""Write provenance.json next to a run: git commit, time, and key=value settings.

    python -m emoseq.provenance runs/test1/sequence/qwen2_audio_pt/provenance.json batch_size=1 tag=pretrained
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from emoseq.config import REPO


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def git_commit() -> str:
    return git("rev-parse", "HEAD")


def packages() -> dict[str, str]:
    from importlib import metadata

    out = {}
    for name in ("torch", "transformers", "peft", "flash-attn"):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        version = dist.version
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        out[name] = f"{version} (git {commit})" if commit else version
    return out


def write(out_json: str | Path, **settings: str) -> None:
    info = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_tracked_changes": len([l for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l]),
        "packages": packages(),
        **settings,
    }
    if settings.get("model") == "qwen3_omni" and "attn_implementation" not in settings:
        from emoseq.models.qwen3_omni_fix import attention_backend

        info["attn_implementation"] = attention_backend()
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    out, *kv = sys.argv[1:]
    write(out, **dict(x.split("=", 1) for x in kv))
