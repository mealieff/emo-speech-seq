"""Helpers for qwen3_preflight.sh.

    python data_tools.py hash <out_dir> <jsonl>...         sha256 of each JSONL + every referenced wav
    python data_tools.py longest <jsonl> <k> <out.jsonl>   the k longest rows by audio_duration
    python data_tools.py over <cap_sec> <jsonl>...         rows above a duration cap, per file
    python data_tools.py timing <trainer_state.json> <n_train> <grad_accum> <epochs>
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def cmd_hash(out_dir: str, *jsonls: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"jsonl_sha256": {}, "audio": {}}
    wavs: dict[str, Path] = {}
    for j in jsonls:
        summary["jsonl_sha256"][Path(j).name] = sha256(Path(j))
        for r in rows(j):
            wavs[Path(r["audio_path"]).name] = Path(r["audio_path"])
    lines = [f"{sha256(p)}  {name}" for name, p in sorted(wavs.items())]
    (out / "audio_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["audio"] = {"n_files": len(lines),
                        "manifest_sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()}
    (out / "data_hashes.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    flat = {f"sha256_{k}": v for k, v in summary["jsonl_sha256"].items()}
    flat["audio_manifest_sha256"] = summary["audio"]["manifest_sha256"]
    flat["audio_n_files"] = summary["audio"]["n_files"]
    print(" ".join(f"{k}={v}" for k, v in flat.items()))


def cmd_longest(jsonl: str, k: str, out: str) -> None:
    rs = sorted(rows(jsonl), key=lambda r: float(r.get("audio_duration") or 0), reverse=True)[: int(k)]
    Path(out).write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8")
    print(" ".join(f"{r['seq_id']}:{float(r['audio_duration']):.1f}s" for r in rs))


def cmd_over(cap: str, *jsonls: str) -> None:
    for j in jsonls:
        rs = rows(j)
        n = sum(float(r.get("audio_duration") or 0) > float(cap) for r in rs)
        print(f"{Path(j).name}: {n}/{len(rs)} rows > {cap} s ({100 * n / max(1, len(rs)):.1f}%)")


def cmd_timing(state: str, n_train: str, grad_accum: str, epochs: str) -> None:
    s = json.loads(Path(state).read_text())
    runtime = next(e["train_runtime"] for e in s["log_history"] if "train_runtime" in e)
    sec_per_step = runtime / s["global_step"]
    steps_per_epoch = -(-int(n_train) // int(grad_accum))
    hours = sec_per_step * steps_per_epoch * float(epochs) / 3600
    print(f"{s['global_step']} steps in {runtime:.0f} s = {sec_per_step:.1f} s/step; "
          f"{steps_per_epoch} steps/epoch x {epochs} epochs = {hours:.1f} h training "
          f"(plus per-epoch Dev-256 eval and full-Dev re-selection)")


def cmd_timing_log(log: str, steps: str, n_train: str, eff_batch: str, epochs: str) -> None:
    import re

    runtime = float(re.findall(r"'train_runtime': '?([\d.]+(?:e[+-]?\d+)?)", Path(log).read_text(errors="replace"))[-1])
    sec_per_step = runtime / int(steps)
    steps_per_epoch = -(-int(n_train) // int(eff_batch))
    hours = sec_per_step * steps_per_epoch * float(epochs) / 3600
    print(f"{steps} steps in {runtime:.0f} s = {sec_per_step:.2f} s/step; "
          f"{steps_per_epoch} steps/epoch x {epochs} epochs = {hours:.1f} h training")


if __name__ == "__main__":
    {"hash": cmd_hash, "longest": cmd_longest, "over": cmd_over, "timing": cmd_timing,
     "timing_log": cmd_timing_log}[sys.argv[1]](*sys.argv[2:])
