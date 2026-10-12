"""Backends for the ALMs with their own (non-Hugging-Face) stacks: SALMONN and Kimi-Audio.

Both need transformers 4.x and run in the legacy environment (scripts/setup/install_legacy.sh).
Decoding is greedy for both (SALMONN's stock 4-beam config is overridden). SALMONN's encoder
takes at most 30 s, so callers cut sequences into 30 s windows; Kimi-Audio slices long audio
into 30 s Whisper windows internally and gets the whole sequence.
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from emoseq import config

SR = 16000
KIMI_SNAPSHOT_GLOB = "models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/*"
KIMI_REVISION = "9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"


class Salmonn:
    name = "salmonn"
    native_long_audio = False

    def __init__(self, sft_dir: str | None = None) -> None:
        from emoseq.models.adapters import get_adapter

        self.adapter = get_adapter("salmonn")
        self.cfg = self.adapter._load_cfg()  # noqa: SLF001
        self.processor = self.adapter.load_processor()
        self.model = self.adapter.load_model()
        if sft_dir:
            from emoseq.models.bespoke_lora import load_trainable, salmonn_add_lora

            salmonn_add_lora(self.model)
            load_trainable(self.model, Path(sft_dir))
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        self.base_model = str(Path(self.cfg.config.model.ckpt).parent.parent)

    def _samples(self, audio: np.ndarray) -> dict:
        if len(audio) < SR:
            audio = np.concatenate([audio, np.zeros(SR - len(audio), dtype=audio.dtype)])
        audio = audio[: SR * 30]
        spec = self.processor(audio, sampling_rate=SR, return_tensors="pt")["input_features"]
        samples = {
            "spectrogram": spec,
            "raw_wav": torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),
            "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        }
        dev = next(self.model.parameters()).device
        return {k: v.to(dev) for k, v in samples.items()}

    def generate(self, instruction: str, audio: np.ndarray, max_new: int) -> str:
        return self.generate_batch(instruction, [audio], max_new)[0]

    def generate_batch(self, instruction: str, audios: list[np.ndarray], max_new: int) -> list[str]:
        """Same per-clip preprocessing as batch 1; raw audio right-padded with padding_mask=True
        on the padding (BEATs then runs per clip, see SALMONN.encode_speech)."""
        gen_cfg = copy.deepcopy(dict(self.cfg.config.generate))
        gen_cfg.update(num_beams=1, do_sample=False, max_new_tokens=max_new)
        prompt = self.cfg.config.model.prompt_template.format(
            "<Speech><SpeechHere></Speech>\n" + instruction.strip())
        items = [self._samples(a) for a in audios]
        if len(items) == 1:
            samples = items[0]
        else:
            n_max = max(it["raw_wav"].shape[1] for it in items)
            samples = {
                "spectrogram": torch.cat([it["spectrogram"] for it in items]),
                "raw_wav": torch.cat([torch.nn.functional.pad(it["raw_wav"], (0, n_max - it["raw_wav"].shape[1]))
                                      for it in items]),
                "padding_mask": torch.cat([torch.nn.functional.pad(it["padding_mask"], (0, n_max - it["padding_mask"].shape[1]),
                                                                   value=True) for it in items]),
            }
        with torch.no_grad():
            return [t.strip() for t in self.model.generate(samples, gen_cfg, prompts=[prompt] * len(items))]


class KimiAudioBackend:
    name = "kimi_audio"
    native_long_audio = True

    def __init__(self, sft_dir: str | None = None) -> None:
        root = str(config.KIMI_AUDIO_ROOT)
        if not Path(root, "kimia_infer").is_dir():
            raise SystemExit(f"Kimi-Audio source not found at {root}; run scripts/setup/install_legacy.sh")
        if root not in sys.path:
            sys.path.insert(0, root)
        from kimia_infer.api.kimia import KimiAudio

        cache = config.MODEL_CACHE
        snaps = sorted(cache.glob(KIMI_SNAPSHOT_GLOB))
        if not snaps:
            raise SystemExit(f"Kimi-Audio snapshot not found under {cache}/{KIMI_SNAPSHOT_GLOB}; "
                             "run python scripts/setup/download_models.py --models kimi_audio")
        pinned = [s for s in snaps if s.name == KIMI_REVISION]
        self.base_model = str((pinned or snaps)[-1])
        self.model = KimiAudio(model_path=self.base_model, load_detokenizer=False)
        if sft_dir:
            from emoseq.models.bespoke_lora import kimi_add_lora, load_trainable

            alm = kimi_add_lora(self.model.alm)
            load_trainable(alm, Path(sft_dir))
            self.model.alm = alm.merge_and_unload().eval()
        self.tmp = Path(tempfile.mkdtemp(prefix="kimi_"))

    def generate(self, instruction: str, audio: np.ndarray, max_new: int) -> str:
        wav = self.tmp / "clip.wav"
        sf.write(wav, audio, SR)
        # Kimi's prompt manager expects instruction before audio content (kimia.py notes).
        messages = [
            {"role": "user", "message_type": "text", "content": instruction},
            {"role": "user", "message_type": "audio", "content": str(wav)},
        ]
        _, text = self.model.generate(
            messages, output_type="text", max_new_tokens=max_new,
            text_temperature=0.0, text_top_k=1, text_repetition_penalty=1.0,
            audio_temperature=0.0, audio_top_k=1,
        )
        return (text or "").strip()

    def generate_batch(self, instruction: str, audios: list[np.ndarray], max_new: int) -> list[str]:
        """No batched path in kimia_infer; one clip per forward pass."""
        return [self.generate(instruction, a, max_new) for a in audios]


def load_backend(name: str, sft_dir: str | None = None):
    return Salmonn(sft_dir) if name == "salmonn" else KimiAudioBackend(sft_dir)
