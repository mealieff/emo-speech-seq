#!/usr/bin/env python3
"""Model-adapter registry for the audio LLMs.

Each model is wrapped in a ``ModelAdapter`` that hides how its model and processor are
loaded, how the audio placeholder is spliced into the prompt, which processor kwarg takes
raw audio, which sub-modules can be frozen, and which modules LoRA targets.

  * qwen2_audio     — AutoProcessor + Qwen2AudioForConditionalGeneration,
                      audio token ``<|audio_bos|><|AUDIO|><|audio_eos|>``, kwarg ``audio``.
  * audio_flamingo3 — AutoProcessor + AudioFlamingo3ForConditionalGeneration,
                      audio placeholder ``<sound>``, kwarg ``audio``.
  * qwen3_omni      — Qwen3OmniMoeProcessor + Qwen3OmniMoeThinkerForConditionalGeneration
                      (thinker only). Not run for the paper: needs more than 40 GB in bf16.
  * salmonn         — vendored stack (Whisper + BEATs + Q-Former + Vicuna-7B) from
                      third_party/salmonn + configs/salmonn_decode.yaml.

Hugging Face weights are read from ``MODEL_CACHE`` (a hub cache); SALMONN components from
``SALMONN_DIR``. Kimi-Audio is driven through its own inference package
(``emoseq.models.bespoke``) and has no adapter here.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from emoseq import config

EMOTION_NAMES = ["Angry", "Sad", "Happy", "Surprised", "Fear", "Disgust", "Contempt", "Neutral"]

SALMONN_SRC = config.REPO / "third_party" / "salmonn"

# LLM attention projections — the safe, architecture-agnostic LoRA default.
LLM_ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _import_attr(path: str) -> Any:
    """Import 'pkg.module.Attr' or 'transformers.Attr'."""
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"Cannot import from path without a module: {path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def model_cache_root() -> Path:
    return config.MODEL_CACHE


@dataclass
class ModelAdapter:
    """Everything the trainer/eval/collators need to drive one model.

    The base class implements the HuggingFace-native path (AutoProcessor / a
    `from_pretrained` model class). Custom models subclass and override
    `load_model` / `load_processor` / `process`.
    """

    key: str
    repo_dirname: str                 # dir under the HF cache root
    native: bool                      # HF-native loadable via from_pretrained?
    model_class_path: str             # import path of the model class
    processor_class_path: str         # import path of the processor/feature builder
    audio_placeholder: str            # text spliced where audio is attended
    prompt_template: str              # uses {audio} and {instruction}
    audio_kwarg: str                  # processor kwarg for raw waveforms
    lora_target_modules: list[str]
    freeze_prefixes: dict[str, str]   # logical name -> param-name prefix
    supports_qlora: bool = False
    supports_generic_sft: bool = True  # fits the generic HF processor/Trainer path?
    # Zero-shot / --base-only eval: some instruct models ignore the SFT "Output:"
    # suffix and emit an immediate chat end token. Wrap the same instruction in the
    # native chat template instead (finetuned eval still uses format_prompt).
    use_chat_template_for_base_inference: bool = False
    default_sampling_rate: int = 16000
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    process_kwargs: dict[str, Any] = field(default_factory=dict)  # extra kwargs for processor()
    generation_overrides: dict[str, Any] = field(default_factory=dict)  # set on model.generation_config
    notes: str = ""
    confirm_before_train: list[str] = field(default_factory=list)

    # ── prompt assembly ────────────────────────────────────────────────────
    def format_prompt(self, instruction: str) -> str:
        return self.prompt_template.format(audio=self.audio_placeholder, instruction=instruction)

    # ── loading (HF-native default) ──────────────────────────────────────────
    def load_processor(self, base_model: str) -> Any:
        cls = _import_attr(self.processor_class_path)
        processor = cls.from_pretrained(base_model, trust_remote_code=True)
        if self.key == "qwen3_omni":
            from emoseq.models import qwen3_omni_fix
            qwen3_omni_fix.fix_processor(processor)
        return processor

    def load_model(self, base_model: str, dtype: Any, quantization_config: Any = None,
                   device_map: Any = None) -> Any:
        cls = _import_attr(self.model_class_path)
        kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        kwargs.update(self.load_kwargs)
        if self.key == "qwen3_omni":
            from emoseq.models import qwen3_omni_fix
            qwen3_omni_fix.apply()
            kwargs.setdefault("attn_implementation", qwen3_omni_fix.attention_backend())
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
        else:
            kwargs["torch_dtype"] = dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = cls.from_pretrained(base_model, **kwargs)
        for key, value in self.generation_overrides.items():
            setattr(model.generation_config, key, value)
        return model

    # ── processor I/O ──────────────────────────────────────────────────────
    def sampling_rate(self, processor: Any) -> int:
        fe = getattr(processor, "feature_extractor", None)
        return int(getattr(fe, "sampling_rate", self.default_sampling_rate))

    def target_tokenizer(self, processor: Any) -> Any:
        return getattr(processor, "tokenizer", processor)

    def process(self, processor: Any, texts: list[str], audios: list, sampling_rate: int):
        """Build model inputs for a batch of (prompt, audio) pairs."""
        audio_kwargs = dict(self.process_kwargs.get("audio_kwargs", {}))
        text_kwargs = dict(self.process_kwargs.get("text_kwargs", {}))
        extra = {k: v for k, v in self.process_kwargs.items() if k not in ("audio_kwargs", "text_kwargs")}
        common = dict(text=texts, **{self.audio_kwarg: audios}, sampling_rate=sampling_rate, return_tensors="pt", **extra)
        # Most processors (Qwen2-Audio, etc.) expect top-level padding. Audio-Flamingo-3
        # needs mel padding via audio_kwargs; passing both causes a duplicate-padding error.
        if audio_kwargs or text_kwargs:
            text_kwargs.setdefault("padding", True)
            return processor(**common, audio_kwargs=audio_kwargs, text_kwargs=text_kwargs)
        return processor(**common, padding=True)

    def build_generation_inputs(
        self,
        processor: Any,
        instruction: str,
        audio,
        sampling_rate: int,
        *,
        audio_path: str | None = None,
        base_only: bool = False,
        prompt_layout: str = "user",
    ) -> tuple[Any, str]:
        """Tokenize one (instruction, audio) pair for eval-time generation.

        Returns (processor outputs, prompt_mode) where prompt_mode is
        ``chat_template`` or ``sft_output_suffix``.
        ``prompt_layout=user`` (default) folds the instruction into the user
        turn. ``system`` puts it in a system turn and leaves the user audio-only.
        """
        if base_only and self.use_chat_template_for_base_inference:
            if audio_path is None:
                raise ValueError(
                    f"[{self.key}] chat-template base inference requires audio_path."
                )
            audio_block = {"type": "audio", "audio_url": audio_path}
            if prompt_layout == "system":
                conversation = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": [audio_block]},
                ]
            elif prompt_layout == "user":
                conversation = [{
                    "role": "user",
                    "content": [audio_block, {"type": "text", "text": instruction}],
                }]
            else:
                raise ValueError(f"unknown prompt_layout {prompt_layout!r}")
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            return self.process(processor, [text], [audio], sampling_rate), "chat_template"
        prompt = self.format_prompt(instruction)
        return self.process(processor, [prompt], [audio], sampling_rate), "sft_output_suffix"


# ── native adapter factory ──────────────────────────────────────────────────
def _native(
    key: str,
    repo_dirname: str,
    model_class_path: str,
    audio_placeholder: str,
    prompt_template: str,
    audio_kwarg: str = "audio",
    freeze_prefixes: dict[str, str] | None = None,
    supports_qlora: bool = False,
    processor_class_path: str = "transformers.AutoProcessor",
    process_kwargs: dict[str, Any] | None = None,
    generation_overrides: dict[str, Any] | None = None,
    notes: str = "",
    confirm_before_train: list[str] | None = None,
    use_chat_template_for_base_inference: bool = False,
) -> ModelAdapter:
    return ModelAdapter(
        key=key,
        repo_dirname=repo_dirname,
        native=True,
        model_class_path=model_class_path,
        processor_class_path=processor_class_path,
        audio_placeholder=audio_placeholder,
        prompt_template=prompt_template,
        audio_kwarg=audio_kwarg,
        lora_target_modules=list(LLM_ATTN_TARGETS),
        freeze_prefixes=freeze_prefixes
        or {"audio_tower": "audio_tower", "multi_modal_projector": "multi_modal_projector", "language_model": "language_model"},
        supports_qlora=supports_qlora,
        use_chat_template_for_base_inference=use_chat_template_for_base_inference,
        process_kwargs=process_kwargs or {},
        generation_overrides=generation_overrides or {},
        notes=notes,
        confirm_before_train=confirm_before_train or [],
    )


# Qwen2-Audio uses explicit audio sentinel tokens.
_QWEN2_AUDIO = _native(
    key="qwen2_audio",
    repo_dirname="models--Qwen--Qwen2-Audio-7B-Instruct",
    model_class_path="transformers.Qwen2AudioForConditionalGeneration",
    audio_placeholder="<|audio_bos|><|AUDIO|><|audio_eos|>",
    prompt_template="{audio}{instruction}\nOutput:",
    audio_kwarg="audio",
    freeze_prefixes={
        "audio_tower": "audio_tower",
        "multi_modal_projector": "multi_modal_projector",
        "language_model": "language_model",
    },
    use_chat_template_for_base_inference=True,
    notes="Reference implementation.",
)

# Qwen3-Omni is a MoE thinker+talker. For SER we load only the *thinker*
# (text+audio understanding) to avoid the talker/code2wav TTS stack and its VRAM.
# Thinker modules: audio_tower (Whisper-style encoder ending in proj1/proj2, its connector
# to the LLM), visual, model (48-layer MoE LLM), lm_head.
_QWEN3_OMNI = _native(
    key="qwen3_omni",
    repo_dirname="models--Qwen--Qwen3-Omni-30B-A3B-Instruct",
    model_class_path="transformers.Qwen3OmniMoeThinkerForConditionalGeneration",
    processor_class_path="transformers.Qwen3OmniMoeProcessor",
    audio_placeholder="<|audio_start|><|audio_pad|><|audio_end|>",
    prompt_template="{audio}{instruction}\nOutput:",
    audio_kwarg="audio",
    freeze_prefixes={
        "audio_tower": "audio_tower",
        "multi_modal_projector": "audio_tower.proj",
        "language_model": "model",
    },
    use_chat_template_for_base_inference=True,
    # The checkpoint's generation_config.json only carries talker settings, so the thinker
    # loaded alone has no stop token and decodes to max_new_tokens.
    generation_overrides={"eos_token_id": [151645, 151643], "pad_token_id": 151643},
    notes="MoE, ~3B active per token, 59 GiB of bf16 thinker weights. Loading the Thinker class only. "
          "Requires torchvision (the processor pulls in a video processor). Inference and LoRA "
          "training fit one 80 GB GPU for sequences up to ~300 s; longer ones need --device-map auto. "
          "bitsandbytes 4-bit does not quantize the fused MoE experts under transformers 5.x.",
)

# Audio-Flamingo-3 (NVIDIA) is HF-native; loads without trust_remote_code.
_AUDIO_FLAMINGO3 = _native(
    key="audio_flamingo3",
    repo_dirname="models--nvidia--audio-flamingo-3-hf",
    model_class_path="transformers.AudioFlamingo3ForConditionalGeneration",
    audio_placeholder="<sound>",
    prompt_template="{audio}{instruction}\nOutput:",
    audio_kwarg="audio",
    freeze_prefixes={
        "audio_tower": "audio_tower",
        "multi_modal_projector": "multi_modal_projector",
        "language_model": "language_model",
    },
    notes="HF-native AudioFlamingo3ForConditionalGeneration; plain-text prompt with a "
          "'<sound>' placeholder. Mel features must be padded to the 30 s Whisper window "
          "(3000 frames) or the audio encoder position embeddings mismatch.",
    process_kwargs={"audio_kwargs": {"padding": "max_length", "truncation": True}},
    confirm_before_train=[
        "Confirm freeze_prefixes and LoRA targets against the loaded module tree.",
    ],
)


# ── custom (vendored) adapters ────────────────────────────────────────────────
# These drive the vendored model code under third_party/ so weights load from the
# correct component paths. They do NOT fit the generic HF processor/Trainer SFT
# path (bespoke forward/generate), so supports_generic_sft=False.

_GENERIC_SFT_MSG = (
    "uses a bespoke (non-HuggingFace) forward/generate path and cannot be driven by "
    "the generic processor/Trainer SFT pipeline. Load it with this adapter and run its "
    "native inference/training recipe (see the adapter docstring)."
)


@dataclass
class SalmonnAdapter(ModelAdapter):
    """SALMONN-7B: Whisper + BEATs -> Q-Former -> Vicuna-7B (vendored stack).

    Source is vendored at third_party/salmonn. The load/decode config is
    configs/salmonn_decode.yaml; this adapter overrides its component-weight paths from
    ``SALMONN_DIR`` ({vicuna,whisper,beats,salmonn}/) and its prompt paths from the vendored
    prompts. Inference and LoRA training go through ``emoseq.models.bespoke.Salmonn``.

    LoRA (rank 8, alpha 32, dropout 0.1) on the Vicuna LLM is configured inside
    SALMONN.from_config via the YAML (``lora: True``); Whisper/BEATs are frozen.
    """

    _cfg: Any = None

    def _load_cfg(self) -> Any:
        root = SALMONN_SRC
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import argparse
        from config import Config as SALMONNConfig  # type: ignore

        cfg_path = config.REPO / "configs" / "salmonn_decode.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"[salmonn] decode config not found at {cfg_path}")
        cfg = SALMONNConfig(argparse.Namespace(cfg_path=str(cfg_path), options=None))

        # Resolve all path fields against SALMONN_DIR, which is not the Hugging Face hub cache.
        # A missing directory is not a Hub repo id: from_pretrained would reject the absolute path.
        cache = config.SALMONN_DIR
        beats = cache / "beats" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
        ckpt = cache / "salmonn" / "salmonn_7b_v0.pth"
        needed = {"vicuna": cache / "vicuna", "whisper": cache / "whisper", "beats": beats, "ckpt": ckpt}
        missing = [f"{name} ({path})" for name, path in needed.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"[salmonn] weights not found under {cache} (SALMONN_DIR). "
                f"Missing: {'; '.join(missing)}. "
                "Set SALMONN_DIR to the directory that contains vicuna/, whisper/, beats/ and salmonn/."
            )
        m = cfg.config.model
        m.llama_path = str(needed["vicuna"])
        m.whisper_path = str(needed["whisper"])
        m.beats_path = str(beats)
        m.ckpt = str(ckpt)
        m.prompt_path = str(root / "prompts" / "train_prompt.json")
        m.test_prompt_path = str(root / "prompts" / "test_prompt.json")

        self._cfg = cfg
        return self._cfg

    def load_model(self, base_model: str | None = None, dtype: Any = None,
                   quantization_config: Any = None, device_map: Any = None) -> Any:
        cfg = self._cfg or self._load_cfg()
        from models.salmonn import SALMONN  # type: ignore
        model = SALMONN.from_config(cfg.config.model)
        model.eval()
        return model

    def load_processor(self, base_model: str | None = None) -> Any:
        cfg = self._cfg or self._load_cfg()
        from transformers import WhisperFeatureExtractor
        return WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    def process(self, processor, texts, audios, sampling_rate):  # pragma: no cover
        raise NotImplementedError(f"[salmonn] {_GENERIC_SFT_MSG}")


_SALMONN = SalmonnAdapter(
    key="salmonn",
    repo_dirname="models--tsinghua-ee--SALMONN-7B",
    native=False,
    model_class_path="",
    processor_class_path="",
    audio_placeholder="<Speech><SpeechHere></Speech>",
    prompt_template="USER: {audio}\n{instruction}\nASSISTANT:",
    audio_kwarg="",
    lora_target_modules=list(LLM_ATTN_TARGETS),
    freeze_prefixes={"speech_encoder": "speech_encoder", "beats": "beats"},
    supports_qlora=False,
    supports_generic_sft=False,
    notes="Vendored SALMONN stack (third_party/salmonn); components under "
          "$SALMONN_DIR/{vicuna,whisper,beats,salmonn}. Runs in the legacy environment "
          "(transformers 4.44) via scripts/inference/bespoke.py.",
)

ADAPTERS: dict[str, ModelAdapter] = {
    a.key: a
    for a in (_QWEN2_AUDIO, _AUDIO_FLAMINGO3, _SALMONN, _QWEN3_OMNI)
}

# Hugging Face revisions used for every reported run (scripts/setup/download_models.sh).
PINNED_REVISIONS = {
    "qwen2_audio": "0a095220c30b7b31434169c3086508ef3ea5bf0a",
    "audio_flamingo3": "5c47d2de83c3163162586561f3ddba75b4a1194e",
    "qwen3_omni": "26291f793822fb6be9555850f06dfe95f2d7e695",
}

# Canonical order used in result tables / paper.
MODEL_ORDER = ["qwen2_audio", "audio_flamingo3", "salmonn", "qwen3_omni"]


def get_adapter(key: str) -> ModelAdapter:
    if key not in ADAPTERS:
        raise KeyError(f"Unknown model key {key!r}. Known: {sorted(ADAPTERS)}")
    return ADAPTERS[key]


def list_models() -> list[str]:
    return list(MODEL_ORDER)


def resolve_snapshot(key: str, cache_root: str | None = None) -> str:
    """Return the on-disk snapshot dir for a model, or raise with guidance.

    Picks the snapshot under <cache_root>/<repo_dirname>/snapshots/: the pinned revision
    if present, else the only snapshot, else the HF cache `refs/main` pointer, else the
    most recently modified snapshot.
    """
    adapter = get_adapter(key)
    repo = (Path(cache_root) if cache_root else model_cache_root()) / adapter.repo_dirname
    root = repo / "snapshots"
    if not root.is_dir():
        raise FileNotFoundError(f"[{key}] no snapshots dir at {root}")
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    if not snaps:
        raise FileNotFoundError(f"[{key}] no snapshot revisions under {root}")
    pinned = PINNED_REVISIONS.get(key)
    if pinned and (root / pinned).is_dir():
        return str(root / pinned)
    if len(snaps) == 1:
        return str(snaps[0])

    ref = repo / "refs" / "main"
    if ref.exists():
        rev = ref.read_text().strip()
        cand = root / rev
        if cand.is_dir():
            return str(cand)
    # Fall back to the newest snapshot.
    return str(max(snaps, key=lambda p: p.stat().st_mtime))
