# emo-span — methods notes (paper)

Living notes for the methods section. Hyperparameters below are read from
Per-model hyperparameters live in `model/{model}/task_{a,b}.json`; trainable-parameter counts are
emitted by `scripts/finetune.py` at train start (`Trainable params: ...`) and
should be pasted in once each run begins.

## Tasks

See [`docs/MODELS.md`](MODELS.md). Current protocol:

- **Utterance SER.** One MSP-Podcast clip → one of 8 emotion labels.
- **Sequence SER.** Unique-ID back-to-back concat (L≥2, L≤40, drop X/O) →
  chronological `(start, end, emotion)` spans, scored by quartile `{2, 3, 4-6, 7+}`.

## LoRA configuration per model

LoRA is applied to the LLM attention projections (`q_proj, k_proj, v_proj,
o_proj`); the audio encoder is frozen and excluded from adaptation. Targets must
be confirmed against each loaded module tree before a real run (see each config's
`confirm_before_train`).

| Model | rank r | alpha | dropout | target_modules | notes |
|---|---|---|---|---|---|
| qwen2_audio | 16 | 32 | 0.05 | q,k,v,o_proj | reference impl |
| salmonn | 8 | 32 | 0.1 | (PEFT default) | LoRA on Vicuna-7B, set inside SALMONN.from_config via its YAML |
| mellow | n/a | n/a | n/a | n/a | no LoRA (full custom model, frozen encoder + SmolLM2 decoder) |
| qwen3_omni | 8 | 16 | 0.05 | q,k,v,o_proj | 30B MoE Thinker; QLoRA fallback |
| audio_flamingo3 | 16 | 32 | 0.05 | q,k,v,o_proj | HF-native |

salmonn/mellow LoRA values come from the vendored configs
(`configs/salmonn_decode.yaml` for SALMONN), not from this repo's
`ser_training.json` (those are documentation-only for the custom models, which do
not use the generic trainer).

Utterance LoRA uses `model/{model}/task_a.json`. Sequence ALM LoRA is
`scripts/E3/finetune/train_lora.py` (r=16, alpha=32, audio tower frozen).

## Trainable parameter counts

Fill in from each run's startup log (`Trainable params: X / Y (Z%)`).

| Model | Task A trainable / total (%) | Task B trainable / total (%) |
|---|---|---|
| qwen2_audio | _TBD_ | _TBD_ |
| salmonn | _TBD_ | _TBD_ |
| mellow | _TBD_ | _TBD_ |
| qwen3_omni | _TBD_ | _TBD_ |
| audio_flamingo3 | _TBD_ | _TBD_ |

## Controlled design choices

- **Official splits.** Train / Development / Test1 from Partitions.txt. Test3
  is unlabeled packed IDs and is not used for sequence train/eval.
- **Sequence audio.** Podcast-clip concatenation at 16 kHz (no inserted silence).
  Gold timestamps are cumulative clip durations. Manifests:
  `python3 scripts/E7/data/build_manifests.py`.
- **Prompt.** Frozen open-span instruction in `scripts/prompts/sequence.txt`.
- **Audio over 30 s.** Models whose encoder takes at most 30 s follow one rule per
  setting. Pretrained (Qwen2-Audio PT, SALMONN PT): non-overlapping 30 s windows,
  each window's spans shifted by its offset. Spans A and B are merged only if they are
  in consecutive windows, A ends and B starts at the shared window boundary (within
  0.05 s), and they have the same label; merging undoes the split that windowing
  creates and never changes what a model predicted inside a window. Qwen2-Audio SFT: truncated to the first 30 s, matching its training
  (203 / 3,648 Test1 sequences affected). SALMONN SFT is trained and decoded on the
  same 30 s windows. Audio-Flamingo-3 and Kimi-Audio take the whole sequence.
- **Scoring spans past the audio.** No clipping for any system in accuracy, F1 and
  recall-side tIoU: a predicted span beyond the end of the audio is an unmatched false
  positive, and a span overrunning it lowers tIoU through the union. Precision-side tIoU
  clips predictions to the audio first, so a span past the end has zero overlap.
- **Temporal overlap.** Recall-side tIoU (the original metric: matched IoU averaged over
  gold spans) can be raised by over-segmenting, since extra predicted spans cost nothing;
  a content-free 5 s Neutral grid scores 0.639 on it. Precision-side tIoU uses the same
  one-to-one matching, averaged over predicted spans, and tIoU-F is their harmonic mean.
  Added once for every system and reference, from saved raw outputs; the main tables
  report tIoU-F, the components table all three. The grid (5 s against a 5.8 s mean gold segment)
  still keeps tIoU-F 0.544, so its row is the timing-only floor for tIoU-F.
- **Batching.** Every model and task decodes batched only if a batch-1 vs batched
  spot-check on 30 items (40 sequences for SALMONN, 10 of them over 30 s) passes:
  no label change, or |delta| <= 0.01 on UA (utterance) / macro-F1 (sequence);
  otherwise batch 1.
- **Sequence parsing.** The sequence parser accepted timestamps in seconds or
  clock format (m:ss, h:mm:ss); all systems were re-scored from saved raw
  outputs with the same parser, and strict-format compliance rates are
  reported separately.

## QLoRA usage and rationale

- **qwen3_omni** is the only model flagged for QLoRA. It is a 30B-total MoE
  (~3B active/forward). bf16 LoRA optimizer/gradient state can exceed the 40 GB
  A100; training therefore falls back to 4-bit NF4 QLoRA (`--qlora`,
  double-quant, bf16 compute) when bf16 OOMs. Only the Thinker is loaded (the
  talker/code2wav TTS stack is not needed for SER). All other models train in
  bf16 LoRA.

## Model loading conventions

Native loaders follow the upstream model docs; custom-model loaders drive the
source vendored under `third_party/`.

| Model | class | processor | audio placeholder | audio kwarg | sr |
|---|---|---|---|---|---|
| qwen2_audio | `Qwen2AudioForConditionalGeneration` | `AutoProcessor` | `<\|audio_bos\|><\|AUDIO\|><\|audio_eos\|>` | `audio` | 16 kHz |
| qwen3_omni | `Qwen3OmniMoeThinkerForConditionalGeneration` | `Qwen3OmniMoeProcessor` | `<\|audio_start\|><\|audio_pad\|><\|audio_end\|>` | `audio` | 16 kHz |
| audio_flamingo3 | `AudioFlamingo3ForConditionalGeneration` | `AutoProcessor` | `<sound>` | `audio` | 16 kHz |
| salmonn | `SALMONN.from_config` (vendored) | `WhisperFeatureExtractor` | `<Speech><SpeechHere></Speech>` | n/a (spectrogram dict) | 16 kHz |
| mellow | `MellowWrapper` (vendored) | none (file paths) | none | n/a (2 file paths) | 32 kHz |

Qwen3-Omni's audio tokens differ from Qwen2-Audio's (confirmed against the
processor tokenizer) and the processor additionally requires `torchvision`.

## Custom model integration (salmonn, mellow)

The two custom models load from source vendored in this repo under
`third_party/{salmonn,mellow}` (override with `$SALMONN_ROOT` / `$MELLOW_ROOT`);
no external checkout is required. Component weights live under
`$MODEL_CACHE/{SALMONN-7B,Mellow}` (default `/media/volume/Model_Cache`). They use
bespoke forward/generate signatures, so `supports_generic_sft=False`: the generic
`scripts/finetune.py` / `scripts/eval_finetuned.py` refuse them and point to the
native recipe.

- **Mellow** loads fully in this environment (validated; 167M params) after
  `pip install --no-deps importlib_resources torchlibrosa`. Inference:
  `model.generate(examples=[[audio1, audio2, prompt]], max_len=128, top_p=0.8, temperature=1.0)`.
- **SALMONN** config + `WhisperFeatureExtractor` load here, and model init reaches
  the Vicuna tokenizer, but its vendored code targets `transformers==4.28.0`;
  under transformers 5.x the Vicuna SentencePiece tokenizer fails to convert. A
  full SALMONN load/train needs a separate env pinned to that version
  (`requirements-custom.txt` documents the extras). SALMONN trains via its own
  vendored `third_party/salmonn/train.py`, not this repo's trainer.

## Split isolation

Train / Development / Test1 come from the official MSP-Podcast 1.12
`Partitions.txt` and are never mixed. Sequence manifests are built per
partition (`scripts/E7/data/build_manifests.py`). Select on Development;
report Test1.
