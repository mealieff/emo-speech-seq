# emo-speech-seq

**Benchmarking Sequential Speech Emotion Recognition in Multi-turn Conversations with Audio Language Models**

Melissa Lieffers, Shuju Shi, Charlotte Zhu, Thai Le (Indiana University Bloomington)

[Paper (PDF)](https://lethaiq.github.io/tql3/pdfs/emo_span.pdf) · [arXiv](#) *(link coming soon)*

## Overview

Speech emotion recognition (SER) benchmarks usually score isolated clips. But in a
real conversation, emotion emerges, persists, and changes. We ask whether audio-language
models (ALMs) can recover **what** emotion is expressed and **when** it occurs, from
unsegmented conversational audio.

Given a recording of consecutive podcast turns concatenated back to back (no boundaries
marked), the model must output an ordered list of timed emotion spans:

```
(start_time, end_time, emotion)
```

over eight classes: Angry, Sad, Happy, Surprised, Fear, Disgust, Contempt, Neutral.

We build this task from MSP-Podcast v1.12 (9,495 train / 2,557 dev / 3,648 Test1
sequences, 2 to 28 segments each) and compare pretrained and LoRA fine-tuned ALMs
(Qwen2-Audio, Audio-Flamingo-3, Qwen3-Omni, Kimi-Audio) against a supervised non-ALM
baseline (WavLM, extended as WavLM-TimeAug). Recognition is scored with accuracy and
macro-F1, and localization with temporal IoU (tIoU).

## What the paper shows

- **Pretrained ALMs do poorly on sequential SER.** Zero-shot macro-F1 ranges from 3.3 to
  21.5, and tIoU from 2.4 to 40.2.
- **LoRA fine-tuning helps substantially.** For example, Audio-Flamingo-3 goes from 3.3 to
  32.1 macro-F1 and from 2.4 to 72.1 tIoU.
- **Fine-tuned ALMs can beat a dedicated sequence model.** Qwen3-Omni, Audio-Flamingo-3,
  and Kimi-Audio all beat WavLM-TimeAug on overall macro-F1 and tIoU, though the margin
  varies by model and metric.
- **Localization degrades on longer sequences.** From 2-segment to 7+ segment recordings,
  tIoU drops from 81.5 to 50.9 (Audio-Flamingo-3), 85.0 to 63.0 (Qwen3-Omni), and
  82.9 to 50.4 (Kimi-Audio).
- **Span-level accuracy overstates tracking ability.** A model that labels a single span
  correctly over half the time recovers only about 13% of three-span emotion contours.
- **Errors collapse onto frequent classes.** Neutral, Happy, and Angry account for
  87–96% of mislabeled predictions, and fine-tuned models never recognize Fear.
- **Models follow the main emotional trajectory but not the nuance.** Predicted VAD
  contours correlate only moderately with human labels (0.41–0.47 for valence at
  L=4–6), and transitions involving Sad, Surprised, and Contempt appear almost only in
  human annotations.

### Overall results (Test1, all lengths)

| Model | Setting | Acc | Macro-F1 | tIoU<sub>F</sub> |
|---|---|---|---|---|
| Qwen2-Audio | PT / SFT | 28.3 / 54.9 | 12.5 / 23.6 | 39.9 / 54.3 |
| Audio-Flamingo-3 | PT / SFT | 41.1 / 54.9 | 3.3 / **32.1** | 2.4 / 72.1 |
| Qwen3-Omni | PT / SFT | 38.1 / **56.5** | 21.5 / 30.7 | 40.2 / **78.0** |
| Kimi-Audio | PT / SFT | 29.7 / 55.6 | 8.5 / 31.0 | 16.3 / 73.7 |
| WavLM-TimeAug | Seq. | 45.9 | 27.7 | 66.4 |

PT = pretrained, SFT = LoRA fine-tuned. Qwen2-Audio SFT only hears the first 30 s of
each sequence (see the paper's Limitations).

Takeaway: evaluating ALMs on isolated clips is not enough. Benchmarks should jointly
assess what emotion is expressed and when, over variable-length conversational audio.

## Setup

```bash
cp configs/paths.env.example configs/paths.env   # set MSP_ROOT and MODEL_CACHE
bash scripts/setup/install.sh
python scripts/setup/download_models.py
```

SALMONN and Kimi-Audio use a second environment:

```bash
bash scripts/setup/install_legacy.sh
python scripts/setup/download_models.py --models salmonn kimi_audio
```

## Data

```bash
bash scripts/data/prepare_all.sh
```

Writes 8-class utterance JSONL, unique-ID sequence manifests, concatenated 16 kHz
WAVs, and ALM JSONL (`data/sequences/{train,dev,test1}.jsonl`).

## Train

```bash
python scripts/finetune/utterance_lora.py --model qwen2_audio
python scripts/finetune/utterance_lora.py --model audio_flamingo3
bash scripts/finetune/run_sequence_sft.sh qwen2_audio
bash scripts/finetune/run_sequence_sft.sh audio_flamingo3
python scripts/finetune/wavlm_timeaug.py --seed 7
```

Sequence SFT trains 3 epochs, then selects the lowest unweighted CE on full
Development. WavLM (IS 2025) seed 7 is not retrained.

Optional (legacy env): `bash scripts/finetune/run_bespoke_sft.sh`

## Test1

```bash
bash scripts/inference/run_test1.sh all
python scripts/evaluate/rescore.py runs/test1/sequence/*
```

`run_test1.sh` also writes Utt. rows (`runs/test1/utt_on_sequences/`) and the
content-free references. Spot-check: batched decode only if batch 1 and batch N
agree (see METHODOLOGY).

## Citation

```bibtex
@misc{lieffers2026sequentialser,
  title  = {Benchmarking Sequential Speech Emotion Recognition in Multi-turn Conversations with Audio Language Models},
  author = {Lieffers, Melissa and Shi, Shuju and Zhu, Charlotte and Le, Thai},
  year   = {2026},
  note   = {arXiv link coming soon}
}
```

## Ethics and license

MSP-Podcast is used under its academic research license. This repo releases no audio or
speaker-identifying information. Sequential emotion tracking can enable surveillance and
unreliable inferences, and the results here show current systems are not reliable enough
for workplace, classroom, or customer-service deployment.