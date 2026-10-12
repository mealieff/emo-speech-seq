# Methodology

How the systems in `tab:main-results` (`docs/MODELS.md`) are built, run and scored.
Run-level details (paths, commits, status) are in `docs/MODELS.md`; working notes are in
`docs/methods_notes.md`.

## Data

MSP-Podcast v1.12, official `Partitions.txt` splits. Train is used for training,
Development for model selection, Test1 for reporting. Test3 is not used: its IDs are
packed challenge IDs, not original unique IDs, so sequences cannot be rebuilt from it.
Eight emotion classes: Angry, Sad, Happy, Surprised, Fear, Disgust, Contempt, Neutral.
Segments labelled O (other) or X (no agreement) are excluded.

**Utterances.** One clip, one consensus label. Train 90,022, Development 25,258,
Test1 34,963.

**Sequences.** Runs of strictly consecutive unique IDs from the same podcast, within one
partition, concatenated back to back at 16 kHz with no inserted silence. An O or X
segment ends a run. Length L is 2 to 40 segments. Gold span boundaries are the
cumulative clip durations. Train 9,495, Development 2,557, Test1 3,648 sequences
(9,447 segments). Results are stratified by length bin:

| Bin | L = 2 | L = 3 | L = 4–6 | L ≥ 7 | All |
|---|---|---|---|---|---|
| Test1 sequences | 2,604 | 590 | 371 | 83 | 3,648 |

Manifests: `scripts/E7/data/build_manifests.py`; ALM JSONL in `data/processed/e7/sft/`.

## Tasks and prompts

**Utterance SER.** Prompt (`model/qwen2_audio/prompt_a.txt`, identical for every ALM):

> Listen to the speech and identify the primary emotion expressed by the speaker. Reply
> with exactly one word, chosen from this list and nothing else: Angry, Sad, Happy,
> Surprised, Fear, Disgust, Contempt, Neutral.

**Sequence SER.** Prompt (`scripts/prompts/sequence.txt`, identical for every ALM, frozen)
asks for a chronological list of `(start_time, end_time, emotion)` with times in seconds,
segment boundaries chosen by the model from the audio, labels from the same eight
classes. Each prompt holds one audio input (one clip, one sequence, or one 30 s window).

## Systems

| System | Base model | Settings in the table |
|---|---|---|
| Qwen2-Audio | `Qwen/Qwen2-Audio-7B-Instruct` | PT, utterance SFT, sequence SFT |
| Audio-Flamingo-3 | `nvidia/audio-flamingo-3-hf` | PT, utterance SFT, sequence SFT |
| SALMONN | SALMONN-7B: Whisper-large-v2 + BEATs → window-level Q-Former → Vicuna-7B (released LoRA r8); vendored in `third_party/salmonn` | PT; sequence SFT (pending) |
| Kimi-Audio | `moonshotai/Kimi-Audio-7B-Instruct`, text output only (audio detokenizer not loaded) | PT; sequence SFT (pending) |
| Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | not run: does not fit in bf16 on the 40 GB GPU |
| WavLM (IS 2025) | released Interspeech 2025 categorical SER checkpoint, seed 7 (not retrained) | utterance classifier on gold segments |
| WavLM-TimeAug | WavLM sequence model initialised from the IS 2025 utterance model | sequence model |

PT = pretrained, zero-shot. SFT = LoRA fine-tuning on the corresponding task.
SALMONN and Kimi-Audio run in a separate environment (transformers 4.44.1, torch 2.6,
peft 0.12); the others use the repo environment.

## Fine-tuning

All ALM fine-tuning is LoRA on the language model's attention projections
(`q_proj, k_proj, v_proj, o_proj`), r = 16, α = 32, dropout 0.05, with the audio encoder
frozen. AdamW, learning rate 2e-4 with linear decay and no warmup, batch 1 with 8-step
gradient accumulation, 3 epochs, bf16. One 40 GB A100.

**Utterance SFT (Qwen2-Audio, Audio-Flamingo-3).** Trained on Train utterances with the
utterance prompt; target is the label word. The multimodal projector is trained.
The adapter after 3 epochs (step 33,759) is used. Development (all 25,258): Qwen2-Audio
UA 0.380 / macro-F1 0.389, Audio-Flamingo-3 UA 0.396 / macro-F1 0.405.

**Sequence SFT (Qwen2-Audio, Audio-Flamingo-3).** `scripts/E3/finetune/train_lora.py`.
Target: one `(start, end, Emotion)` line per gold segment, times to two decimals; loss on
the assistant target only. Training loss is the per-example token-mean cross-entropy
times a length-bin weight (sklearn "balanced" over Train bins: 2 → 0.353, 3 → 1.460,
4–6 → 2.575, 7+ → 10.889). One checkpoint per epoch; the checkpoint with the lowest
unweighted cross-entropy on the full Development set is used (Qwen2-Audio epoch 2,
0.225; Audio-Flamingo-3 epoch 3, 0.281). Qwen2-Audio's encoder takes at most 30 s, so
its training and inference audio is truncated to the first 30 s (608 / 9,495 Train and
203 / 3,648 Test1 sequences are longer); it is not retrained. Audio-Flamingo-3 trains on
Train sequences up to 180 s (9,493 of 9,495; GPU memory); its Development and Test1 sets
are complete.

**Sequence SFT (SALMONN, Kimi-Audio)**, configured, not yet run
(`scripts/E5/train_bespoke_lora.py`). Same LoRA settings, optimiser, schedule, epochs,
bin-weighted loss and full-Development selection.
- SALMONN: the released r8 LoRA is merged; a new r16 LoRA is added and the connector
  (Q-Former, query tokens, projection, layer norms) is trained. Sequences are cut into
  the same non-overlapping 30 s windows used at inference; targets are clipped to each
  window and re-timed, pieces under 0.1 s are dropped, and each window carries
  bin weight / number of windows. fp16 autocast with gradient scaling.
- Kimi-Audio: LoRA on the 28-layer main stack; the audio adaptor (`vq_adaptor`) is
  trained. Whole sequences; loss on the text stream. bf16.

## Inference

- **Decoding.** Greedy for every model (SALMONN's stock 4-beam setting is overridden).
  Maximum new tokens: 16 (utterance), 800 (sequence).
- **Batching.** A model decodes batched only if a batch-1 vs batched comparison on 30
  Test1 items passes: no change in predicted labels, or a change of at most 0.01 in UA
  (utterance) or macro-F1 (sequence). Otherwise it decodes one item at a time. The
  SALMONN sequence check uses 40 sequences, 10 of them over 30 s, so it covers final
  windows shorter than 30 s. In SALMONN's batched path BEATs runs per clip on the
  unpadded audio, so its features match batch 1 exactly. Kimi-Audio has no batched path.
- **Audio over 30 s.** Pretrained models whose encoder takes at most 30 s (Qwen2-Audio PT,
  SALMONN PT) decode non-overlapping 30 s windows. Each window's spans are shifted by the
  window offset. Spans A and B are merged only if they are in consecutive windows, A
  ends and B starts at the shared window boundary (within 0.05 s), and they have the
  same label. This undoes the split that windowing creates; spans inside a window are
  never merged, so unwindowed and windowed outputs are scored the same way.
  Qwen2-Audio SFT sees the first 30 s, as trained. Audio-Flamingo-3 and Kimi-Audio take
  the whole sequence.
- **Clip length (SALMONN).** Each clip or window is zero-padded to at least 1 s and capped
  at 30 s, as in the released inference code.
- **Numerics.** Qwen2-Audio, Audio-Flamingo-3 and Kimi-Audio run in bf16. SALMONN runs
  Vicuna and the Q-Former under its stock fp16 autocast, with the BEATs encoder in fp32;
  under fp16 BEATs overflows to NaN and every output comes back empty.

## Parsing model outputs

**Sequence.** One parser for every system (`scripts/E3/e3/parse.py`), tried in order:
prompt-style tuples `(start, end, emotion)` in round or square brackets, JSON/Python
lists of dicts, brace formats, then prose that pairs time intervals with emotion words.
Timestamps may be seconds (`ss.ff`) or clock format (`m:ss[.ff]`, `h:mm:ss[.ff]`). Clock
fields after the first must be two digits and below 60; anything else fails rather than
being guessed. All systems are re-scored from their saved raw outputs with this parser.
Two parse rates are reported: the main rate, and a strict-format rate that accepts
seconds-only timestamps, as a measure of format compliance.

**Utterance.** Primary: the first canonical label in the output (`first_n`); if none,
the prediction falls back to Neutral and counts as a parse failure. A strict parse (the
whole output must be exactly one label) is reported alongside.

## Scoring

**Sequence (Seq. rows).** Per sequence, predicted and gold spans are matched one-to-one,
greedily by highest temporal IoU, ties broken by nearest midpoint. A matched pair with the
same label is a true positive; a matched pair with different labels is a false negative
for the gold label and a false positive for the predicted one; unmatched gold spans are
false negatives and unmatched predictions false positives. Counts are pooled over all
sequences in a bin, then:
- **Acc**: true positives / gold segments.
- **F1**: macro-F1 over the eight classes.
- **tIoU-R** (recall side; the original tIoU): mean matched IoU over every gold span;
  an unmatched gold span, or any span of an unparsed output, counts as 0.
- **tIoU-P** (precision side): predictions are first clipped to the audio, then matched
  to gold with the same one-to-one rule; mean matched IoU over every predicted span. An
  unmatched prediction, including one wholly past the end of the audio, counts as 0.
- **tIoU-F**: harmonic mean of the bin's pooled tIoU-R and tIoU-P. The main tables
  report tIoU-F; all three are in the tIoU components table.

tIoU-R alone only measures how well each gold span is covered, so it can be raised by
over-segmenting: extra predicted spans cost nothing. The content-free references show
this. Back-to-back 5 s Neutral spans over the audio score tIoU-R 0.639 on Test1, above
Qwen2-Audio PT (0.428) and close to Qwen2-Audio SFT (0.764). tIoU-P was added for that
reason, and applied once to every system and reference from the saved raw outputs;
tIoU-R, accuracy and F1 are unchanged. Clipping applies to tIoU-P only. For accuracy, F1
and tIoU-R, predicted spans are not clipped for any system: a span past the end of the
audio is an unmatched false positive, and a span that overruns it lowers tIoU-R through
the union. Unparsed outputs have no predicted spans, so they lower tIoU-R but do not
enter tIoU-P. Scorer: `e5.metrics.quartile_report`.

**Content-free references.** Two predictions that ignore the audio content, scored like
every system (`scripts/E5/reference_baselines.py`): back-to-back 5 s Neutral spans from 0 s
to the end of the audio (the last span is the remainder), and one Neutral span covering
the whole audio. They mark what timing alone, or the majority class alone, scores. The
precision side penalises over-segmentation, but not a segmentation at roughly gold
granularity: gold segments average 5.8 s, so the 5 s grid writes 1.35 spans per gold span,
none past the end, and keeps tIoU-F 0.544. No overlap metric that ignores labels can
remove that, so the grid row is the floor for tIoU-F, and temporal scores are read
relative to it.

**Utterance-level systems on sequences (Utt. rows).** Every sequence segment is one Test1
utterance, so its predicted label is that utterance's Test1 prediction, placed on the gold
boundaries and scored with the same sequence scorer (`scripts/E6/score_sequences.py`).
tIoU is 1 by construction and not reported. Segment gold labels equal the utterance gold
labels on all 9,447 segments.

**Utterance Test1 (34,963 clips).** WA (accuracy), UA (mean per-class recall) and
macro-F1, plus first_n and strict parse-failure rates.

## Supervised speech baselines

**WavLM (IS 2025).** The released seed-7 categorical classifier, run on Test1
utterances; Utt. rows as above.

**WavLM-TimeAug.** A WavLM sequence model initialised from the IS 2025 utterance model,
with a boundary branch and a segment-level emotion head, trained on Train sequences with
a length-bin-balanced sampler. It reads 10 s windows with 2 s overlap and predicts
boundaries, then an emotion for each resulting segment (encoder lr 1e-5, head lr 1e-4,
seed 7). Selected on Development by 0.5 × (macro-F1 + boundary AP); the best epoch is 3
(training stopped during epoch 7). Scored with the same sequence scorer as the ALMs.
Full design: "WavLM-TimeAug design" below.

## Reproducibility

Every Test1 output directory stores per-item predictions (JSON/JSONL) and a
`provenance.json` with the git commit, adapter, batch size and decoding settings.
Re-scored sequence results are in `<eval_dir>/rescored/`, with `diff.json` comparing them
against the original scoring. The file map is in `docs/MODELS.md`. Commit hashes below
are those of the current history (rewritten 2026-10-09 to drop commit trailers; the
old → new map is `runs/e7/logs/commit_rewrite_map.txt`).

### Hardware and software

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA A100-SXM4-40GB, driver 550.163.01 |
| Host | 32 vCPU, 115 GB RAM, Ubuntu 24.04.3 LTS, Linux 6.14.0-37 |
| Python | 3.12.3 (both environments) |
| Main environment (Qwen2-Audio, Audio-Flamingo-3, scoring) | `venv/`: torch 2.6.0+cu124 (CUDA 12.4, cuDNN 9.1), transformers 5.6.0.dev0 from git commit `d3c7a19176496fc1ea246aa9df4f678f4fceb1cf`, peft 0.18.1, accelerate 1.13.0, tokenizers 0.22.2, safetensors 0.7.0, huggingface_hub 1.9.0, datasets 4.8.4, torchaudio 2.6.0, numpy 2.4.4, scikit-learn 1.8.0, librosa 0.11.0, soundfile 0.13.1 |
| SALMONN / Kimi-Audio environment | `/media/volume/more-more-data/kimi-venv/`: torch 2.6.0+cu124, transformers 4.44.1, peft 0.12.0 |
| Kimi-Audio inference code | `MoonshotAI/Kimi-Audio` at commit `349251e1d8f4f98d58fda59246381faecd7392e0` |
| SALMONN code | vendored in `third_party/salmonn` from `bytedance/SALMONN`; local changes: BEATs in fp32 (`0407b2e`), generate fix (`43bfca6`), per-clip BEATs and per-row stopping for batches (`70d015b`) |
| WavLM baseline code | vendored from `Lab-MSP/MSP-Podcast_Challenge_IS2025` (`scripts/E6/baseline`) |

All runs set `HF_HUB_OFFLINE=1` and load models from the local caches below.

### Model checkpoints

| Model | Source | Revision / SHA-256 |
|---|---|---|
| Qwen2-Audio | `Qwen/Qwen2-Audio-7B-Instruct` | snapshot `0a095220c30b7b31434169c3086508ef3ea5bf0a` |
| Audio-Flamingo-3 | `nvidia/audio-flamingo-3-hf` | snapshot `5c47d2de83c3163162586561f3ddba75b4a1194e` |
| Kimi-Audio | `moonshotai/Kimi-Audio-7B-Instruct` | snapshot `9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b` |
| SALMONN-7B checkpoint | `tsinghua-ee/SALMONN-7B`, `salmonn_7b_v0.pth` | revision `ed583f2c565c1110b3bf5772ed42d6744f3cc443`; SHA-256 `2cb2782495b2e3f487222763a30b53b02f727d49059201cc5fa88a7a1fd9dff9` |
| SALMONN Whisper | `openai/whisper-large-v2` | revision `ae4642769ce2ad8fc292556ccea8e901f1530655` |
| SALMONN BEATs | `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` | SHA-256 `e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c` |
| SALMONN LLM | `lmsys/vicuna-7b-v1.5` | revision `3321f76e3f527bd14065daf69dad9344000a201d` |
| WavLM (IS 2025) | released categorical model, seed 7 (`cat_ser.zip`, backbone `microsoft/wavlm-large`) | SHA-256 of `cat_ser.zip` `63673f21c58adcc7589762e97052c98ec90e8f2060b67e1ff1dbcb517cb0fbc7`; unpacked at `/media/volume/more-more-data/e6/models/cat_ser/7/` |
| WavLM-TimeAug | `/media/volume/more-more-data/e7/checkpoints/seed7_v2/best.pt` (epoch 3) | SHA-256 `04963f11c8b38597b7c2ae77916561dbe177ad8691e84156a389a1c9a167946f` |

Model caches: `/media/volume/more-data` (Hugging Face models, Kimi-Audio code) and
`/media/volume/Model_Cache/SALMONN-7B` (SALMONN components).

### Data

| File | Rows | SHA-256 |
|---|---|---|
| `MSP-PODCAST-Publish-1.12/Partitions.txt` | — | `8fc1f885a5b6fa2f055654321b585d90376b58392f2d47356f418f252a834c8a` |
| `MSP-PODCAST-Publish-1.12/Labels/labels_consensus.csv` | — | `395f978cf52720d1d07c0dbda3d59b664660d2dde5d5da85dde269733968d1fc` |
| `data/processed/e7/sft/train.jsonl` | 9,495 | `6f31fbb8fa4ba009ca30f18f699e06eaac86d92f1e232843f338ba0a2eae4e43` |
| `data/processed/e7/sft/dev.jsonl` | 2,557 | `4cb2633be706c579707f46421c56cc24884f4f9e3baea74458d1cd723368f4bf` |
| `data/processed/e7/sft/test1.jsonl` | 3,648 | `95860a727f37ed56227ea7afa0caa5ff42fb67242700388e84f926aab1cfaba6` |
| `data/processed/msp_podcast_test1.jsonl` (utterances) | 34,963 | `10726f25df2d3bd2dc8ae51d0c5a7758fb0eb67d2366ffca8b577a5bdcd9a028` |

Corpus root: `/media/volume/data/MSP-PODCAST-Publish-1.12`. Sequence manifests
(`data/processed/e7/manifests/e7_{train,dev,test1}.jsonl`) come from
`scripts/E7/data/build_manifests.py`; concatenated WAVs and the ALM JSONL (including
`train_af3.jsonl`, the ≤ 180 s subset) from `scripts/E7/data/export_sft_jsonl.py`.

### Fine-tuning runs

Trainer settings shared by all four ALM adapters (from the saved `training_args.bin`):
seed 42 (no separate data seed), AdamW (`adamw_torch`, β = 0.9/0.999, ε = 1e-8), weight
decay 0, max gradient norm 1.0, linear schedule with 0 warmup steps, bf16, 0 dataloader
workers.

| Adapter | Task | Data | Steps | Gradient checkpointing | Selected | Adapter SHA-256 |
|---|---|---|---|---|---|---|
| `runs/task_a/qwen2_audio/` | utterance | Train utterances, max length 512 tokens | 33,759 | on | final (step 33,759) | `0f6acd78663202dd2ded7d3a6d672af291e212f763b743084e350731bbdd6ce2` |
| `runs/task_a/audio_flamingo3/` | utterance | Train utterances, max length 512 tokens | 33,759 | on | final (step 33,759) | `dc408bc37e1df740bfac597fd34e5b3668e97f83903ca9fab01dd86228a473a6` |
| `runs/e7/sft/qwen2_audio/checkpoint-2374` | sequence | `train.jsonl` (9,495) | 3,561 | off | epoch 2, full-Dev CE 0.2247 (epochs 1/2/3: 0.2476/0.2247/0.2421) | `667c8fdee7327adfb63bd7635b160e0bdc2cf9979e9478380f816a3b3479acd2` |
| `runs/e7/sft/audio_flamingo3/checkpoint-3561` | sequence | `train_af3.jsonl` (9,493) | 3,561 | on | epoch 3, full-Dev CE 0.2814 (epochs 1/2/3: 0.3435/0.3006/0.2814) | `59da1a8dfd37aa8a52bb67b574518ff15a132357becc8e533ba30903c7ec0748` |

Sequence SFT wall-clock: Qwen2-Audio 10,009 s (2.8 h), Audio-Flamingo-3 16,420 s
(4.6 h). Per-run configuration is saved next to each adapter (`ser_training.json` for
utterance, `e3_training.json`, `selection_full_dev.json` and `log_history.json` for
sequence). Sequence SFT ran from the working tree before `train_lora.py` was first
committed (`6f8258b`); full-Development selection and Test1 decoding ran from committed
code. WavLM-TimeAug: `scripts/E7/train.py --seed 7` (Python, NumPy and torch seeded),
log `/media/volume/more-more-data/e7/logs/train_seed7.log`.

### Test1 runs

Decoding is greedy, so outputs do not depend on a seed; batched and unbatched GPU kernels
can still differ slightly, which is what the batching spot-check measures.

| System | Task | Code commit | Batch | Spot-check (30 items) | Output |
|---|---|---|---|---|---|
| Qwen2-Audio SFT | sequence | `fb3c08f`¹ | 8 | passed | `runs/e7/eval/qwen2_audio/` |
| Audio-Flamingo-3 SFT | sequence | `fb3c08f`¹ | 8 | passed | `runs/e7/eval/audio_flamingo3/` |
| Qwen2-Audio PT | sequence | `76647a1` | 1 | — (30 s window path is unbatched) | `runs/e7/eval/pretrained/qwen2_audio/` |
| Audio-Flamingo-3 PT | sequence | `76647a1` | 8 | passed | `runs/e7/eval/pretrained/audio_flamingo3/` |
| Qwen2-Audio SFT | utterance | `fb3c08f`² | 16 | passed (0 label changes) | `runs/task_a/qwen2_audio/` |
| Audio-Flamingo-3 SFT | utterance | `fb3c08f`² | 1 | failed (2 label changes) | `runs/task_a/audio_flamingo3/` |
| Qwen2-Audio PT | utterance | `76647a1` | 16 | passed (1 label change; ΔUA 0, Δmacro-F1 0.0011) | `runs/pretrained/test1/utt/qwen2_audio/` |
| Audio-Flamingo-3 PT | utterance | `70d015b` | 16 | passed (0 label changes) | `runs/pretrained/test1/utt/audio_flamingo3/` |
| SALMONN PT | sequence, utterance | `70d015b` | set by spot-check (40 sequences / 30 clips) | pending | `runs/e7/eval/pretrained/salmonn/`, `runs/pretrained/test1/utt/salmonn/` |
| Kimi-Audio PT | sequence, utterance | `70d015b` | 1 (no batched path) | — | `runs/e7/eval/pretrained/kimi_audio/`, `runs/pretrained/test1/utt/kimi_audio/` |

¹ Two tracked files were modified at run time (`docs/MODELS.md` and
`eval_finetuned.py`); neither is used by sequence decoding.
² With the working-tree `eval_finetuned.py` that was later committed as `1ec76b4` (adds
the strict-parse metrics only).

Spot-check outputs: `runs/e7/eval/spotcheck/` (SFT) and
`runs/e7/eval/pretrained/spotcheck/` (PT). Re-scoring of every sequence run and the Utt.
rows ran at parser commit `f48712d`; windowed runs (Qwen2-Audio PT, SALMONN PT) were
re-scored at `4558013` with boundary-only seam merging. The pretrained chain's stage log is
`runs/e7/logs/official_pretrained.status`.

### Re-running

```bash
bash scripts/prepare_data.sh                         # manifests, concatenated audio, ALM JSONL
bash scripts/E5/official_post_train.sh               # full-Dev selection + Test1 for the sequence SFT adapters
bash scripts/E5/official_pretrained.sh               # pretrained Qwen2-Audio / Audio-Flamingo-3 (sequence + utterance)
bash scripts/E5/official_pretrained_resume.sh        # SALMONN, Kimi-Audio, rescoring, Utt. rows
python3 scripts/E5/post_train_tools.py rescore <eval_dir> ...
python3 scripts/E6/score_sequences.py --jsonl data/processed/e7/sft/test1.jsonl --out-dir runs/e7/eval/wavlm_is2025
bash scripts/run_sequence.sh wavlm eval --split test1   # WavLM-TimeAug
```

## WavLM-TimeAug design

Code: `scripts/E7/model.py` (architecture), `train.py`, `dataset.py`, `decode.py`,
`eval.py`. Values below are those of the reported checkpoint
(`/media/volume/more-more-data/e7/checkpoints/seed7_v2/best.pt`).

**Starting point.** The IS 2025 categorical baseline is a WavLM-large encoder (a
convolutional feature encoder followed by 24 transformer layers, giving 1,024-dimensional
frames at 50 Hz), then attentive statistics pooling (ASP), which collapses an utterance's
frames into one 2,048-dimensional vector (an attention-weighted mean and standard
deviation), then a small MLP that maps that vector to the eight classes. It takes one
utterance and returns one label. WavLM-TimeAug keeps the encoder and the pooling, drops
the MLP, and adds the two things a sequence model needs and the utterance model lacks: a
boundary branch that decides where one segment ends and the next begins, and a
sequence-level emotion head that labels each segment with knowledge of its neighbours.
It is trained from a copy of the released seed-7 weights; the released checkpoint is not
modified.

**Encoder and long audio.** The input is the concatenated sequence waveform, normalised
with the IS 2025 training mean and standard deviation. Audio of 10 s or less is encoded
in one pass. Longer audio is cut into 10 s windows with an 8 s hop (2 s overlap), the last
window aligned to the end of the audio. Each window is encoded independently, and from
each one the model keeps only frames at least 1 s from an internal window edge, so every
kept frame has at least 1 s of context on both sides except at the start and end of the
recording. The kept frames are joined into one 50 Hz frame sequence for the whole
recording. One encoder pass yields two streams. The last transformer layer, which the
IS 2025 fine-tuning made emotion-specific, feeds the emotion branch. A learned
softmax-weighted sum of all 25 hidden states (the convolutional embedding plus the 24
layers, as in SUPERB) feeds the boundary branch, so the boundary detector can draw on
lower layers, where acoustic change is more visible than emotion. At the reported
checkpoint the mix puts 0.04 of its weight on the embedding, 0.49 on the bottom 12 layers
and 0.47 on the top 12, with the largest single weight on the ninth layer.

**Boundary branch.** The mixed frames are average-pooled from 50 Hz to 10 Hz (five frames
per step), passed through a two-layer bidirectional LSTM (256 units per direction,
dropout 0.2), and a linear layer outputs one boundary logit per 100 ms. That layer's bias
starts at log(0.02 / 0.98), the prior probability of a boundary frame. At inference the
logits go through a sigmoid and Gaussian smoothing (σ of one frame, 0.1 s). Local maxima
strictly inside the audio with a score of at least θ become boundaries; then, while any
segment is shorter than 2 s, the boundary at the shortest gap is removed. The start and
end of the audio are always boundaries, so the predicted segments tile the recording
exactly: no gaps, no overlaps, nothing past the end. θ is tuned on Development (0.25 for
the reported checkpoint; see below).

**Emotion branch.** For each segment, the emotion-stream frames inside it are pooled by
ASP, initialised from the IS 2025 pooling weights, into a 2,048-dimensional vector. The
vectors of all segments in the sequence, in order, go through a second two-layer
bidirectional LSTM (256 units per direction, dropout 0.2), so each segment's
representation depends on the segments before and after it, and a linear layer gives
eight logits per segment. The predicted label is the most probable class. The segment
LSTM and the output layer are new and randomly initialised. At inference the segments
come from the boundary branch; run on gold boundaries instead, the same branch gives the
oracle-boundary variant, which is not in the main tables.

**What is trained.** The convolutional feature encoder and the bottom 12 transformer
layers are frozen (155 M parameters). Trained: the top 12 layers together with WavLM's
feature projection, positional convolution and final layer norm (160 M), the pooling
(1.0 M), the boundary branch (4.2 M), the segment LSTM and output layer (6.3 M) and the
25 layer-mix weights. The model has 327 M parameters in all.

**Training objective.** The loss is an emotion term plus a weighted boundary term. For
emotion, each gold segment's target is its annotator vote distribution, reweighted by the
inverse frequency of each class among Train votes and renormalised; the loss is the KL
divergence from that target to the predicted distribution, averaged over segments. For
boundaries, the target is a 10 Hz heatmap with a Gaussian bump at every internal gold
boundary (the start and end of the sequence are not targets) and the nearest frame set to
exactly 1. The bump width σ starts at 0.25 s and shrinks linearly towards 0.15 s over the
20 planned epochs (0.239 s at epoch 3). The boundary loss is CenterNet's penalty-reduced
focal loss (α = 2, β = 4), normalised by the number of boundary frames. Its weight λ_b
is fixed once at the start of training so the two terms start at equal size: the ratio
of mean KL to mean focal loss on 8 Train sequences, clipped to [0.1, 20], which gave
0.402. A first version with plain binary cross-entropy on the heatmap collapsed to
predicting the base rate everywhere, and no boundary ever crossed 0.5; the focal loss,
the wider target, the prior-initialised bias and the tuned θ are the fixes.

**Augmentation and sampling.** Two augmentations act on time. A Train sequence longer than
100 s (22 of 9,495) is replaced, each time it is drawn, by a run of whole consecutive
segments totalling at most 100 s, chosen uniformly among all such runs and re-timed from
zero. And with probability 0.5, each internal gold boundary used to pool the emotion
branch is shifted by a uniform random offset of up to ±0.3 s (kept at least 0.05 s from
its neighbours and inside the sequence), so the emotion head does not rely on exact
boundaries it will not get at test time; the boundary targets stay at the true positions.
The code also has a scheduled-sampling mode, from epoch 11, in which half the sequences
are pooled on the model's own predicted boundaries with soft targets from the overlapping
gold segments; training stopped before epoch 11, so the reported model never used it.
Each step sees one sequence. The sampler cycles through the four length bins in turn,
drawing from each bin's shuffled list and reshuffling it when exhausted, so all bins are
seen equally often even though L = 2 is 71% of Train. WavLM-large's stock training-time
regularisation is left on: time masking of the feature sequence (probability 0.075, spans
of 10 frames), LayerDrop 0.1 and dropout 0.1.

**Optimisation and selection.** Adam with two learning rates (1e-5 for WavLM, 1e-4 for
everything else), no weight decay, constant learning rate, gradients accumulated over 8
sequences and clipped at norm 5, fp16 autocast with gradient scaling, seed 7. An epoch is
9,495 draws. After each epoch the model is run on the full Development set with the
current θ and scored two ways: macro-F1 from the sequence scorer used in the tables, and
boundary average precision. For AP, the smoothed local maxima are ranked by score, each
matched to at most one unmatched gold internal boundary within ±0.5 s, and missed gold
boundaries count as zero; no threshold is involved. θ is then re-tuned on the same
Development outputs by grid search over 0.01–0.30 in steps of 0.01, maximising mean
boundary F1 at ±0.5 s, and saved with the checkpoint. The checkpoint with the highest
0.5 × (macro-F1 + boundary AP) is kept, with early stopping planned after 5 epochs
without improvement. The Development combined score for epochs 1–6 was 0.482, 0.507,
0.508, 0.496, 0.490 and 0.480. Epoch 3 is the reported model (macro-F1 0.257, boundary AP
0.759, θ 0.25). The run was restarted once from its best checkpoint and stopped during
epoch 7, before early stopping triggered.

**Test1 inference.** Whole sequences with no cropping, the same 10 s windowing, θ = 0.25
from the checkpoint, one sequence at a time (`scripts/E7/eval.py`). The predicted segments
are scored by the shared sequence scorer. Test1 boundary AP is 0.656.