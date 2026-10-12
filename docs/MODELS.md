## Saved Models

All models use the original Train / Development / Test1 splits.
X and O are excluded. Eval JSON is required for error analysis.

### Single-segment SER (no timestamping)

Same 8-class label set as the sequence prompt, one emotion per clip.

| Model | Status | Saved at |
|---|---|---|
| WavLM Interspeech 2025 | ready (do not retrain seed-7) | `/media/volume/more-more-data/e6/models/cat_ser/7` (categorical); `dim_ser/7` (attributes) |
| AF3 | adapter trained; Test1 eval queued in tmux `emo-post` | `runs/task_a/audio_flamingo3/` (`eval_msp_podcast_test1.json`) |
| Qwen2 | adapter trained (Dev macro-F1 0.389); Test1 eval queued in tmux `emo-post` | `runs/task_a/qwen2_audio/` (`eval_msp_podcast_test1.json`) |
| Qwen3-Omni | needs to be run | — |
| Kimi-Audio | needs to be run (`bash scripts/setup.sh --kimi`) | — |
| SALMONN | needs to be run | — |

Utterance ALM scoring: `parse_mode=first_n` is primary (first canonical label in
the output), matching the Dev numbers in `eval_dev.json`. The adapters repeat
the label with no separator or EOS (`NeutralNeutral…`), so a strict whole-output
parse fails on most outputs; strict-parse metrics and `strict_parse_failure_rate`
(0.933 on Dev) are reported alongside. `scripts/eval_finetuned.py` decodes
greedily, `max_new_tokens` 16, in left-padded batches (`--batch-size`, default
16; `EVAL_BATCH_SIZE`).

```bash
bash scripts/run_utterance.sh wavlm eval --split test1
bash scripts/run_utterance.sh qwen2 train
```

### Sequence SER

Concatenated back-to-back unique-ID segments (length 2, 3, … ≤ 40) on Train /
Development / Test1. Scored by quartile `{2, 3, 4-6, 7+}`. Test3 is omitted
(packed challenge IDs, not original unique IDs).

| Model | Status | Saved at |
|---|---|---|
| WavLM-TimeAug | stopped mid-epoch 7 (best still epoch 3) | `/media/volume/more-more-data/e7/checkpoints/seed7_v2/` |
| AF3 | training (3 epochs, Dev 256, grad-ckpt on; started 17:16 UTC, ETA ~21:30 UTC); full-Dev selection + Test1 eval queued in tmux `emo-post` | `runs/e7/sft/audio_flamingo3/` |
| Qwen2 | trained (3 epochs; Dev-256 best epoch 2); full-Dev selection + Test1 eval queued in tmux `emo-post` | `runs/e7/sft/qwen2_audio/` |
| Qwen3-Omni | needs to be run | — |
| Kimi-Audio | needs to be run | — |
| SALMONN | needs to be run | — |

The earlier 20-epoch Qwen2 run (epochs 1–2 done; Dev CE 0.262 → 0.251) is
archived at `runs/e7/sft/qwen2_audio_20ep_partial/` and is not used.

WavLM-TimeAug uses the E7 timestamp encoder intercalated with the IS2025 WavLM
utterance init (10 s / 2 s windows). Sequence ALMs use the prompt below
(open-span list).

```bash
bash scripts/prepare_data.sh
bash scripts/run_sequence.sh wavlm eval --split test1
# AF3 + Qwen2 on official splits (waits until WavLM-Seq releases the GPU):
bash scripts/E5/queue_official_sft.sh
# Training chain (Qwen2 3ep -> AF3 3ep; 2 if >= 9 s/step at step 50):
tmux new-session -d -s emo-official "bash runs/e7/official_chain.sh 2>&1 | tee -a runs/e7/logs/official_chain.log"
# Post-training chain: waits for AF3 training, then full-Dev re-selection of every
# epoch checkpoint -> GPU batch-1 vs batched spot-check (30 Test1 examples per model
# and task) -> Test1 sequence eval (max_new_tokens 800) -> Test1 utterance eval:
tmux new-session -d -s emo-post "bash scripts/E5/official_post_train.sh 2>&1 | tee -a runs/e7/logs/official_post.log"
cat runs/e7/logs/official_post.status
```

Each Test1 output dir gets a `provenance.json` with the git commit, adapter,
batch size and `max_new_tokens`. Full-Dev selection is written to
`runs/e7/sft/<model>/selection_full_dev.json` (per-checkpoint full-Dev CE, plus
whether it agrees with the Dev-256 choice); spot-check reports go to
`runs/e7/eval/spotcheck/`.

Long jobs must run detached (tmux / `setsid nohup`). Jobs started from an
editor-agent shell die when that shell is torn down (Qwen2 died twice this way).

Do not reuse conversation-split adapters under `runs/e5/`.

Eval JSON:

- WavLM-Seq: `runs/e7/report_test1_pred.json`, `runs/e7/preds_test1_pred.jsonl`
- ALMs: `runs/e7/eval/<model>/metrics.json`, `predictions.jsonl`

Prompt:

```
Listen to the entire audio sequence and identify the emotion expressed by the speaker(s) over time.

The sequence may contain multiple speakers and multiple emotional states. Track how emotion changes throughout the audio rather than assigning a single emotion to the entire recording. Use the audio itself to determine the segment boundaries. 

Segments comprise utterances at the phrasal level. For each distinct segment, provide:

* the start time in seconds 
* the end time in seconds
* the emotion

Use only these emotion labels:
Angry, Sad, Happy, Surprised, Fear, Disgust, Contempt, Neutral.

Return the result as a chronological list of:

(start_time, end_time, emotion)
```

Frozen copy: [`scripts/prompts/sequence.txt`](../scripts/prompts/sequence.txt).



## LaTeX summary tables

Copy-paste blocks. Numbers are official Train / Development / Test1, 8-class
(drop X/O), unique-ID concat, L≤40. Test3 is omitted. Fill Test1 ALM cells after
`runs/e7/eval/<model>/metrics.json` exists. Do not paste conversation-split
`runs/e5/` scores here.

### Sequence corpus (Table A1)

```latex
\begin{table}[t]
\centering
\caption{Unique-ID concat sequences (strict consecutive, drop X/O, $L\in[2,40]$).
Bins $\{2\}$, $\{3\}$, $\{4$--$6\}$, $\{7+\}$. Test3 omitted (packed challenge IDs).}
\label{tab:seq-corpus}
\begin{tabular}{lrrrrrrr}
\toprule
Split & $\{2\}$ & $\{3\}$ & $\{4$--$6\}$ & $\{7+\}$ & $L{\ge}2$ & Segs. & Max $L$ \\
\midrule
Train & 6{,}729 & 1{,}626 & 922 & 218 & 9{,}495 & 24{,}492 & 23 \\
Development & 1{,}810 & 428 & 267 & 52 & 2{,}557 & 6{,}568 & 16 \\
Test1 & 2{,}604 & 590 & 371 & 83 & 3{,}648 & 9{,}447 & 28 \\
\bottomrule
\end{tabular}
\end{table}
```

22 Train / 9 Test1 sequences exceed 100\,s (max 206.6\,s / 174.3\,s). Dev max is 83.8\,s. AF3 train drops 2 sequences $>$180\,s, both in $\{7+\}$ $\rightarrow$ 9{,}493 (218 $\to$ 216); AF3 Dev/Test1 unchanged.

Verified 2026-10-08: `build_manifests.py --max-skip 0 --max-len 40` rebuilt to a temp dir reproduces these counts exactly, and the `seq_id` sets equal `data/processed/e7/sft/{train,dev,test1}.jsonl` that the live Qwen2 run loaded (`train n=9495`, `eval n=2557` in its log).

### Sequence ALM SFT

```latex
\begin{table}[t]
\centering
\caption{Sequence ALM LoRA SFT on official-split concat JSONL
(\texttt{scripts/E5/sft/train\_vibevoice\_style.py} $\to$ \texttt{train\_lora.py}).
Prompt: frozen open-span list in the user turn.}
\label{tab:alm-sft}
\begin{tabular}{ll}
\toprule
Item & Setting \\
\midrule
Base models & Qwen2-Audio-7B-Instruct; Audio Flamingo 3 \\
Audio & 16\,kHz PCM\_16 concat, no inserted silence \\
Gold times & cumulative clip durations \\
Labels & Angry, Sad, Happy, Surprised, Fear, Disgust, Contempt, Neutral \\
Prompt layout & user (instruction in user turn) \\
LoRA & $r{=}16$, $\alpha{=}32$, dropout $0.05$; $q,k,v,o\_proj$ \\
Frozen / saved & audio tower frozen; \texttt{multi\_modal\_projector} saved \\
Trainable (Qwen2) & 25.96M / 8.42B (0.31\%) \\
Optim. & AdamW, lr $2{\times}10^{-4}$, linear decay, no warmup \\
Precision & bf16; grad.\ ckpt off for Qwen2 (peak 28.6\,GB), on for AF3 (off OOMs at 40\,GB) \\
Batch & 1 $\times$ grad-accum 8 (effective 8) \\
Epochs / steps & 3; 1{,}187 opt.\ steps/epoch; 3{,}561 total \\
Train CE & length-bin-weighted (sklearn balanced); even-span off \\
Eval & first 256 Development sequences every epoch \\
Selection & lowest full-Dev ($n{=}2557$) unweighted token CE over epoch checkpoints \\
Checkpoints & save every epoch, keep all \\
AF3 audio cap & 180\,s train / 496\,s infer \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\centering
\caption{Length-bin CE weights (sklearn \texttt{balanced} on Train $n{=}9495$).}
\label{tab:bin-weights}
\begin{tabular}{lrr}
\toprule
Bin & $n$ & Weight \\
\midrule
$\{2\}$ & 6{,}729 & 0.353 \\
$\{3\}$ & 1{,}626 & 1.460 \\
$\{4$--$6\}$ & 922 & 2.575 \\
$\{7+\}$ & 218 & 10.889 \\
\bottomrule
\end{tabular}
\end{table}
```

Qwen2 (2026-10-08, 14:28–17:15 UTC, 2.8 s/step): Dev-256 CE by epoch 0.368 / **0.346** / 0.363; best = epoch 2 (`checkpoint-2374`), saved as the final adapter. Dev-256 CE is not comparable to the archived 20-epoch run's full-Dev CE (0.262 / 0.251). AF3 started 17:16 UTC; 4.84 s/step at step 59 (under the 9 s/step cut-off, so 3 epochs kept).

Training CE is length-bin-weighted (verified in both 3-epoch run logs: `bin_weight train CE on`, weights as in Table `tab:bin-weights`; AF3 {7+} weight 10.987 on its 216-sequence bin). Dev `eval_loss` is unweighted token CE. Model selection is redone on the full Development set: every epoch checkpoint of both models is scored with that same unweighted Trainer `eval_loss` on all 2,557 Dev sequences (`train_lora.py --eval-only-checkpoint`), and the lowest is used for Test1. The Dev-256 losses above were on the first 256 Dev rows, which is not a random sample.

Test1 decoding (`scripts/E5/run_inference.py`): greedy, `max_new_tokens` 800 (20 tokens per output line, so complete outputs up to $L{=}40$; no Test1 sequence is truncated). Left-padded batches of 8 grouped by audio duration (`--batch-size`, `SEQ_EVAL_BATCH_SIZE`). Batched text matched batch-1 text on CPU; the post-training chain repeats the check on GPU in bf16 (30 random Test1 examples per model) and falls back to batch 1 if pooled macro-F1 moves by more than 0.01. Generation is not stopped at the end of the audio, since that would change the scoring. Unbatched, the archived epoch-2 adapter took ~4.4 s/example on Test1 (runaway outputs averaging 5.9 lines vs 2.9 gold).

### WavLM-Seq (paper E6.3)

```latex
\begin{table}[t]
\centering
\caption{WavLM-Seq (E6.3) on the same concat as the ALMs. Init: released IS2025
\texttt{cat\_ser} seed-7. Do not overwrite those utterance weights.}
\label{tab:wavlm-seq}
\begin{tabular}{ll}
\toprule
Item & Setting \\
\midrule
Encoder & WavLM-large; freeze CNN + bottom 12; finetune top 12 \\
Windows & 10\,s / 2\,s overlap, 50\,Hz frames \\
Layer mix & SUPERB softmax mix of all layers (boundary branch) \\
Timestamp head & BiLSTM $2{\times}256$; heatmap focal; $\sigma{=}0.25$s $\to$ $0.15$s \\
Boundary bias & $\log(\pi/(1{-}\pi))$, $\pi{=}0.02$; $\lambda_b{\approx}0.402$ \\
Decode $\theta$ & swept on Development; F1 at ${\pm}0.5$s \\
Emotion head & ASP (from \texttt{cat\_ser}) + segment BiLSTM $2{\times}256$ $\to$ 8 logits \\
Train crop & whole-segment window ${\le}100$s; bin-balanced sampling \\
Infer & full concat, sliding windows \\
Selection & $0.5\times($Dev macro-F1 $+$ boundary AP$)$ \\
Seed & 7 (13, 42 later) \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\begin{table}[t]
\centering
\caption{WavLM-Seq Development (seed-7 v2). Best checkpoint is epoch 3
(\texttt{best.pt}). Run stopped mid-epoch 7; AP falling after epoch 2.}
\label{tab:wavlm-seq-dev}
\begin{tabular}{crrrrc}
\toprule
Epoch & Macro-F1 & Bdry F1 & AP & Combined & $\theta$ \\
\midrule
1 & 0.209 & 0.258 & 0.756 & 0.482 & 0.22 \\
2 & 0.253 & 0.543 & 0.762 & 0.507 & 0.24 \\
\textbf{3} & \textbf{0.257} & \textbf{0.552} & 0.759 & \textbf{0.508} & 0.25 \\
4 & 0.249 & 0.498 & 0.744 & 0.496 & 0.28 \\
5 & 0.246 & 0.541 & 0.735 & 0.490 & 0.23 \\
6 & 0.252 & 0.520 & 0.708 & 0.480 & 0.24 \\
\bottomrule
\end{tabular}
\end{table}
```

Epoch-3 AP by bin: $\{2\}$ 0.768, $\{3\}$ 0.736, $\{4$--$6\}$ 0.742, $\{7+\}$ 0.752.

### Utterance SER (E6.1)

Released IS2025 seed-7; do not retrain in place. Dev is 8-class ($n{=}25{,}258$).
Test3 F1 is the challenge paper number (hidden labels).

```latex
\begin{table}[t]
\centering
\caption{Utterance WavLM (IS2025 seed-7). Development is local 8-class consensus;
Test3 categorical F1 is the public challenge table.}
\label{tab:utt-wavlm}
\begin{tabular}{llcc}
\toprule
Task & Metric & Development & Test3 \\
\midrule
Categorical & F1-macro / F1-micro & 0.370 / 0.538 & 0.329 / 0.355 \\
Attributes & CCC A / V / D / avg & 0.693 / 0.734 / 0.625 / 0.684 & 0.623 / 0.638 / 0.477 / 0.579 \\
\bottomrule
\end{tabular}
\end{table}
```

Dev per-class F1: Angry 0.689, Happy 0.678, Sad 0.495, Neutral 0.466, Surprise 0.233, Disgust 0.167, Contempt 0.153, Fear 0.076.

### Test1 sequence results (fill after eval)

Primary scorer: `e5.metrics.quartile_report`. Columns match `{n, acc, P, R, F1, tIoU, parse}`.

Parser: one parser for every system (`scripts/E3/e3/parse.py`, timestamp grammar in
`e3/timestamps.py`): seconds `ss.ff`, `m:ss[.ff]`, `h:mm:ss[.ff]`; clock fields after the
first must be two digits and < 60, anything else fails. Table numbers come from
`<eval_dir>/rescored/` (`post_train_tools.py rescore`, re-parsed from saved `pred_text`;
`diff.json` holds old-vs-new cells). `parse_rate_strict` = parse rate with seconds-only
timestamps (format compliance), reported per model alongside the table.

Re-scoring at parser commit `f48712d` left every existing ALM row unchanged (0 predictions
changed; seconds-only parse flag equal to the old parse flag on every row).

Windowed runs (Qwen2-Audio PT, SALMONN PT, and the Qwen2 30 s-window comparison) were
re-scored again at commit `4558013`. The stitcher had merged every touching same-label
pair, including neighbours inside one window. It now merges spans A and B only if they are
in consecutive windows, A ends and B starts at the shared window boundary (within 0.05 s),
and they have the same label; nothing inside a window is merged. Under this rule neither
Qwen2 run has any seam merge (0 in 203 multi-window sequences). Qwen2-Audio PT: 403 / 3,648
sequences changed, All-bin Acc 0.271 -> 0.283, F1 0.123 -> 0.125, tIoU 0.424 -> 0.428.

Temporal overlap (commit `09702c7`): recall-side tIoU (the original metric) is kept;
precision-side tIoU (predictions clipped to the audio, same one-to-one matching, averaged
over predicted spans) and tIoU-F (harmonic mean) were added for every system from saved
raw outputs, after the content-free 5 s Neutral grid scored 0.639 on the recall side.
Accuracy, F1 and recall-side tIoU are unchanged on every row. Main tables: tIoU-F; all three
in `tab:tiou-components`. References: `runs/e7/eval/reference/{grid5_neutral,single_neutral}/`
(`scripts/E5/reference_baselines.py`); WavLM-TimeAug re-scored into
`runs/e7/eval/wavlm_timeaug/`.

| System | parse rate (main) | strict-format parse rate |
|---|---|---|
| Qwen2-Audio PT | 0.998 | 0.998 |
| Qwen2-Audio SFT | 1.000 | 1.000 |
| Audio-Flamingo-3 PT | 0.972 | 0.972 |
| Audio-Flamingo-3 SFT | 1.000 | 1.000 |
| SALMONN PT | 0.999 | 0.002 (writes `(m:ss-m:ss) Label`) |
| Kimi-Audio PT | pending | pending |

Utt. rows (second table): each sequence segment is one Test1 utterance, so the segment
label is the utterance model's Test1 prediction (`pred_for_metrics`: first_n, Neutral
fallback) on gold boundaries, scored with the same quartile scorer
(`scripts/E6/score_sequences.py`). Segment gold = utterance gold on all 9,447 segments.
Utterance inference is the utterance prompt with one clip per prompt on Test1. Batching
rule (all models, both tasks): decode batched if the batch-1 vs batched spot-check on 30
items passes (no label change, or |delta| <= 0.01 on UA / macro-F1), otherwise batch 1.

Result files (all JSON / JSONL, per-item records kept for error analysis):

| Row | Per-item predictions | Table numbers |
|---|---|---|
| ALM Seq. SFT | `runs/e7/eval/{qwen2_audio,audio_flamingo3}/predictions.jsonl` | `.../rescored/quartile_report.json` |
| Qwen2 Seq. SFT, 30 s chunks | `runs/e7/eval/qwen2_audio_chunk30/predictions.jsonl` | `.../rescored/quartile_report.json` |
| ALM Seq. PT | `runs/e7/eval/pretrained/{model}/predictions.jsonl` | `.../rescored/quartile_report.json` |
| ALM utterance Test1 (34,963) SFT | `runs/task_a/{model}/predictions_msp_podcast_test1.json` | `eval_msp_podcast_test1.json` |
| ALM utterance Test1 PT | `runs/pretrained/test1/utt/{model}/predictions_msp_podcast_test1.json` | `eval_msp_podcast_test1.json` |
| ALM Utt. on sequences | `runs/e7/eval/utt_oracle/{model}_{pt,sft}/predictions.jsonl` | `quartile_report.json` |
| WavLM (IS 2025) utterance | `runs/e6/cat_test1_preds.json` (+ `.csv`) | `runs/e6/cat_test1.json` |
| WavLM (IS 2025) Utt. on sequences | `runs/e7/eval/wavlm_is2025/predictions.jsonl` | `quartile_report.json` |
| WavLM-TimeAug Seq. | `runs/e7/preds_test1_pred.jsonl` | `runs/e7/eval/wavlm_timeaug/quartile_report.json` (original: `runs/e7/report_test1_pred.json`) |
| Content-free references | `runs/e7/eval/reference/{grid5_neutral,single_neutral}/predictions.jsonl` | `quartile_report.json` |

Each eval dir also has `provenance.json` (git commit, settings); `rescored/diff.json` holds
old-vs-new parser cells.

% Preamble: \usepackage{booktabs,xcolor}
\newcommand{\tbd}{\textcolor{black!35}{\rule[0.5ex]{1.2em}{0.6pt}}}  % pending result
\newcommand{\na}{\textcolor{black!50}{--}}                            % not applicable

\begin{table*}[t]
\centering
\caption{Performance on MSP-Podcast v1.12 Test1 ($n=3{,}648$ sequences),
stratified by sequence length $L$. PT denotes the pretrained model and SFT
denotes sequence-aware LoRA fine-tuning. WavLM (Interspeech 2025) is an
utterance-level classifier; WavLM-TimeAug is a separately trained
time-augmented sequence model. Accuracy and F1 measure emotion
classification. tIoU$_F$ measures temporal overlap: it is the harmonic mean of
recall-side tIoU (each gold span's IoU with its one-to-one matched prediction,
averaged over gold spans) and precision-side tIoU (the same matching on predictions
clipped to the audio, averaged over predicted spans; unmatched spans and spans past
the end count as 0), so over-segmentation is penalised. Both components are in
Table~\ref{tab:tiou-components}. Content-free references give the scores of a
back-to-back 5\,s Neutral grid and of one Neutral span covering the audio. Gold
segments average 5.8\,s, so the grid is close to gold granularity and marks what timing
alone earns: read tIoU$_F$ relative to the grid row, not to zero. Audio over 30 s: Qwen2-Audio SFT sees the first
30 s (as trained); Qwen2-Audio PT and SALMONN decode non-overlapping 30 s
windows whose spans are shifted by the window offset and stitched (two same-label
spans are merged only if one ends and the next starts at the same window boundary);
Audio-Flamingo-3 and Kimi-Audio take the whole sequence. Predicted spans are
not clipped for accuracy, F1 or recall-side tIoU, for any system. -- = not applicable.}
\label{tab:main-results}

\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}ll ccc ccc ccc ccc ccc@{}}
\toprule
& & \multicolumn{3}{c}{$L=2$} & \multicolumn{3}{c}{$L=3$}
  & \multicolumn{3}{c}{$L=4$--$6$} & \multicolumn{3}{c}{$L\geq7$}
  & \multicolumn{3}{c}{\textbf{All}} \\
& & \multicolumn{3}{c}{($n=2{,}604$)} & \multicolumn{3}{c}{($n=590$)}
  & \multicolumn{3}{c}{($n=371$)} & \multicolumn{3}{c}{($n=83$)}
  & \multicolumn{3}{c}{($n=3{,}648$)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
\cmidrule(lr){12-14}\cmidrule(lr){15-17}
Model & Setting & Acc & F1 & tIoU$_F$ & Acc & F1 & tIoU$_F$ & Acc & F1 & tIoU$_F$
               & Acc & F1 & tIoU$_F$ & Acc & F1 & tIoU$_F$ \\
\midrule
\multicolumn{17}{@{}l}{\textit{Audio-language models}} \\
Qwen2-Audio      & PT  & 0.325 & 0.128 & 0.416 & 0.258 & 0.129 & 0.434 & 0.210 & 0.106 & 0.333 & 0.212 & 0.110 & 0.291 & 0.283 & 0.125 & 0.399 \\
% Qwen2 SFT: trained and evaluated on audio truncated to 30 s (Qwen2-Audio encoder limit; affects 203/3,648 Test1 sequences, 78/83 of L>=7). Not retrained; needs a table footnote.
                 & SFT & 0.591 & 0.308 & 0.825 & 0.531 & 0.277 & 0.705 & 0.463 & 0.162 & 0.265 & 0.493 & 0.081 & 0.197 & 0.549 & 0.236 & 0.543 \\
Audio-Flamingo-3 & PT  & 0.410 & 0.030 & 0.022 & 0.404 & 0.033 & 0.024 & 0.389 & 0.035 & 0.033 & 0.478 & 0.064 & 0.035 & 0.411 & 0.033 & 0.024 \\
                 & SFT & 0.606 & 0.344 & 0.815 & 0.516 & 0.304 & 0.664 & 0.442 & 0.271 & 0.567 & 0.475 & 0.229 & 0.509 & 0.549 & 0.321 & 0.721 \\
Qwen3-Omni       & PT  & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
                 & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
Kimi-Audio       & PT  & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
                 & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
SALMONN          & PT  & 0.291 & 0.021 & 0.160 & 0.302 & 0.044 & 0.212 & 0.264 & 0.032 & 0.220 & 0.217 & 0.029 & 0.227 & 0.282 & 0.026 & 0.180 \\
                 & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
\addlinespace
\multicolumn{17}{@{}l}{\textit{Content-free references}} \\
5\,s grid, all Neutral & -- & 0.344 & 0.050 & 0.572 & 0.362 & 0.053 & 0.531 & 0.404 & 0.059 & 0.507 & 0.370 & 0.051 & 0.464 & 0.360 & 0.052 & 0.544 \\
Single Neutral span    & -- & 0.168 & 0.050 & 0.398 & 0.110 & 0.039 & 0.218 & 0.090 & 0.035 & 0.112 & 0.039 & 0.020 & 0.030 & 0.133 & 0.044 & 0.296 \\
\addlinespace
\multicolumn{17}{@{}l}{\textit{Supervised speech baselines}} \\
WavLM (IS 2025)  & Utt. & 0.525 & 0.345 & \na & 0.506 & 0.303 & \na & 0.516 & 0.332 & \na & 0.516 & 0.292 & \na & 0.519 & 0.335 & \na \\
WavLM-TimeAug    & Seq. & 0.485 & 0.287 & 0.715 & 0.445 & 0.256 & 0.645 & 0.408 & 0.249 & 0.579 & 0.423 & 0.234 & 0.524 & 0.459 & 0.277 & 0.664 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Performance on MSP-Podcast v1.12 Test1 ($n=3{,}648$ sequences),
stratified by sequence length $L$. \emph{Utt.} rows are single-label
discrete SER baselines that classify each segment independently using gold
segment boundaries; \emph{Seq.} rows predict timed emotion spans over the
whole concatenated sequence. PT denotes the pretrained model and SFT denotes
LoRA fine-tuning on the corresponding task. WavLM (Interspeech 2025) is an
utterance-level classifier; WavLM-TimeAug is a separately trained
time-augmented sequence model. Accuracy and F1 measure emotion
classification. tIoU$_F$ measures temporal overlap: it is the harmonic mean of
recall-side tIoU (each gold span's IoU with its one-to-one matched prediction,
averaged over gold spans) and precision-side tIoU (the same matching on predictions
clipped to the audio, averaged over predicted spans; unmatched spans and spans past
the end count as 0), so over-segmentation is penalised. Both components are in
Table~\ref{tab:tiou-components}. Content-free references give the scores of a
back-to-back 5\,s Neutral grid and of one Neutral span covering the audio. Gold
segments average 5.8\,s, so the grid is close to gold granularity and marks what timing
alone earns: read tIoU$_F$ relative to the grid row, not to zero. -- marks tIoU$_F$ for utterance-level systems, which
are given gold boundaries. Audio over 30 s (Seq.): Qwen2-Audio SFT sees the
first 30 s (as trained); Qwen2-Audio PT and SALMONN decode non-overlapping
30 s windows whose spans are shifted by the window offset and stitched (two same-label
spans are merged only if one ends and the next starts at the same window boundary);
Audio-Flamingo-3 and Kimi-Audio take the whole sequence. Predicted spans are
not clipped for accuracy, F1 or recall-side tIoU, for any system.}
\label{tab:main-results}

\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lll ccc ccc ccc ccc ccc@{}}
\toprule
& & & \multicolumn{3}{c}{$L=2$} & \multicolumn{3}{c}{$L=3$}
  & \multicolumn{3}{c}{$L=4$--$6$} & \multicolumn{3}{c}{$L\geq7$}
  & \multicolumn{3}{c}{\textbf{All}} \\
& & & \multicolumn{3}{c}{($n=2{,}604$)} & \multicolumn{3}{c}{($n=590$)}
  & \multicolumn{3}{c}{($n=371$)} & \multicolumn{3}{c}{($n=83$)}
  & \multicolumn{3}{c}{($n=3{,}648$)} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}
\cmidrule(lr){13-15}\cmidrule(lr){16-18}
Model & Task & Setting & Acc & F1 & tIoU$_F$ & Acc & F1 & tIoU$_F$ & Acc & F1 & tIoU$_F$
                       & Acc & F1 & tIoU$_F$ & Acc & F1 & tIoU$_F$ \\
\midrule
\multicolumn{18}{@{}l}{\textit{Audio-language models}} \\
Qwen2-Audio      & Utt. & PT  & 0.326 & 0.204 & \na & 0.329 & 0.177 & \na & 0.310 & 0.150 & \na & 0.360 & 0.172 & \na & 0.327 & 0.189 & \na \\
                 &      & SFT & 0.607 & 0.374 & \na & 0.608 & 0.340 & \na & 0.565 & 0.347 & \na & 0.564 & 0.297 & \na & 0.596 & 0.360 & \na \\
                 & Seq. & PT  & 0.325 & 0.128 & 0.416 & 0.258 & 0.129 & 0.434 & 0.210 & 0.106 & 0.333 & 0.212 & 0.110 & 0.291 & 0.283 & 0.125 & 0.399 \\
                 &      & SFT & 0.591 & 0.308 & 0.825 & 0.531 & 0.277 & 0.705 & 0.463 & 0.162 & 0.265 & 0.493 & 0.081 & 0.197 & 0.549 & 0.236 & 0.543 \\
\addlinespace
Audio-Flamingo-3 & Utt. & PT  & 0.343 & 0.094 & \na & 0.360 & 0.096 & \na & 0.417 & 0.113 & \na & 0.400 & 0.098 & \na & 0.364 & 0.098 & \na \\
                 &      & SFT & 0.620 & 0.397 & \na & 0.605 & 0.382 & \na & 0.570 & 0.343 & \na & 0.591 & 0.306 & \na & 0.606 & 0.381 & \na \\
                 & Seq. & PT  & 0.410 & 0.030 & 0.022 & 0.404 & 0.033 & 0.024 & 0.389 & 0.035 & 0.033 & 0.478 & 0.064 & 0.035 & 0.411 & 0.033 & 0.024 \\
                 &      & SFT & 0.606 & 0.344 & 0.815 & 0.516 & 0.304 & 0.664 & 0.442 & 0.271 & 0.567 & 0.475 & 0.229 & 0.509 & 0.549 & 0.321 & 0.721 \\
\addlinespace
Qwen3-Omni       & Utt. & PT  & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na \\
                 &      & SFT & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na \\
                 & Seq. & PT  & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
                 &      & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
\addlinespace
Kimi-Audio       & Utt. & PT  & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na \\
                 &      & SFT & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na \\
                 & Seq. & PT  & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
                 &      & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
\addlinespace
SALMONN          & Utt. & PT  & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na \\
                 &      & SFT & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na & \tbd & \tbd & \na \\
                 & Seq. & PT  & 0.291 & 0.021 & 0.160 & 0.302 & 0.044 & 0.212 & 0.264 & 0.032 & 0.220 & 0.217 & 0.029 & 0.227 & 0.282 & 0.026 & 0.180 \\
                 &      & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
\addlinespace
\multicolumn{18}{@{}l}{\textit{Content-free references}} \\
5\,s grid, all Neutral & Seq. & -- & 0.344 & 0.050 & 0.572 & 0.362 & 0.053 & 0.531 & 0.404 & 0.059 & 0.507 & 0.370 & 0.051 & 0.464 & 0.360 & 0.052 & 0.544 \\
Single Neutral span    & Seq. & -- & 0.168 & 0.050 & 0.398 & 0.110 & 0.039 & 0.218 & 0.090 & 0.035 & 0.112 & 0.039 & 0.020 & 0.030 & 0.133 & 0.044 & 0.296 \\
\addlinespace
\multicolumn{18}{@{}l}{\textit{Supervised speech baselines}} \\
WavLM (IS 2025)  & Utt. & --  & 0.525 & 0.345 & \na & 0.506 & 0.303 & \na & 0.516 & 0.332 & \na & 0.516 & 0.292 & \na & 0.519 & 0.335 & \na \\
WavLM-TimeAug    & Seq. & --  & 0.485 & 0.287 & 0.715 & 0.445 & 0.256 & 0.645 & 0.408 & 0.249 & 0.579 & 0.423 & 0.234 & 0.524 & 0.459 & 0.277 & 0.664 \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\centering
\caption{Temporal overlap components on MSP-Podcast v1.12 Test1 sequences (Seq. rows of
Table~\ref{tab:main-results}). R: recall-side tIoU, each gold span's IoU with its
one-to-one matched prediction (closest-span greedy matching), averaged over gold spans,
predictions not clipped. P: precision-side tIoU, the same matching on predictions clipped to
the audio, averaged over predicted spans. Unmatched spans, spans past the end of the audio
and gold spans of unparsed outputs count as 0. F: harmonic mean of R and P, computed per
bin from the pooled R and P. Content-free references as in Table~\ref{tab:main-results}.}
\label{tab:tiou-components}
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}ll ccc ccc ccc ccc ccc@{}}
\toprule
& & \multicolumn{3}{c}{$L=2$} & \multicolumn{3}{c}{$L=3$}
  & \multicolumn{3}{c}{$L=4$--$6$} & \multicolumn{3}{c}{$L\geq7$}
  & \multicolumn{3}{c}{\textbf{All}} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
\cmidrule(lr){12-14}\cmidrule(lr){15-17}
Model & Setting & R & P & F & R & P & F & R & P & F & R & P & F & R & P & F \\
\midrule
\multicolumn{17}{@{}l}{\textit{Audio-language models}} \\
Qwen2-Audio      & PT  & 0.510 & 0.352 & 0.416 & 0.402 & 0.471 & 0.434 & 0.286 & 0.399 & 0.333 & 0.249 & 0.350 & 0.291 & 0.428 & 0.373 & 0.399 \\
                 & SFT & 0.865 & 0.788 & 0.825 & 0.722 & 0.689 & 0.705 & 0.603 & 0.170 & 0.265 & 0.536 & 0.121 & 0.197 & 0.764 & 0.421 & 0.543 \\
Audio-Flamingo-3 & PT  & 0.206 & 0.011 & 0.022 & 0.191 & 0.013 & 0.024 & 0.181 & 0.018 & 0.033 & 0.100 & 0.021 & 0.035 & 0.190 & 0.013 & 0.024 \\
                 & SFT & 0.836 & 0.795 & 0.815 & 0.630 & 0.701 & 0.664 & 0.515 & 0.630 & 0.567 & 0.507 & 0.512 & 0.509 & 0.713 & 0.730 & 0.721 \\
Qwen3-Omni       & PT  & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
                 & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
Kimi-Audio       & PT  & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
                 & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
SALMONN          & PT  & 0.618 & 0.092 & 0.160 & 0.581 & 0.130 & 0.212 & 0.555 & 0.137 & 0.220 & 0.561 & 0.143 & 0.227 & 0.595 & 0.106 & 0.180 \\
                 & SFT & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd & \tbd \\
\addlinespace
\multicolumn{17}{@{}l}{\textit{Content-free references}} \\
5\,s grid, all Neutral & -- & 0.677 & 0.496 & 0.572 & 0.617 & 0.467 & 0.531 & 0.581 & 0.450 & 0.507 & 0.566 & 0.394 & 0.464 & 0.639 & 0.474 & 0.544 \\
Single Neutral span    & -- & 0.299 & 0.597 & 0.398 & 0.145 & 0.435 & 0.218 & 0.069 & 0.311 & 0.112 & 0.016 & 0.155 & 0.030 & 0.205 & 0.532 & 0.296 \\
\addlinespace
\multicolumn{17}{@{}l}{\textit{Supervised speech baselines}} \\
WavLM-TimeAug    & Seq. & 0.702 & 0.729 & 0.715 & 0.608 & 0.688 & 0.645 & 0.529 & 0.638 & 0.579 & 0.480 & 0.576 & 0.524 & 0.635 & 0.695 & 0.664 \\
\bottomrule
\end{tabular}
\end{table*}
