# Grammar Scoring Engine (v3 and v4)

A speech-based grammar scoring pipeline for a Kaggle competition. Each input is a short audio clip of a person speaking. The target is a grammar score (a number from 0 to 5). Submissions are scored by RMSE (lower is better).

This repository holds two notebooks:

| Notebook | Role | Status |
|---|---|---|
| `Grammar_Scoring_Engine_v3.ipynb` | Working baseline. Text features, grammar-error features and one Whisper audio embedding, blended by a stacker. | Run end to end on real data. Leaderboard 0.64 to 0.39. |
| `Grammar_Scoring_Engine_v4.ipynb` | Upgrade. Several audio models, a layer sweep, fusion models, diagnostics and safer submission handling. | Modeling and submission logic tested on synthetic data only. The audio-embedding extraction has not yet been run on real clips. |

---

## Table of contents

1. [Task and data](#1-task-and-data)
2. [Version 3 in detail](#2-version-3-in-detail)
3. [Version 4 in detail](#3-version-4-in-detail)
4. [Differences between v3 and v4](#4-differences-between-v3-and-v4)
5. [Results so far](#5-results-so-far)
6. [How to run](#6-how-to-run)
7. [Configuration reference](#7-configuration-reference)
8. [Outputs and caches](#8-outputs-and-caches)
9. [Known issues and caveats](#9-known-issues-and-caveats)
10. [Troubleshooting](#10-troubleshooting)
11. [Ideas for further improvement](#11-ideas-for-further-improvement)

---

## 1. Task and data

| Item | Value |
|---|---|
| Input | One audio clip per row (`filename`) |
| Target | `label`, a grammar score between 0 and 5 |
| Training clips | 769 (`train.csv` plus a `train/` audio folder) |
| Test clips | 216 (`test.csv` plus a `test/` audio folder) |
| Metric | RMSE |
| Baseline RMSE | About 1.24 (predicting the training mean, the label standard deviation) |

Expected folder layout on Google Drive:

```
MyDrive/data/
    train.csv
    test.csv
    sample_submission.csv
    train/            # training audio files
    test/             # test audio files
```

Notes about the data that matter for modeling:

* The training set is small (769 rows). Complex models overfit easily, so both versions use repeated cross-validation and regularised models.
* Very low scores are rare. Only 41 training clips have a score of 1.5 or lower, and the v3 model's smallest test prediction was about 1.75, so very poor speakers are probably over-predicted.
* `sample_submission.csv` has 204 rows while `test.csv` has 216. See [Known issues](#9-known-issues-and-caveats).

---

## 2. Version 3 in detail

### 2.1 Idea

A pure text approach loses most of the signal, because Whisper silently corrects grammar when it transcribes. v3 therefore combines three kinds of evidence:

1. **Grammar-error features** measured directly on the transcript.
2. **Whisper confidence and timing features**, which act as a proxy for unclear or erroneous speech.
3. **A learned audio embedding** from the Whisper encoder, which captures pronunciation and fluency.

Many different models are trained on these views, and their out-of-fold predictions are blended by a stacker.

### 2.2 Pipeline

```
audio clip
   |
   +--> Whisper small (word timestamps, greedy decoding, English)
   |        |-- transcript
   |        |-- word probabilities, segment log-probs, timing, pauses
   |
   +--> Transcript analysis
   |        |-- POS and lexical features (NLTK)
   |        |-- LanguageTool error counts
   |        |-- CoLA grammatical-acceptability scores
   |        |-- GPT-2 sentence perplexity
   |
   +--> Small acoustic set (librosa)
   |
   +--> Whisper encoder embedding (last layer, mean + std pooled)

feature sets:  tabular (132)  |  audio embedding (1536)  |  text embedding (768)  |  TF-IDF

8 base models, repeated stratified 5-fold x 3  -->  out-of-fold predictions
                                   |
              non-negative linear stacker  -->  clip to label range  -->  submission
```

### 2.3 Feature groups

| Group | What it contains |
|---|---|
| Whisper confidence | Word-probability mean, std, min, 10th and 25th percentile, share below 0.5 and 0.3. Segment log-probability, compression ratio, no-speech probability |
| Timing and fluency | Speech rate, articulation rate, phonation ratio, pause counts (0.25 s, 0.5 s, 1 s), pause lengths, leading and trailing silence, adjacent repeated words, filler tokens |
| Text and POS | POS ratios, verb/noun ratio, past-tense share, lexical diversity, word length, subordination rate, filler phrases, sentence-length statistics |
| LanguageTool | Total and "core" error counts (punctuation, casing and style are excluded because Whisper decides them), per-100-word rates, counts by category |
| CoLA | Mean, minimum, std and share-below-0.5 of per-sentence acceptability from `textattack/roberta-base-CoLA` |
| GPT-2 | Mean, max, median and std of per-sentence negative log-likelihood |
| Acoustic | Duration, RMS energy, zero-crossing rate, spectral centroid, 13 MFCC means, pitch statistics, voiced fraction |
| Whisper encoder embedding | Last encoder layer, mean and standard deviation over time, processed in 30-second chunks (1536 dimensions for Whisper small) |
| Text embedding | `all-mpnet-base-v2` sentence embedding of the transcript (768 dimensions) |

Tabular preprocessing: columns that are mostly empty or constant are dropped, gaps are filled with the training median, and values are clipped to the training 0.5% and 99.5% percentiles. Test data uses training statistics only.

### 2.4 Models

| Model | Input | Notes |
|---|---|---|
| `xgb_tab` | Tabular | 500 shallow trees, strong regularisation |
| `et_tab` | Tabular | ExtraTrees, 500 trees |
| `ridge_tab` | Tabular | Standardised, Ridge with automatic penalty |
| `ridge_whisper` | Audio embedding | Standardised Ridge |
| `svr_whisper` | Audio embedding | RBF SVR, `C` chosen by inner CV |
| `ridge_textemb` | Text embedding | Ridge |
| `svr_textemb` | Text embedding | SVR |
| `tfidf_ridge` | Transcript TF-IDF | Ridge |

### 2.5 Validation and stacking

* **Repeated stratified 5-fold, 3 repeats** (15 fits per model), stratified on five score bands. This is more stable than a single split on a small dataset.
* For each model, out-of-fold (OOF) predictions are averaged across repeats, and every fold model also predicts the test set. The test predictions are averaged. The model that is evaluated is therefore the model that is submitted.
* The stacker is a **non-negative linear regression with intercept** on the OOF predictions. Non-negativity stops it from using large opposite-sign weights to chase noise. The intercept and weight sum let it undo the shrinkage that averaging introduces.
* Final predictions are clipped to the training label range.

### 2.6 What v3 taught us

* The audio embedding models were the strongest single models, and the stacker gave most of its weight to them and to XGBoost.
* Transcript-only models were weak (RMSE 0.93 or worse). Whisper's own correction removes much of the evidence of errors.
* Grammar-error features from LanguageTool, CoLA and GPT-2 help the tabular model but do not replace the audio signal.

---

## 3. Version 4 in detail

### 3.1 Idea

v3 used one layer of one audio model. Speech models are layered: the last layer is specialised for the model's own training task, while middle layers usually keep pronunciation, fluency and prosody information. v4 therefore:

1. Extracts embeddings from **several layers of several audio models**.
2. **Measures** which layers work best instead of guessing.
3. **Fuses** the best layers from all models before predicting.
4. Adds **diagnostics** (train/test shift, error by score band) and a **safer submission step**.

v4 reuses the v3 transcript and tabular cache, so nothing is re-transcribed.

### 3.2 Pipeline

```
v3 cache (transcripts + tabular features)  ---------------------+
                                                                 |
audio clip --> per backbone, per chosen layer:                   |
               mean + std pooling over time (30 s chunks)        |
                                                                 v
   Whisper small   layers 6, 8, 10, 12          tabular + text embedding
   Whisper medium  layers 12, 16, 20, 24                         |
   WavLM large     layers 8, 12, 16, 20, 24                      |
   HuBERT large    layers 8, 12, 16, 20, 24                      |
                |                                                |
        layer sweep: one Ridge per layer, CV RMSE table          |
        keep the best TOP_K_LAYERS (default 2) per backbone      |
                |                                                |
   +------------+---------------+--------------------+-----------+
   |                            |                    |
 Ridge + SVR on          fusion_audio          fusion_all
 selected layers         (best layer of        (+ tabular + text
                          each backbone)        embedding)
   |                            |                    |
   +----------------------------+--------------------+
                                |
              non-negative stacker (+ optional isotonic calibration)
                                |
              ID reconciliation --> submission files
```

### 3.3 Audio backbones

| Name | Source | Layers pooled |
|---|---|---|
| `whisper_small` | OpenAI Whisper encoder | 6, 8, 10, 12 |
| `whisper_medium` | OpenAI Whisper encoder | 12, 16, 20, 24 |
| `wavlm_large` | `microsoft/wavlm-large` | 8, 12, 16, 20, 24 |
| `hubert_large_ft` | `facebook/hubert-large-ls960-ft` | 8, 12, 16, 20, 24 |

For each chosen layer the notebook computes the mean and standard deviation of the frame vectors over time. The mean captures typical voice quality, and the standard deviation captures variation (prosody, hesitation). Clips are processed in 30-second chunks, and padding frames are excluded for Whisper. Each clip's result is cached as a separate `.npy` file, so an interrupted run resumes where it stopped.

### 3.4 New steps in v4

| Step | What it does | Why |
|---|---|---|
| Layer sweep | Trains one Ridge model per backbone and layer, prints a ranked table, keeps the top `TOP_K_LAYERS` per backbone | Finds which depth carries the signal |
| SVR on selected layers | Adds a non-linear model per kept layer | Different errors from Ridge, adds diversity |
| Fusion models | Ridge and SVR on a matrix that joins the best layer of every backbone (`fusion_audio`), and the same plus tabular and text blocks (`fusion_all`). Each block is standardised and divided by the square root of its dimension so every block has equal total weight | Lets one model see all audio views at once |
| Adversarial validation | A classifier tries to separate train clips from test clips and reports the AUC | Detects train/test shift (about 0.50 means similar, above about 0.70 means a clear shift) |
| Error analysis | RMSE and bias per true-score band | Shows whether low scores dominate the error |
| Isotonic calibration check | Tests a monotone recalibration of the stacked output and uses it only if CV RMSE improves by at least 0.003 | Avoids a calibration that does not help |
| Lazy model loading | Whisper, LanguageTool, CoLA and GPT-2 load only for clips missing from the v3 cache | Saves minutes and GPU memory when the cache is complete |
| Safe submission step | Matches IDs exactly and after normalisation, never fills unmatched rows with 0, and writes a sample-aligned file only if at least 95% of IDs match | Fixes the silent-zero problem seen in v3 |

### 3.5 Models kept from v3

`xgb_tab`, `et_tab`, `ridge_tab` and `ridge_textemb` are kept. `tfidf_ridge` and the old `svr_textemb` are dropped because they added nothing in v3.

---

## 4. Differences between v3 and v4

| Area | v3 | v4 |
|---|---|---|
| Audio representation | Whisper small, last layer only | Whisper small and medium, WavLM large, HuBERT large, several layers each |
| Choosing a layer | None | Layer sweep with a ranked CV table |
| Audio model combination | Only inside the stacker | Fusion models plus the stacker |
| Base models | 8 (tabular, audio, text, TF-IDF) | 4 tabular and text models, one Ridge and one SVR per selected audio layer, 4 fusion models |
| Feature extraction | Always loads all extraction models | Lazy loading, reuses the v3 cache |
| Cache folders | `feature_cache_v3_small` | Same folder reused, plus `emb_cache_v4/<backbone>/` |
| Train/test shift check | None | Adversarial validation |
| Error analysis | None | Per score band RMSE and bias |
| Calibration | None | Optional isotonic, chosen by CV |
| Submission handling | Exact filename match, unmatched rows became 0 | Exact and normalised match, no zero fill, 95% match rule, always writes an all-test-rows file |
| Optional extras | DeBERTa fine-tuning cell (off by default) | Removed (text signal is weak) |
| Documentation | Short cell descriptions | A detailed markdown cell before every code cell |
| Runtime | Moderate | Longer the first time (model downloads and embedding extraction), then cached |
| Evidence | Run on real data | Modeling logic tested on synthetic data, extraction not yet run on real clips |

What stays the same: Whisper small transcripts and the 132 tabular features, the repeated stratified 5-fold x 3 splits (so numbers are comparable), the fold-averaged test predictions, and the non-negative stacker.

---

## 5. Results so far

### v3 (cross-validated RMSE on 769 training clips)

| Model family | CV RMSE |
|---|---|
| Whisper-small embedding, Ridge | 0.566 |
| Whisper-small embedding, SVR | 0.573 |
| Tabular, XGBoost | 0.611 |
| Tabular, ExtraTrees | 0.644 |
| Tabular, Ridge | 0.664 |
| Transcript sentence embedding | 0.93 to 0.94 |
| Transcript TF-IDF | about 1.05 |
| **Stacked** | **0.5275** |

Leaderboard: 0.64 before, **0.39** with v3. Top competitors are around 0.30.

CV and leaderboard do not match exactly, so use the CV difference between versions as the guide and the leaderboard as the final check.

### v4

No v4 results have been recorded yet. The notebook prints its own comparison: `v4 stacked CV` against the v3 value of `0.5275`. Any real improvement should show up as a clearly lower stacked CV plus a lower leaderboard score. Please record your numbers here after running it.

---

## 6. How to run

Both notebooks are written for Google Colab with a GPU (a T4 is enough).

1. Upload the competition data to `MyDrive/data/` as shown in [section 1](#1-task-and-data).
2. Open the notebook in Colab and select a GPU runtime.
3. Run the cells from top to bottom. The first cell installs the packages and Java (needed by LanguageTool).
4. **v3 first run:** transcribes every clip and caches the features. This is the slow part.
5. **v4 run:** loads the v3 cache, then extracts audio embeddings for each enabled backbone (a one-off download of about 4 GB of models, then a per-clip cache).
6. Upload the produced submission file to Kaggle.

Suggested order if you start from scratch: run v3 first (it builds the cache that v4 reuses), then v4.

Rough runtime guide (not measured): the v4 embedding extraction is probably around 10 to 15 minutes per backbone on a T4 the first time. If time is short, disable a backbone in the configuration.

---

## 7. Configuration reference

### v3 (config cell)

| Setting | Default | Meaning |
|---|---|---|
| `WHISPER_SIZE` | `"small"` | Whisper model for transcription |
| `N_SPLITS` | 5 | Cross-validation folds |
| `N_REPEATS` | 3 | Repeats of the fold split |
| `RUN_DEBERTA` | `False` | Optional DeBERTa fine-tuning cell |

### v4 (config cell)

| Setting | Default | Meaning |
|---|---|---|
| `WHISPER_SIZE` | `"small"` | Must stay `"small"` to reuse the v3 cache |
| `N_SPLITS`, `N_REPEATS` | 5, 3 | Same scheme as v3 |
| `TOP_K_LAYERS` | 2 | Best layers kept per backbone after the sweep |
| `V3_STACK_CV` | 0.5275 | Reference value for the before/after comparison |
| `BACKBONES` | four models | List of dicts with `name`, `kind` (`"whisper"` or `"hf"`), `id`, `layers`, `enabled` |

To add a larger model, append an entry such as:

```python
{"name": "whisper_large", "kind": "whisper", "id": "large-v3",
 "layers": [16, 20, 24, 28, 32], "enabled": True}
```

For Whisper backbones, layer `L` is the output of encoder block `L`. For Hugging Face backbones, layer `L` indexes `hidden_states` (0 is the input projection, 1 to N are transformer layers). Large models need more GPU memory.

---

## 8. Outputs and caches

| File or folder | Created by | Content |
|---|---|---|
| `feature_cache_v3_small/` | v3 (reused by v4) | One pickle per clip: features, transcript |
| `emb_cache_v4/<backbone>/` | v4 | One `.npy` per clip: pooled embeddings of the chosen layers |
| `submission_all_216.csv` and similar | v3 | Predictions for every row in `test.csv` |
| `submission_v4_all_test.csv` | v4 | One row for every `test.csv` file, always written |
| `submission_v4.csv` | v4 | Aligned to `sample_submission.csv`, written only when at least 95% of IDs match |
| `pipeline_v3.pkl`, `pipeline_v4.pkl` | v3, v4 | Feature settings, stacker, OOF and test predictions (v4 also stores the layer sweep table) |

The pickled OOF predictions let you test new stacking ideas in seconds without rerunning extraction or cross-validation. The fold models themselves are not stored.

---

## 9. Known issues and caveats

1. **Submission ID mismatch (important).** `sample_submission.csv` has 204 rows and `test.csv` has 216. In the v3 notebook, the sample-aligned `submission_v3.csv` printed a mean of about 0.40 with most quantiles at 0. That means most of its rows were filled with the fallback `0.0` because the IDs did not match. Check which file you uploaded to Kaggle. v4 reports the match counts and does not silently fill unmatched rows.
2. **v4 extraction is untested on real audio.** Only the modeling, stacking and submission cells were run (on synthetic data). The first real run may need small fixes, for example GPU memory limits or model-specific feature-extractor behaviour.
3. **Optimistic stacking estimate.** The stacker is scored by CV on out-of-fold predictions produced on the same data, so its RMSE is slightly optimistic. Use it to compare versions, not as a leaderboard forecast.
4. **Mild selection bias in the layer sweep.** Picking the best layers by the same CV that scores them slightly flatters those models. This is why the final stack also includes other models.
5. **Whisper changes the text.** Whisper tends to fix grammar, so transcript-based error counts under-report real errors. This is why audio features carry so much weight.
6. **Rare low scores.** With only 41 clips at or below 1.5, very poor speakers are likely over-predicted. The v4 error-analysis table shows the effect.
7. **Test shift is unknown.** The v3 test predictions had a standard deviation of about 0.77 against 1.24 for the labels. This can be normal shrinkage or a shift. v4's adversarial validation helps tell the two apart.
8. **Dependencies.** LanguageTool needs Java. Newer library versions can change attribute names or behaviour, so pin versions if you need exact reproducibility.

---

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `LanguageTool NOT available` | Java was not installed. Re-run the install cell. The notebook then skips those features, which changes the feature set, so rerun the whole pipeline for consistency |
| CUDA out of memory in v4 | Disable the largest backbone (`"enabled": False`), restart the runtime, rerun. Finished clips are cached |
| v4 says some files were "newly extracted" | The v3 cache was incomplete or came from a different `WHISPER_SIZE`. The extraction models load automatically for those files |
| `Unmatched` IDs in the submission step | `sample_submission.csv` and `test.csv` use different IDs. Read the printed examples, then upload the file whose IDs match what Kaggle expects |
| Very different `xgb_tab` score from v3 | The cache or feature set changed. Check `Tabular features` shape (should be 769 rows by about 132 columns) |
| Download or authentication errors for models | Check the Colab network and Hugging Face access, then rerun the cell |

---

## 11. Ideas for further improvement

In rough order of expected value:

1. Read the v4 layer sweep, stacker weights and error-analysis table. Add neighbouring layers and larger sibling models for the backbone family that works best (for example `whisper large-v3` or `wav2vec2-large-xlsr-53`).
2. Replace mean and std pooling with a small neural head (attention pooling) on frame-level features of the best layer, so the model can see where pauses and errors occur.
3. Fine-tune a speech encoder end to end with a regression head, using low learning rates, early stopping and several seeds.
4. Target low scores explicitly, for example with a "very low score" classifier or sample weights.
5. Use more CV repeats (for example `N_REPEATS = 5`) for smoother out-of-fold and test predictions.
6. Resolve the submission-ID question first, because it affects how every score should be read.
