# 🎙️ Grammar Scoring Engine

An end-to-end **Audio-to-Grammar Scoring Machine Learning pipeline** that predicts a continuous grammar/proficiency score from spoken English audio.

The project combines **speech recognition, NLP-based linguistic features, speech-timing information, acoustic features, and classical machine learning regression models** to estimate a grammar score on a **0–5 scale**.

---

## 📌 Project Overview

The goal is to transform raw speech recordings into a numerical grammar score.

Instead of relying only on the transcript, the system extracts multiple complementary signals:

1. **Speech content** — what the speaker said.
2. **Linguistic structure** — basic NLP/POS information from the transcript.
3. **Speech timing** — segment-level timing information obtained from Whisper.
4. **Acoustic characteristics** — measurable properties of the speech signal.
5. **Machine learning regression** — predicts the final continuous grammar score.

The final system uses **108 numerical features** and compares multiple regression models using **5-fold cross-validation**.

---

## 🎯 Objective

Given an audio file containing spoken English, predict a grammar score between **0 and 5**.

```text
Audio → Speech Transcription → Feature Extraction → Regression Model → Grammar Score
```

### Primary competition metrics

- **Pearson Correlation** — higher is better
- **RMSE (Root Mean Squared Error)** — lower is better

### Supplementary metrics

- MAE
- R²
- Spearman Correlation

> Pearson Correlation and RMSE are the primary competition metrics. MAE, R², and Spearman Correlation are included as supplementary evaluation metrics.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │     Audio Dataset    │
                    │      (.wav files)    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Whisper        │
                    │  Speech Recognition  │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
          ┌────────────┐ ┌────────────┐ ┌────────────┐
          │ Transcript │ │  Segment   │ │   Audio    │
          │    NLP     │ │  Timing    │ │  Features  │
          └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Feature Fusion     │
                    │    108 Features      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    │ Numeric conversion   │
                    │ NaN / Inf handling   │
                    │ Median imputation    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ 5-Fold Cross         │
                    │ Validation           │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       Random Forest      Extra Trees         XGBoost
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Model Comparison     │
                    │ RMSE / Pearson /    │
                    │ MAE / R² / Spearman │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Best Model: XGBoost  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Test Set Prediction  │
                    │ Score clipped 0–5    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    submission.csv    │
                    └──────────────────────┘
```

---

# 📂 Dataset

The project uses three CSV files.

### `train.csv`

Contains training audio filenames and their ground-truth grammar scores.

| Column | Description |
|---|---|
| `index` | Dataset index |
| `filename` | Audio filename |
| `label` | Grammar score between 0 and 5 |

**Training samples: 769**

### `test.csv`

Contains filenames of unseen audio recordings.

**Test samples: 216**

### `sample_submission.csv`

Provided as the submission template. The final submission is constructed using the exact filename order from `test_df`.

---

# 🎧 Audio Data

The recordings are `.wav` files.

The notebook verifies that CSV filenames map correctly to available audio files.

Expected verification:

```text
TRAIN: 769/769 files found
TEST: 216/216 files found
```

Filename matching is based on the actual filename rather than assuming filesystem order.

---

# 🧠 Feature Engineering

The final fused representation contains **108 numerical features**.

## 1. Speech Transcription — Whisper

The project uses **Whisper Base** for automatic speech recognition.

```text
Audio
  ↓
Whisper
  ↓
English Speech Transcript
```

Whisper converts spoken audio into text so linguistic information can be extracted.

The transcription uses:

```python
temperature=0
```

This provides deterministic decoding behaviour.

---

## 2. NLP Features

The baseline used five linguistic features:

- `total_words`
- `num_nouns`
- `num_verbs`
- `num_adjs`
- `avg_word_len`

These features provide basic information about lexical and grammatical structure.

They are **indirect predictive signals**, not direct measurements of grammatical correctness.

---

## 3. Whisper Timing Features

Whisper provides **segment-level timestamps**.

These are converted into timing-related features such as:

- Speech duration
- Number of detected segments
- Segment-duration statistics
- Timing distribution information

These features provide information about how the speech unfolds over time.

> The implementation uses segment timestamps, not token-level timestamps.

---

## 4. Acoustic Features

Acoustic information is extracted from the waveform using **Librosa**.

The acoustic feature group captures complementary speech characteristics, including information related to:

- Energy
- Spectral properties
- Pitch-related characteristics
- Timbre
- Temporal/acoustic variation

These features are not direct grammar measurements. They provide additional speech-level information alongside linguistic and timing features.

---

## 5. Feature Fusion

The three feature groups are combined:

```text
NLP Features
      +
Whisper Timing Features
      +
Acoustic Features
      ↓
108-Dimensional Feature Vector
```

This is the main improvement over the original five-feature baseline.

---

# 🧹 Data Preprocessing

Before model training:

1. Feature columns are converted to numeric values.
2. Infinite values are replaced with missing values.
3. Missing numerical values are handled using median imputation.
4. Non-feature columns such as `filename`, `label`, and text fields are excluded.
5. The resulting matrix contains **108 numerical features**.

---

# 🤖 Machine Learning Models

## Random Forest Regressor

Used as the original baseline and as a model on the fused representation.

## Extra Trees Regressor

A randomized tree ensemble evaluated against Random Forest and XGBoost.

## XGBoost Regressor

The final selected model.

Configuration:

```python
XGBRegressor(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
```

---

# 🔄 Cross-Validation

The project uses **5-fold cross-validation**:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

For each fold:

```text
4 folds → Training
1 fold  → Validation
```

Every training example is used for validation once.

The resulting out-of-fold predictions are used to calculate the evaluation metrics.

---

# 📊 Evaluation Metrics

## Pearson Correlation

Measures linear correlation between predicted and actual scores.

Higher is better.

## RMSE

Measures the magnitude of prediction errors.

Lower is better.

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

## MAE

Average absolute prediction error. Lower is better.

## R²

Proportion of target variance explained by the model. Higher is better.

## Spearman Correlation

Measures rank-order correlation. Higher is better.

---

# 📈 Model Evaluation Results

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Pearson ↑ | Spearman ↑ |
|---|---:|---:|---:|---:|---:|
| Original 5-Feature Random Forest | 1.0774 | 0.8572 | 0.2429 | 0.5000 | 0.4313 |
| Fused Random Forest | 0.7425 | 0.5868 | 0.6404 | 0.8022 | 0.7005 |
| Fused Extra Trees | 0.7269 | 0.5759 | 0.6554 | 0.8128 | 0.7227 |
| **Fused XGBoost** | **0.7074** | **0.5495** | **0.6736** | **0.8211** | **0.7309** |

### 🏆 Best Model

**Fused XGBoost**

- **RMSE:** `0.7074`
- **Pearson Correlation:** `0.8211`

A Pearson value of `0.8211` can be expressed as **82.11% correlation**, but it should **not** be called model accuracy.

Recommended wording:

> **Achieved a Pearson Correlation of 0.8211 with an RMSE of 0.7074.**

---

# 📌 Baseline vs Improved Pipeline

### Original baseline

```text
Audio
 ↓
Whisper
 ↓
5 NLP Features
 ↓
Random Forest
```

### Improved pipeline

```text
Audio
 ↓
Whisper
 ├── Transcript → NLP features
 ├── Segment timing → Timing features
 └── Waveform → Acoustic features
                    ↓
              108 features
                    ↓
        Model comparison with CV
                    ↓
                 XGBoost
```

| Metric | Original RF | Fused XGBoost |
|---|---:|---:|
| RMSE | 1.0774 | **0.7074** |
| MAE | 0.8572 | **0.5495** |
| R² | 0.2429 | **0.6736** |
| Pearson | 0.5000 | **0.8211** |
| Spearman | 0.4313 | **0.7309** |

The fused approach substantially improves the validation metrics over the original five-feature baseline.

---

# 🏆 Model Selection

The notebook compares the models using cross-validation results.

The model with the **lowest cross-validated RMSE** is selected.

For this experiment:

```text
XGBoost Regressor
```

It also produced the highest Pearson Correlation among the evaluated models.

The selected model is then retrained on the complete training dataset.

---

# 🔮 Test Prediction

The final inference pipeline is:

```text
Full Training Dataset
        ↓
Train Selected XGBoost
        ↓
Test Features
        ↓
216 Predictions
        ↓
Clip to [0, 5]
        ↓
submission.csv
```

Predictions are constrained with:

```python
predictions = np.clip(predictions, 0, 5)
```

Final test prediction statistics:

- **Count:** 216
- **Minimum:** ≈ 2.0925
- **Maximum:** 5.0
- **Mean:** ≈ 3.2622

---

# 📤 Submission Generation

The submission is created from the test dataframe:

```python
submission = pd.DataFrame({
    "filename": test_df["filename"],
    "label": test_predictions
})
```

Integrity checks verify:

- Row count matches the test set.
- Filename order matches `test_df`.
- Predictions are within `[0, 5]`.
- No predictions are missing.

Output:

```text
submission.csv
```

---

# 💾 Feature Caching

Audio transcription and feature extraction are computationally expensive.

The project therefore caches extracted features and creates consolidated feature tables.

This separates:

```text
Expensive Feature Extraction
          ↓
Cached Features
          ↓
Fast Model Experiments
```

This makes repeated model experiments significantly more efficient.

---

# ☁️ Google Drive Structure

The notebook uses Google Drive for persistent audio storage.

Expected structure:

```text
MyDrive/
└── data/
    ├── train/
    │   ├── audio_....wav
    │   └── ...
    │
    └── test/
        ├── audio_....wav
        └── ...
```

---

# 🛠️ Technologies Used

### Programming

- Python

### Speech Recognition

- OpenAI Whisper

### NLP

- NLTK
- Tokenization
- Part-of-Speech tagging

### Audio Processing

- Librosa

### Machine Learning

- Scikit-learn
- Random Forest
- Extra Trees
- XGBoost

### Data Processing

- NumPy
- Pandas

### Environment

- Google Colab
- Google Drive

---

# 📦 Main Libraries

```text
numpy
pandas
scikit-learn
xgboost
librosa
nltk
openai-whisper
torch
```

---

# 🚀 How to Run

## 1. Open the notebook

Open the Grammar Scoring Engine notebook in Google Colab.

## 2. Prepare the data

Place audio files under:

```text
MyDrive/data/train/
MyDrive/data/test/
```

and provide the CSV files.

## 3. Mount Google Drive

Run the notebook's Drive setup cell.

## 4. Verify audio files

Run the audio-file validation step.

Expected:

```text
TRAIN: 769/769 files found
TEST: 216/216 files found
```

## 5. Extract features

For every audio file:

1. Load audio.
2. Transcribe using Whisper.
3. Extract NLP features.
4. Extract segment timing features.
5. Extract acoustic features.
6. Combine the features.
7. Cache the resulting feature vector.

## 6. Prepare the feature matrix

Perform numeric conversion, infinity handling, missing-value handling, and feature selection.

## 7. Run 5-fold cross-validation

Compare:

```text
Random Forest
Extra Trees
XGBoost
```

## 8. Select the best model

Select the model with the lowest cross-validated RMSE.

## 9. Train on all training data

Retrain the selected model using the complete labeled dataset.

## 10. Predict the test set

Generate predictions for all 216 test recordings.

## 11. Generate submission

Save:

```text
submission.csv
```

---

# 📁 Suggested Repository Structure

```text
Grammar-Scoring-Engine/
│
├── README.md
│
├── notebooks/
│   └── Grammar_Scoring_Engine.ipynb
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── features/
│   ├── train_features.csv
│   └── test_features.csv
│
├── models/
│   └── xgb_grammar_model.pkl
│
├── outputs/
│   └── submission.csv
│
├── requirements.txt
│
└── .gitignore
```

Raw audio and other large files should generally not be committed directly to GitHub.

---

# 🔬 Experimentation Journey

### Phase 1 — Baseline

```text
Whisper → NLP Features → Random Forest
```

A simple five-feature pipeline established a working baseline.

### Phase 2 — Feature Expansion

Additional timing and acoustic information was introduced.

### Phase 3 — Feature Fusion

The feature groups were combined into a **108-feature representation**.

### Phase 4 — Model Benchmarking

Random Forest, Extra Trees, and XGBoost were compared using 5-fold cross-validation.

### Phase 5 — Final Model

XGBoost achieved the strongest validation performance.

### Phase 6 — Submission

Predictions were generated for all 216 test recordings and written to the required submission format.

---

# ⚠️ Important Considerations

### Pearson is not accuracy

Do not report:

> "82.11% accuracy"

Instead report:

> **Pearson Correlation = 0.8211**

or:

> **Achieved a Pearson Correlation of 0.8211 with RMSE 0.7074.**

### Features are predictive signals

NLP, timing, and acoustic features do not individually measure grammar correctness. The model learns their relationship with the provided grammar scores.

### Validation vs test performance

The reported metrics are **5-fold cross-validation results on the labeled training data**.

The hidden test labels are unavailable locally, so final test-set Pearson/RMSE can only be obtained from the competition/evaluation system.

---

# 🚧 Current Limitations

- Whisper transcription errors can propagate into NLP features.
- Simple POS-based features do not fully capture grammatical structure.
- Acoustic features can capture speaking style in addition to grammar-related information.
- The labeled dataset is relatively small for complex end-to-end deep learning.
- Classical models depend strongly on feature engineering.
- Median imputation is performed before the cross-validation prediction step; a stricter publication-grade setup would place preprocessing inside each CV fold using a scikit-learn `Pipeline`.

---

# 🔮 Future Improvements

## Advanced linguistic features

Potential additions:

- Dependency parsing
- Sentence complexity
- Clause counts
- Grammatical error counts
- Vocabulary richness
- Lexical diversity
- Repetition rate
- Sentence-length statistics

## Transformer-based text embeddings

Generate semantic embeddings from transcripts and combine them with acoustic features.

## Learned audio embeddings

Use pretrained speech/audio models to replace or complement handcrafted acoustic features.

## Multimodal neural model

```text
Audio Representation
        +
Text Representation
        +
Timing Representation
        ↓
Multimodal Regression Model
        ↓
Grammar Score
```

## Better CV preprocessing

Move imputation and other preprocessing steps inside each cross-validation fold to avoid using information from validation folds during preprocessing.

## Hyperparameter optimization

Potential tools:

- Optuna
- RandomizedSearchCV
- Bayesian optimization

---

# 📊 Key Results

| Item | Result |
|---|---|
| Training samples | **769** |
| Test samples | **216** |
| Feature count | **108** |
| CV folds | **5** |
| Models compared | **3** |
| Best model | **XGBoost Regressor** |
| Best RMSE | **0.7074** |
| Best Pearson | **0.8211** |
| Best MAE | **0.5495** |
| Best R² | **0.6736** |
| Best Spearman | **0.7309** |
| Prediction range | **0–5** |
| Submission rows | **216** |

---

# 💡 What This Project Demonstrates

This project demonstrates an end-to-end machine learning workflow involving:

- Audio preprocessing
- Automatic Speech Recognition
- NLP feature engineering
- Acoustic feature engineering
- Feature fusion
- Missing-value handling
- Regression modeling
- Cross-validation
- Model benchmarking
- Metric-based model selection
- Test-set inference
- Submission validation

The key idea is to move from a simple transcript-based baseline to a richer feature-fusion system that uses **linguistic + timing + acoustic information**.

---

# 👨‍💻 Author

**Naman Jain**

B.Tech Computer Science & Engineering

Areas of interest:

- Artificial Intelligence
- Machine Learning
- Natural Language Processing
- Computer Vision
- Software Development

---

# ⭐ Final Pipeline

```text
🎙️ Audio
   ↓
🗣️ Whisper Base
   ↓
📝 NLP Features
   +
⏱️ Segment Timing Features
   +
🔊 Acoustic Features
   ↓
🔗 108-Feature Fusion
   ↓
🧹 Preprocessing
   ↓
🔄 5-Fold Cross-Validation
   ↓
🤖 RF / Extra Trees / XGBoost
   ↓
🏆 XGBoost
   ↓
🎯 Grammar Score (0–5)
   ↓
📤 submission.csv
```

**Final cross-validation result: Pearson = 0.8211, RMSE = 0.7074.**
