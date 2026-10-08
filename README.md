# FMA Music Genre Classifier

Classical-feature genre classification on the [Free Music Archive (FMA) medium](https://github.com/mdeff/fma) subset: ~25k thirty-second clips, 16 top-level genres.

Paper: [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840).

## Data leakage

FMA tracks from the same artist share similar auditory features. A random split puts some of those tracks in train and some in test. Nearest-neighbor and tree models then partly re-identify the artist instead of learning genre.

The official FMA `training` / `validation` / `test` split keeps artists entirely on one side. Grouped 5-fold CV uses `StratifiedGroupKFold` on `artist_id` and never touches the official test set.

| Model               | Random-split test (leaked) | After leakage fix | Grouped 5-fold CV |
| ------------------- | -------------------------- | ----------------- | ----------------- |
| LightGBM            | —                          | **0.591**         | 0.606             |
| XGBoost             | —                          | 0.578             | **0.610**         |
| MLP                 | 0.634                      | 0.577             | 0.599             |
| Random Forest       | 0.613                      | 0.550             | 0.556             |
| k-NN (84 features)  | 0.577                      | 0.504             | 0.529             |
| k-NN (top-10 ANOVA) | 0.511                      | 0.456             | 0.480             |
| Logistic Regression | 0.452                      | 0.427             | 0.449             |

XGBoost and LightGBM were added after the split was fixed, so they have no leaked-holdout number. k-NN dropped the most (−7.3 points), which is what you would expect if the old test set contained neighbors from the same artist.

Accuracy is dominated by Rock and Electronic. Macro-F1 on the official test set is much lower (~0.33–0.37 for the tree/MLP models), which is the number that actually reflects rare classes (Blues, Easy Listening, Pop).

## Setup

1. Download [fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) and [fma_medium.zip](https://os.unil.cloud.switch.ch/fma/fma_medium.zip) from the [FMA repo](https://github.com/mdeff/fma).

```
data/
  fma_metadata/tracks.csv
  fma_medium/000/000002.mp3
  ...
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the repo root:

```bash
python3 -m src.basepipeline

```

## Repo layout

| Path                  | Role                                                                               |
| --------------------- | ---------------------------------------------------------------------------------- |
| `src/config.py`       | Paths, sample rate, model constants                                                |
| `src/features.py`     | Parallel librosa extraction (84-d vectors)                                         |
| `src/data.py`         | Where you put FMA files, metadata, etc.                                            |
| `src/train.py`        | Model defs, fit, pickle artifacts (`model`, `scaler`, `label_encoder`, `features`) |
| `src/evaluate.py`     | Test reports, confusion matrix, grouped CV                                         |
| `src/basepipeline.py` | End-to-end entry point                                                             |
| `src/utils.py`        | FMA metadata loader and mp3 path helper                                            |
