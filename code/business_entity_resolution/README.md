# Business Entity Resolution — Amazon ML Challenge 2026

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (from student_resource/ directory)
python -m src.train --train-dir ../../dataset/train --out-model output/model.joblib

# 3. Generate predictions on test set
python -m src.predict --test-dir ../../dataset/test --model output/model.joblib --out-dir ../../output

# 4. Validate submission
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

## Project Structure

```
src/
├── __init__.py
├── config.py           # Central configuration & hyperparameters
├── preprocessing.py    # Text normalization (names, addresses, Unicode)
├── blocking.py         # Multi-key inverted-index blocking
├── features.py         # Pairwise similarity feature extraction
├── model.py            # LightGBM training, threshold tuning, prediction
├── evaluate.py         # F_0.5 macro-average scoring
├── train.py            # End-to-end training pipeline entry point
├── predict.py          # End-to-end test inference entry point
└── utils.py            # I/O helpers (TSV read/write, logging)
```

## Pipeline Overview

1. **Preprocessing** — Unicode NFKD normalization, legal suffix stripping,
   address abbreviation expansion, ASCII folding for diacritics, lowercasing.
2. **Blocking** — Country-partitioned, frequency-capped inverted index on
   cleaned name tokens + address number/locality keys. Generates a tight
   candidate set (≤20 candidates per S1 entity on average).
3. **Feature Extraction** — For each (S1, candidate) pair: Jaro-Winkler,
   token-sort ratio, token-set ratio, Jaccard (char 3-grams), address token
   overlap, numeric overlap, name/address TF-IDF cosine.
4. **Classification** — LightGBM binary classifier trained on ground-truth
   pairs vs. hard negatives from blocking. Threshold calibrated to maximize
   macro-averaged F_0.5 on validation split.
5. **Output** — `matching_results.tsv` (final matches) and
   `candidate_pairs.tsv` (blocking output), both tab-separated, validated
   with the official `validate_submission.py`.

## Reproducibility

The entire pipeline is deterministic given the same data and random seed
(configured in `src/config.py`). Runtime on 8-core / 16 GB RAM machine:
~30–60 minutes for training, ~20–40 minutes for test inference.

## License

All dependencies and model code use MIT / Apache 2.0 licensed components.
LightGBM is MIT-licensed.
