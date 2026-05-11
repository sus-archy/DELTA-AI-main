# DELTA-AI

Reproducible ML stack for IOC severity prediction using the DB `severity` label as ground truth.

This repository packages the code, configs, benchmark results, and the small best `logreg` artifact.
The large DB, transformer checkpoints, and model caches are intentionally not committed.

## What Is In This Repo

- `ml/`
  - dataset building, feature engineering, training, fine-tuning, benchmarking, reporting
- `configs/`
  - reproducible benchmark configs, including the best 20k round-2 setup and the prepared 100k shortlist
- `scripts/`
  - PowerShell launchers (Windows) and bash equivalents (cross-platform)
  - the `severityClassifier.js` reference scorer
- `results/`
  - curated benchmark summaries and visuals
- `released_models/`
  - exported best `logreg` model artifact
- `reports/`
  - reasoning notes and benchmark planning documents
- `Data/`
  - place your `DB-ThreatIndicators.csv` here (see below)

## What Is Not In This Repo

- `Data/DB-ThreatIndicators.csv` — your IOC database (see Data Placement below)
- Hugging Face caches (downloaded on first use under `.cache/`)
- SecureBERT checkpoints (downloaded on first use under `models/`)
- local scratch artifacts under `artifacts/`

## Ground Truth

- The DB `severity` label is the canonical target.
- Supported classes: `medium`, `high`, `critical`

## Requirements

### Core (training and inference)

- Python `3.11` or higher
- pip or uv

### Optional — Neural models (LoRA fine-tuning, embeddings)

- `torch>=2.6`
- `transformers>=5.2.0`
- `faiss-cpu>=1.13` (or `faiss-gpu`)
- `sentence-transformers>=5.2`
- `huggingface_hub>=1.5`
- GPU recommended (CUDA) for reasonable training speed

### Research extras

- `catboost`, `captum`, `shap`

## Installation

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd DELTA-AI-main

# Create virtual environment (any Python 3.11+)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell
# .venv\Scripts\Activate.bat     # Windows CMD

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install core dependencies

```bash
# Core ML stack (scikit-learn, XGBoost, pandas, etc.)
python -m pip install numpy pandas pyarrow pyyaml scikit-learn xgboost joblib tqdm rich

# Optional: neural dependencies (only needed for LoRA fine-tuning or embedding models)
# pip install torch transformers sentence-transformers huggingface_hub faiss-cpu

# Or install from pyproject.toml
python -m pip install -e .          # core only
python -m pip install -e ".[research]"  # core + research extras
```

### 3. Place your data

Only **one file is required** — your IOC database CSV:

```
Data/
└── DB-ThreatIndicators.csv
```

Optional enrichment files (if you have them):

```
Data/
├── DB-ThreatIndicators.csv   ← REQUIRED
├── merged_ip_list.txt        ← optional: known bad IPs / CIDR ranges
└── merged_phishing_data.csv   ← optional: phishing domain database
```

If optional files are missing or the paths are set to `null` in the config, the pipeline runs without enrichment features.

See [Data/README.md](Data/README.md) for the exact CSV column requirements.

## Quick Start

### Build dataset and train the best model

```bash
# 1. Build a 20k stratified dataset and train the winning LogReg model
python -m ml.train --model logreg_tfidf --class-balance effective --feature-variant value_char --tune-c --c-grid 0.5 1.0 2.0 4.0 8.0

# 2. Run the full 20k benchmark suite
python -m ml.benchmark --config configs/benchmark_20k.yaml

# 3. Use the pre-trained model for predictions
python -c "
import joblib, pandas as pd
from ml.settings import build_paths
from ml.features import augment_features
from ml.data import load_ip_sidecar, load_phishing_sidecar

paths = build_paths()
model = joblib.load('artifacts/runs/<run_name>/model.joblib')
ip = load_ip_sidecar(paths.merged_ip_txt)
phish = load_phishing_sidecar(paths.merged_phishing_csv)
df = augment_features(pd.read_csv('your_iocs.csv'), ip, phish)
preds = model.predict(df)
print(['medium','high','critical'][p] for p in preds)
"
```

### PowerShell launchers (Windows)

```powershell
# Train just the best LogReg model
powershell -ExecutionPolicy Bypass -File .\scripts\train_best_logreg.ps1

# Run 20k benchmark
powershell -ExecutionPolicy Bypass -File .\scripts\run_benchmark_20k.ps1

# Run 20k round-2 shortlist
powershell -ExecutionPolicy Bypass -File .\scripts\run_benchmark_20k_round2.ps1
```

### Bash launchers (Linux/macOS)

```bash
# Train just the best LogReg model
bash ./scripts/train_best_logreg.sh

# Run 20k benchmark
bash ./scripts/run_benchmark_20k.sh
```

## Train All Models At Once

Use `scripts/train_all.py` to train **all 16 models** in one run and get a ranked leaderboard by macro-F1.

```bash
# Install all dependencies first (needed for neural models)
pip install torch transformers sentence-transformers huggingface_hub faiss-cpu

# Train ALL 16 models on 20k rows (LogReg, XGBoost, + 4 transformers with LoRA)
python scripts/train_all.py

# Train only classical models (no GPU needed)
python scripts/train_all.py --classical-only

# Train only neural models (LoRA fine-tuning)
python scripts/train_all.py --neural-only

# Use your full dataset (no row limit)
python scripts/train_all.py --no-limit

# Re-run even completed experiments
python scripts/train_all.py --force

# Just build the dataset, skip training
python scripts/train_all.py --prepare-only

# Train specific models only
python scripts/train_all.py --experiments logreg_tfidf_valuechar_effective_tuned modernbert_lora_balanced_softmax_effective
```

### What gets trained

| Group | Models | Depends on |
|---|---|---|
| **Best LogReg** | `logreg_tfidf_valuechar_effective_tuned` | scikit-learn only |
| **LogReg variants** | effective, none, balanced class weighting | scikit-learn only |
| **XGBoost hybrid** | ModernBERT + SecureBERT2 embeddings | scikit-learn + transformers |
| **ModernBERT LoRA** | CE, weighted-CE, balanced-softmax, focal-loss | GPU recommended |
| **SecureBERT2 LoRA** | balanced-softmax, cb-focal | GPU recommended |
| **CySecBERT / SecBERT** | balanced-softmax | GPU recommended |

### Output

After training completes, results are saved to:

```
artifacts/benchmarks/full_model_competition/
├── leaderboard.md       ← ranked by macro-F1 (open this!)
├── aggregate.csv       ← all metrics per model
├── results.json        ← full per-seed results
└── status.md           ← run status
```

Each trained model is also saved individually:

```
artifacts/runs/<experiment_name>__seed42/
├── model.joblib         ← trained model artifact
├── metrics.json          ← macro-F1, ECE, Brier, etc.
└── test_predictions.npz  ← raw predictions
```

The `leaderboard.md` shows every model ranked from best to worst by macro-F1, so you always know which model won.

## Cache And Model Download Policy

All downloads stay inside the project:

- `.cache/` — HuggingFace, Transformers, PyTorch caches
- `models/` — downloaded transformer checkpoints
- `artifacts/` — datasets, runs, benchmarks

## Main Models

### Best overall model — `logreg_tfidf_valuechar_effective_tuned`

- family: sparse logistic regression + TF-IDF
- features: word TF-IDF on `text_input` + character TF-IDF on IOC `value` + categorical + numeric
- class balance: `effective`
- hyperparameter search: `C` over `0.5, 1.0, 2.0, 4.0, 8.0`
- **99.4% macro-F1, no GPU required, ~5ms per prediction**

Included artifact: `released_models/logreg_tfidf_valuechar_effective_tuned/model.joblib`

### Best neural model — `securebert2_lora_balanced_softmax_effective_epoch3_temp`

- base: `cisco-ai/SecureBERT2.0-base`
- training: LoRA, balanced softmax, `3` epochs, temperature scaling
- **98.9% macro-F1** (requires GPU)

## Current Best Results

### 20k round-2 shortlist

| Model | Macro-F1 | Balanced Acc | ECE | Brier |
|---|---|---|---|---|
| `logreg_tfidf_valuechar_effective_tuned` | 0.9943 | 0.9943 | 0.0091 | 0.0108 |
| `securebert2_lora_balanced_softmax_effective_epoch3_temp` | 0.9887 | 0.9887 | 0.0028 | 0.0184 |

## Portable Configs

All configs are path-portable — `project_root` resolves relative to the repo root:

- `configs/benchmark_20k_round2.yaml` — best classical config
- `configs/benchmark_100k_shortlist.yaml` — larger-scale config

## Repo Notes

- `Data/` contents are gitignored except `Data/README.md`
- `artifacts/`, `models/`, `.cache/` are gitignored
- only the small winning `logreg` artifact is versioned under `released_models/`

## Architecture

The pipeline is a DAG of composable stages:

```
Data/DB-ThreatIndicators.csv
  → build_dataset.py       (parse, feature engineering, train/val/test split)
    → artifacts/datasets/  (train.parquet, validation.parquet, test.parquet)
      → build_embeddings.py   (optional: transformer embeddings)
      → build_retrieval.py     (optional: FAISS KNN retrieval features)
      → train.py           (LogReg, XGBoost hybrid)
      → finetune.py        (LoRA fine-tuning, temperature calibration)
        → benchmark.py     (orchestrates all experiments, generates reports)
```

See [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) for the full technical deep-dive.