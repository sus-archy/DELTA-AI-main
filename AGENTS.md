# DELTA-AI — Agent instructions

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # core + dev (pytest, ruff, mypy)
pip install -e ".[research]"   # + CatBoost, SHAP, Captum
```

## Run commands

Always use `python -m ml.<module>` — not `python ml/<module>.py`:

```bash
python -m ml.train --model logreg_tfidf --class-balance effective --feature-variant value_char
python -m ml.benchmark --config configs/benchmark_20k.yaml
```

`PYTHONPATH` must include repo root. Shell scripts in `scripts/` set this automatically. For ad-hoc commands: `export PYTHONPATH="$PWD"`.

## Project structure

| Path | Role |
|---|---|
| `ml/` | Core Python package (installed as `ml` via setuptools) |
| `ml/settings.py` | Config loading, path resolution, cache env bootstrap |
| `ml/benchmark.py` | Orchestrator — runs experiment suite, generates reports |
| `ml/train.py` | Classical model training (LogReg, XGBoost) |
| `ml/finetune.py` | Transformer LoRA fine-tuning (PyTorch + PEFT) |
| `ml/data.py` | CSV parsing, feature engineering, dataset splitting |
| `ml/model_registry.py` | Encoder specs and HuggingFace model loading |
| `ml/teacher.py` | Heuristic severity classifier (reference / distillation teacher) |
| `scripts/` | Launchers: bash, PowerShell, `train_all.py` (multi-experiment) |
| `scripts/severityClassifier.js` | Standalone JS reference scorer (not part of Python package) |
| `configs/` | YAML benchmark configs; `default.yaml` is fallback |
| `Data/` | Place `DB-ThreatIndicators.csv` here (gitignored) |

## Architecture

Pipeline DAG:
```
DB-ThreatIndicators.csv
  → build_dataset.py    (parse, features, train/val/test split)
    → build_embeddings.py  (optional: transformer → embeddings)
    → build_retrieval.py   (optional: FAISS KNN)
    → train.py / finetune.py
      → benchmark.py   (orchestrates all runs → leaderboard.md)
```

## Data & artifacts

- `Data/DB-ThreatIndicators.csv` — required, gitignored, must be placed manually
- Everything else is derived: artifacts, embeddings, model checkpoints live under gitignored dirs
- Only `released_models/logreg_tfidf_valuechar_effective_tuned/` is versioned

## Testing & linting

- `ruff` and `mypy` listed in `[dev]` extras but **no config files exist** — no pyproject.toml `[tool.ruff]` or `[tool.mypy]` sections
- There are **zero test files** in the repo — `pytest` is listed as a dev dep but unused
- Validate simply by running: `python -m ml.train --help` (syntax check)

## Key details

- Supported severity classes: `medium`, `high`, `critical`
- Ground truth label: DB `severity` column from `DB-ThreatIndicators.csv`
- Python minimum: 3.11
- GPU is optional — only needed for LoRA fine-tuning / embeddings
- `benchmark.yaml` configs use `project_root: ..` relative to config file
- Configs are loaded from `configs/default.yaml` when no `--config` is given
