# %% [markdown]
# # DELTA-AI — Kaggle Notebook
# Paste or upload this notebook to Kaggle.
# 1. Go to https://kaggle.com/new-notebook
# 2. Add your data: `Data/DB-ThreatIndicators.csv` (or .txt / .json)
# 3. Set `HF_TOKEN` (if using SecureBERT): `os.environ["HF_TOKEN"] = "hf_..."`
# 4. Run all cells

# %% setup
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")

# Clone repo (or upload zip)
if not os.path.exists("DELTA-AI-main"):
    subprocess.run(["git", "clone", "https://github.com/YOUR_ORG/DELTA-AI-main"], check=True)
os.chdir("DELTA-AI-main")
sys.path.insert(0, os.getcwd())

# Install deps (skip torch — Kaggle has its own)
subprocess.run(["pip", "install", "-e", ".[dev]", "--no-deps"], check=True)
subprocess.run(["pip", "install", "matplotlib", "scipy", "scikit-learn", "xgboost", "joblib", "tqdm", "rich"], check=True)

# For neural models (optional — remove if only classical):
# subprocess.run(["pip", "install", "transformers", "datasets", "peft", "sentence-transformers", "faiss-cpu"], check=True)

# HF login for gated models (SecureBERT):
# hf_token = os.environ.get("HF_TOKEN")
# if hf_token:
#     from huggingface_hub import login
#     login(token=hf_token)

# %% build dataset
!python -m ml.build_dataset --limit 0

# %% train a classical model
!python -m ml.train --model logreg_tfidf --class-balance effective --feature-variant value_char

# %% train XGBoost hybrid (needs pre-built embeddings)
# !python -m ml.build_embeddings --encoder modernbert-base
# !python -m ml.train --model xgboost_hybrid --encoder modernbert-base --class-balance effective

# %% fine-tune a transformer (GPU required)
# !python -m ml.finetune --encoder modernbert-base --loss balanced_softmax --class-balance effective --epochs 2

# %% run full benchmark
# !python -m ml.benchmark --config configs/benchmark_20k.yaml

# %% review plots
from pathlib import Path
for p in sorted(Path("artifacts/runs").glob("*/plots/*.png")):
    print(p)
