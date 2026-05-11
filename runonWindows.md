How to run on Windows


1. Install Python dependencies

Open PowerShell in the project root:
# Core dependencies (CPU-only)

pip install -e .

# If you also want neural models (torch/transformers)

pip install -e ".[neural]"

# If you want all dependencies + dev tools

pip install -e ".[all]"

2. Train all models

pwsh scripts/train_all_models.ps1
Or with a custom limit:
pwsh scripts/train_all_models.ps1 --limit 10000

3. Train a single model

# Classical models

pwsh scripts/train_logreg.ps1
pwsh scripts/train_xgboost.ps1

# Neural models (requires GPU + [neural] install)

pwsh scripts/train_securebert.ps1
pwsh scripts/train_phishtank_bert.ps1

4. Run benchmarks

pwsh scripts/run_benchmark_20k.ps1
pwsh scripts/run_benchmark_full.ps1

Optional: Node.js (only for severityClassifier.js)
If you have JavaScript dependencies in scripts/, install Node.js from https://nodejs.org — the scripts/metricsCollector.js stub is already included so no backend needed.