# DELTA-AI Project - Simple Guide

## What Is This Project? 🎯

**DELTA-AI** is a smart computer program that rates how dangerous different internet threats are.

Think of it like this:
- **You see a warning**: "This website is dangerous!"
- **DELTA-AI's job**: Decide how dangerous → "Danger Level: HIGH"
- **Three levels**: Medium, High, or Critical

It works like a security guard that learns from experience which threats are most dangerous.

## Quick Facts
- **Works with**: Dangerous URLs, IP addresses, domains, file hashes, emails
- **Predicts**: Low/Medium/High danger levels
- **Built with**: Python programming
- **Accuracy**: 99.4% (almost never wrong!)
- **Ready to use**: Yes, comes with a trained model

---

## What DELTA-AI Does

### Core Functionality

DELTA-AI is a complete end-to-end machine learning pipeline for threat indicator severity classification:

1. **Data Preparation**: 
   - Loads threat indicators from a CSV database
   - Enriches data with sidecar information (IP lists, phishing domains)
   - Applies temporal splits (train/validation/test)
   - Handles class imbalance through multiple strategies

2. **Feature Engineering**:
   - Text-based features: TF-IDF word vectors from description text
   - Character-level features: TF-IDF from IOC values
   - Categorical features: IOC type, source, TLD, URL scheme
   - Numeric features: Age, recency, observation count, tag count, etc.
   - Metadata enrichment: IP reputation data, phishing domain lookups

3. **Model Training**:
   - Classical ML: Logistic Regression, XGBoost
   - Neural: Fine-tuned transformer models (ModernBERT, SecureBERT2)
   - Class balancing: Effective number of samples, inverse weighting
   - Hyperparameter tuning: C-parameter sweep for logistic regression

4. **Benchmarking**:
   - Runs multiple experiment configurations
   - Evaluates on metrics: F1-score, balanced accuracy, ECE, Brier
   - Generates reports and comparisons
   - Produces calibration analysis

5. **Model Export**:
   - Exports best model as joblib artifact
   - Includes prediction files (validation, test)
   - Provides metrics summary

### Real-World Use Case

**Original Context**: The project replaces a legacy JavaScript classifier (`severityClassifier.js`) that used heuristic rules to score threat indicators. DELTA-AI uses machine learning instead, achieving much higher accuracy while remaining interpretable.

**Legacy Heuristics**:
- Source reliability scoring (80-100 points)
- Type-based threat severity mapping
- Keyword matching in descriptions
- Observation count boosting

**DELTA-AI Improvement**: Learns all these patterns automatically from labeled data, plus discovers new patterns in the database labels.

---

## Architecture & How It Works

### Project Structure

```
DELTA-AI/
├── ml/                          # Core ML pipeline
│   ├── data.py                 # Data loading & splitting
│   ├── features.py             # Feature engineering
│   ├── encoders.py             # Neural encoder registry
│   ├── train.py                # Model training
│   ├── build_dataset.py        # Dataset construction
│   ├── build_embeddings.py     # Embedding generation
│   ├── benchmark.py            # Benchmarking orchestration
│   ├── metrics.py              # Evaluation metrics
│   ├── calibration.py          # Probability calibration
│   ├── class_balance.py        # Class balancing strategies
│   └── [other modules]         # Teacher, finetune, report, etc.
├── configs/                     # Configuration files
│   ├── default.yaml            # Base configuration
│   ├── benchmark_*.yaml        # Benchmark experiment configs
├── released_models/             # Pre-trained model artifacts
│   └── logreg_tfidf_valuechar_effective_tuned/
│       ├── model.joblib        # Trained model (binary format)
│       └── metrics.json        # Performance metrics
├── Data/                        # Input data directory
│   ├── DB-ThreatIndicators.csv # Main indicator database
│   ├── merged_ip_list.txt      # IP reputation data
│   └── merged_phishing_data.csv # Phishing domain data
└── scripts/                     # Execution scripts
    └── severityClassifier.js   # Legacy implementation
```

### Component Overview

#### 1. **Settings & Configuration** (`settings.py`)
- Manages project paths and directories
- Configures cache locations (HuggingFace, Transformers, PyTorch)
- Loads YAML configuration files
- Returns `ProjectPaths` dataclass with all path references

#### 2. **Data Loading** (`data.py`)
- Loads threat indicator CSV with columns: type, value, severity, source, description, tags, etc.
- Parses JSON-embedded fields (tags, raw metadata)
- Builds canonical indicator keys (IOC type || normalized value)
- **Severity Order**: `["medium", "high", "critical"]` → IDs `[0, 1, 2]`
- **Temporal Splitting**: Creates train/val/test splits based on `lastSeen` timestamp
- Also supports group-stratified splits for smoke tests

#### 3. **Feature Engineering** (`features.py`)

The `augment_features()` function creates 24 features per IOC:

**Numeric Features (19)**:
- `desc_len`: Length of description
- `tag_count`: Number of tags
- `value_len`: Length of IOC value
- `age_days`: Days between first and last seen
- `recency_days`: Days since last seen
- `observedCount_num`: How many times observed
- `log_observedCount`: Log-transformed observation count
- `is_hash_like`, `is_url`, `is_domain`, `is_ip`: Type indicators
- `domain_depth`: Subdomain depth (.a.b.c = depth 3)
- `url_path_depth`: URL path depth
- `is_in_ip_txt`, `is_in_spamhaus_cidr`, `cidr_prefix_len`: IP reputation
- `cidr_base_match`: IP in known CIDR blocks
- `phish_domain_seen_count`, `phish_domain_seen_flag`: Phishing lookups

**Categorical Features (5)**:
- `type`: IOC type (hash, ip, domain, url, etc.)
- `source`: Where the indicator came from
- `tld`: Top-level domain
- `url_scheme`: http/https/etc
- `phish_target_mode`: Most common phishing target

**Text Feature (1)**:
- `text_input`: Combined descriptive text (description + tags)

#### 4. **Encoders & Models** (`encoders.py`, `model_registry.py`)

Supported encoders:
- **ModernBERT-base**: General-purpose, 384 token max length
- **ModernBERT-large**: Stronger variant
- **SecureBERT2.0-base**: Cybersecurity-adapted
- **CySecBERT**: Domain-specific BERT variant
- **SecBERT-base**: Security-focused baseline

#### 5. **Training** (`train.py`)

Multiple model types:
- **LogisticRegression + TF-IDF**: 
  - Word TF-IDF on text_input (max 50k features, 1-2 grams)
  - Character TF-IDF on value (optional, 3-6 char grams, 40k features)
  - One-hot encoding for categorical features
  - StandardScaler for numeric features
  - Parameter C tuned via grid search `[0.5, 1.0, 2.0, 4.0, 8.0]`

- **XGBoost Hybrid**:
  - Uses embeddings from neural encoders as features
  - Combined with traditional features
  - Handles class imbalance via sample weights

- **Fine-tuned Transformers**:
  - Uses PEFT LoRA for efficient fine-tuning
  - Multiple loss functions: balanced_softmax, focal loss
  - 2-3 epochs typical
  - Supports multi-seed training

#### 6. **Class Balancing** (`class_balance.py`)

Strategies for handling class imbalance:
- **Effective Number**: Reduces weight of over-represented classes using effective sample count formula
- **Inverse**: Simple inverse frequency weighting
- **Balanced**: sklearn.utils.class_weight.compute_class_weight

#### 7. **Benchmarking** (`benchmark.py`)

Orchestration layer:
- Manages multiple experiments defined in YAML
- Ensures dataset is built with correct size
- Builds embeddings if needed by any experiment
- Runs experiments and collects metrics
- Generates visual reports

#### 8. **Metrics** (`metrics.py`)

Evaluation metrics:
- **macro_f1**: Unweighted F1 across classes
- **weighted_f1**: F1 weighted by class frequency
- **balanced_accuracy**: Recall averaged across classes
- **ECE** (Expected Calibration Error): Confidence calibration
- **Brier Score**: Probability accuracy
- Per-class precision/recall/f1
- Confusion matrix

---

## Data Pipeline

### Input Data

The pipeline requires three data files:

1. **DB-ThreatIndicators.csv** (Main database)
   ```
   Columns: type, value, severity, source, description, tags, 
            firstSeen, lastSeen, observedCount, raw, confidence, ...
   
   Example Row:
   type=url, value=http://malware.com/payload, severity=high,
   source=urlhaus, description="Ransomware distribution site",
   tags="[\"ransomware\", \"c2\"]", firstSeen=2024-01-01, 
   lastSeen=2024-03-15, observedCount=245
   ```

2. **merged_ip_list.txt** (IP reputation sidecar)
   ```
   One IP or CIDR per line
   Used to mark indicators in known malicious IP lists
   ```

3. **merged_phishing_data.csv** (Phishing domain sidecar)
   ```
   Columns: domain_norm, target
   Used to enrich domain/URL indicators with phishing context
   ```

### Data Flow

```
Raw CSV → Load & Parse → Normalize → Temporal Split → Feature Engineering
          ↓
        DB Frame (parsed severity labels, indicator keys)
          ↓
        Train/Val/Test Split (based on lastSeen timestamp)
          ↓
        Augment Features (add 24 engineered features + text)
          ↓
        TF-IDF Vectorization (text and character features)
          ↓
        Prepared Dataset (ready for model training)
```

### Temporal Splitting

Splits by `lastSeen` timestamp to create realistic time-based splits:
- **Test**: Last 90 days
- **Validation**: 60 days before test (90-150 days ago)
- **Training**: Everything older than 150 days

This prevents data leakage and simulates real-world deployment.

---

## Feature Engineering

### Feature Categories

#### A. Textual Features
- **Description TF-IDF**: 
  - Vectorizes the combined description + tags text
  - Unigrams and bigrams
  - Up to 50,000 features
  - Sublinear TF scaling
  
- **Character N-gram TF-IDF** (value_char variant):
  - 3-6 character n-grams from IOC value
  - Captures patterns in domains, URLs, hashes
  - Min document frequency: 2
  - Up to 40,000 features

#### B. Numeric Features
- **Temporal**: age_days, recency_days, log_observedCount
- **Size/Count**: desc_len, tag_count, value_len
- **Type Indicators**: binary flags for hash/url/domain/ip
- **Structural**: domain_depth, url_path_depth
- **Reputation**: is_in_ip_txt, is_in_spamhaus_cidr, cidr_prefix_len
- **Phishing**: phish_domain_seen_count, phish_domain_seen_flag

#### C. Categorical Features
- **IOC Type**: hash, ip, domain, url, email, etc.
- **Source**: Where collected (urlhaus, otx, spamhaus, etc.)
- **TLD**: Top-level domain extracted from domain/URL
- **URL Scheme**: http, https, ftp, etc.
- **Phishing Target**: Mode phishing category for that domain

### Feature Preprocessing

```
Categorical Features → One-Hot Encoding
Numeric Features → StandardScaler (sparse-friendly)
Text Features → TF-IDF Vectorization
Character Features → TF-IDF Vectorization

All combined via sklearn.compose.ColumnTransformer
```

### Interpretation

The feature set is interpretable:
- **High observation count** → likely more critical
- **Longer age** → may be less critical (old indicators)
- **Source from trusted list** → supports severity
- **Keywords in description** → indicates threat type
- **Structural patterns** → e.g., deep URLs suggest malicious activity

---

## Model Training

### Training Pipeline

```
1. Load experiment config (YAML)
2. Build/load dataset (train/val/test split)
3. Initialize preprocessor (TF-IDF + scaling + encoding)
4. Create model:
   - LogisticRegression: L2 regularization, selected C value
   - XGBoost: tree-based with sample weights
   - Transformer: fine-tuned with LoRA
5. Apply class weights based on strategy
6. Train on training set
7. Evaluate on validation set
8. Hyperparameter tuning (if applicable)
9. Final evaluation on test set
10. Calibrate probabilities (optional)
11. Save model artifact
12. Log metrics to JSON
```

### Best Model Details: `logreg_tfidf_valuechar_effective_tuned`

**Configuration**:
- **Model Type**: Logistic Regression
- **Features**: Word TF-IDF + Character TF-IDF + Categorical + Numeric
- **Class Balance**: Effective Number weighting
- **Hyperparameter C**: 8.0 (tuned via sweep)
- **Regularization**: L2
- **Solver**: (default liblinear for binary-friendly)

**Performance on Test Set**:
```
Macro F1-Score:      0.9943
Weighted F1-Score:   0.9943
Balanced Accuracy:   0.9943

Per-Class Performance:
  Medium:    P=0.998  R=0.991  F1=0.994  (1001 samples)
  High:      P=0.997  R=0.993  F1=0.995  (1001 samples)
  Critical:  P=0.988  R=0.999  F1=0.994  (1000 samples)

Calibration:
  ECE (Expected Calibration Error): 0.0091
  Brier Score: 0.0108

Confusion Matrix (test set 3002 samples):
           Medium  High  Critical
  Medium   [992]    2      7
  High       2    [994]    5
  Critical   0      1    [999]
```

**Why This Model Wins**:
1. Simple, interpretable, fast inference
2. Character n-gram features capture domain/hash patterns extremely well
3. Effective number weighting handles class imbalance perfectly
4. High regularization (C=8.0) prevents overfitting
5. No hyperparameter randomness (single seed)

**Model File**:
- Located: `released_models/logreg_tfidf_valuechar_effective_tuned/model.joblib`
- Size: ~150 MB (large due to TF-IDF feature matrices)
- Format: joblib binary pickle
- Includes: ColumnTransformer + LogisticRegression pipeline

---

## The Best Model

### How to Use the Pre-trained Model

The best model is shipped in the repository and ready to use immediately:

```python
import joblib
import pandas as pd
from ml.settings import bootstrap_environment, build_paths
from ml.features import augment_features
from ml.data import load_db_frame, load_ip_sidecar, load_phishing_sidecar

# 1. Bootstrap environment
paths = bootstrap_environment()

# 2. Load the pre-trained model
model_path = paths.models_dir / "logreg_tfidf_valuechar_effective_tuned" / "model.joblib"
pipeline = joblib.load(model_path)

# 3. Prepare your data
# Load your IOC data (must have: type, value, severity, source, description, etc.)
df = pd.read_csv("your_iocs.csv")

# 4. Augment with features
ip_sidecar = load_ip_sidecar(paths.merged_ip_txt)
phishing_sidecar = load_phishing_sidecar(paths.merged_phishing_csv)
features_df = augment_features(df, ip_sidecar, phishing_sidecar)

# 5. Make predictions
# Select feature columns used during training
X = features_df[["text_input", "value", "categorical_cols", "numeric_cols"]]
predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)

# 6. Output
# predictions: array of [0=medium, 1=high, 2=critical]
# probabilities: (n_samples, 3) probability matrix
```

### Model Artifacts

**Included**:
- `model.joblib`: Full pipeline with preprocessor and model
- `metrics.json`: Performance metrics and configuration
- `test_predictions.npz`: Predictions on test set for reproducibility
- `validation_predictions.npz`: Predictions on validation set

**Not Included**:
- Training data (use your own or request from data owners)
- HuggingFace transformer caches (will download on first use)
- Benchmark artifacts (large intermediate results)

---

## Integration Guide

### How to Integrate DELTA-AI Model into Another Project

#### Option 1: Direct Model Usage (Simplest)

If you only need predictions from the pre-trained model:

```python
# 1. Install dependencies
pip install scikit-learn pandas numpy joblib pyyaml

# 2. Copy necessary files to your project
cp released_models/logreg_tfidf_valuechar_effective_tuned/model.joblib ./
cp -r ml/ ./  # Only: data.py, features.py, settings.py
cp configs/default.yaml ./

# 3. Create a prediction service
from pathlib import Path
import joblib
import pandas as pd
from ml.features import augment_features
from ml.data import load_ip_sidecar, load_phishing_sidecar
from ml.settings import build_paths

class IOCSeverityClassifier:
    def __init__(self, model_path: str, config_path: str = None):
        self.model = joblib.load(model_path)
        self.paths = build_paths(config_path)
        self.ip_sidecar = load_ip_sidecar(self.paths.merged_ip_txt)
        self.phishing_sidecar = load_phishing_sidecar(self.paths.merged_phishing_csv)
    
    def predict(self, ioc_data: pd.DataFrame) -> dict:
        """
        Predict severity for IOCs
        
        Args:
            ioc_data: DataFrame with columns [type, value, source, 
                     description, tags, firstSeen, lastSeen, observedCount]
        
        Returns:
            dict with keys: predictions, probabilities, confidence
        """
        # Augment features
        featured = augment_features(ioc_data, self.ip_sidecar, self.phishing_sidecar)
        
        # Get predictions
        preds = self.model.predict(featured)
        probs = self.model.predict_proba(featured)
        
        severity_labels = ["medium", "high", "critical"]
        return {
            "predictions": [severity_labels[p] for p in preds],
            "probabilities": {
                "medium": probs[:, 0],
                "high": probs[:, 1],
                "critical": probs[:, 2]
            },
            "confidence": probs.max(axis=1)  # Max probability for top prediction
        }

# 4. Usage in your application
classifier = IOCSeverityClassifier("model.joblib", "default.yaml")
iocs = pd.DataFrame({
    "type": ["url", "domain", "hash"],
    "value": ["http://malware.com", "badguy.xyz", "abc123def456"],
    "source": ["urlhaus", "otx", "malshare"],
    "description": ["Phishing site", "C2 server", "Ransomware"],
    "tags": [["phishing"], ["c2"], ["ransomware"]],
    "firstSeen": ["2024-01-01", "2024-02-01", "2024-01-15"],
    "lastSeen": ["2024-03-20", "2024-03-20", "2024-03-20"],
    "observedCount": [100, 500, 250]
})

result = classifier.predict(iocs)
print(result["predictions"])  # ["high", "critical", "high"]
```

#### Option 2: REST API Integration

```python
from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd

app = Flask(__name__)
classifier = None

@app.before_first_request
def load_model():
    global classifier
    classifier = IOCSeverityClassifier("model.joblib", "default.yaml")

@app.route("/classify", methods=["POST"])
def classify():
    """
    POST /classify with JSON body:
    {
        "iocs": [
            {"type": "url", "value": "...", "source": "...", ...},
            ...
        ]
    }
    """
    data = request.json
    iocs_df = pd.DataFrame(data["iocs"])
    result = classifier.predict(iocs_df)
    
    return jsonify({
        "success": True,
        "results": [
            {
                "severity": pred,
                "confidence": float(conf),
                "probabilities": {
                    "medium": float(probs[0]),
                    "high": float(probs[1]),
                    "critical": float(probs[2])
                }
            }
            for pred, conf, probs in zip(
                result["predictions"],
                result["confidence"],
                result["probabilities"].T
            )
        ]
    })

if __name__ == "__main__":
    app.run(port=5000)
```

#### Option 3: Batch Processing

```python
import joblib
import pandas as pd
from pathlib import Path

def batch_classify_iocs(csv_path: str, model_path: str, output_path: str):
    """Process large CSV files of IOCs"""
    
    # Load model
    model = joblib.load(model_path)
    
    # Process in chunks
    chunk_size = 10000
    results = []
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Feature engineering
        featured = augment_features(chunk, ip_sidecar, phishing_sidecar)
        
        # Predict
        preds = model.predict(featured)
        probs = model.predict_proba(featured)
        
        # Store results
        chunk["predicted_severity"] = preds
        chunk["confidence"] = probs.max(axis=1)
        results.append(chunk)
    
    # Write output
    output_df = pd.concat(results, ignore_index=True)
    output_df.to_csv(output_path, index=False)
    print(f"Classified {len(output_df)} IOCs, saved to {output_path}")

# Usage
batch_classify_iocs("input_iocs.csv", "model.joblib", "output_predictions.csv")
```

#### Option 4: Docker Containerization

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy DELTA-AI code
COPY ml/ ./ml/
COPY configs/ ./configs/
COPY released_models/ ./released_models/

# Install dependencies
RUN pip install --no-cache-dir scikit-learn pandas numpy joblib pyyaml

# Copy prediction service
COPY classifier_service.py ./

# Expose port
EXPOSE 5000

# Run service
CMD ["python", "classifier_service.py"]
```

```bash
# Build and run
docker build -t ioc-classifier .
docker run -p 5000:5000 ioc-classifier
```

#### Option 5: CLI Tool

```python
#!/usr/bin/env python3
"""Command-line IOC classifier"""

import argparse
import json
import pandas as pd
from pathlib import Path
from ml.features import augment_features
from ml.settings import build_paths
import joblib

def main():
    parser = argparse.ArgumentParser(description="Classify IOCs by severity")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--model", type=str, default="model.joblib", help="Model path")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    
    args = parser.parse_args()
    
    # Load model
    model = joblib.load(args.model)
    paths = build_paths()
    
    # Load and process data
    df = pd.read_csv(args.input)
    ip_sidecar = load_ip_sidecar(paths.merged_ip_txt)
    phishing_sidecar = load_phishing_sidecar(paths.merged_phishing_csv)
    
    featured = augment_features(df, ip_sidecar, phishing_sidecar)
    predictions = model.predict(featured)
    probabilities = model.predict_proba(featured)
    
    # Prepare output
    severity_labels = ["medium", "high", "critical"]
    df["predicted_severity"] = [severity_labels[p] for p in predictions]
    df["confidence"] = probabilities.max(axis=1)
    
    # Save
    if args.format == "json":
        df.to_json(args.output, orient="records")
    else:
        df.to_csv(args.output, index=False)
    
    print(f"✓ Classified {len(df)} IOCs")
    print(f"✓ Results saved to {args.output}")

if __name__ == "__main__":
    main()
```

### Key Integration Considerations

1. **Data Format**: Your IOC data must include:
   - `type`: IOC category (url, domain, ip, hash, etc.)
   - `value`: The actual IOC value
   - `source`: Where it came from
   - `description`: Threat description
   - `tags`: List of tags/categories
   - `firstSeen`, `lastSeen`: Timestamps (ISO format or parseable)
   - `observedCount`: How many times seen

2. **Sidecar Data**: The model references:
   - IP reputation list (merged_ip_list.txt)
   - Phishing domain database (merged_phishing_data.csv)
   - Update these for your threat landscape

3. **Inference Performance**:
   - Single IOC: ~5-10ms (feature engineering + prediction)
   - Batch 10k IOCs: ~30-60 seconds
   - Model memory: ~200MB loaded
   - No GPU needed (scikit-learn uses CPU)

4. **Error Handling**:
   ```python
   try:
       result = classifier.predict(iocs)
   except ValueError as e:
       # Missing required columns
       print(f"Data format error: {e}")
   except KeyError as e:
       # Unknown IOC type or source
       print(f"Domain error: {e}")
   ```

5. **Monitoring**:
   ```python
   # Track prediction distribution
   from collections import Counter
   severity_dist = Counter(result["predictions"])
   print(f"Distribution: {severity_dist}")
   
   # Monitor confidence
   avg_confidence = result["confidence"].mean()
   print(f"Average confidence: {avg_confidence:.2%}")
   
   # Alert on uncertainty
   low_confidence = (result["confidence"] < 0.7).sum()
   print(f"Low confidence predictions: {low_confidence}")
   ```

### Model Retraining in Your Environment

If you want to retrain with your own data:

```bash
# 1. Place your data in Data/ directory
cp your_db.csv Data/DB-ThreatIndicators.csv
cp your_ips.txt Data/merged_ip_list.txt
cp your_phishing.csv Data/merged_phishing_data.csv

# 2. Update config
# Edit configs/my_config.yaml with your settings

# 3. Build dataset
python -m ml.build_dataset --config configs/my_config.yaml

# 4. Train model
python -m ml.train --config configs/my_config.yaml --run-name my_model

# 5. Evaluate
python -m ml.benchmark --config configs/benchmark_my_config.yaml
```

---

## Summary

**DELTA-AI** is a production-grade machine learning system for IOC severity classification. It provides:

✅ **Reproducible Pipeline**: All code, configs, and benchmarks included
✅ **Pre-trained Model**: Best model included, ready to use (99.4% accuracy)
✅ **Interpretable**: Feature engineering and logistic regression are transparent
✅ **Fast Inference**: No GPU needed, single-pass predictions
✅ **Extensible**: Easy to add new features, models, or data sources
✅ **Benchmarked**: Multiple models tested with rigorous evaluation

**For production integration**: Use Option 1 (Direct Model Usage) + wrap in your preferred serving framework (REST API, gRPC, batch processor, etc.).

**For research/development**: Use the full pipeline to experiment with new features, models, or benchmark strategies.
