# Final Benchmark Protocol

## Canonical Target

- Use the DB `severity` column from `Data/DB-ThreatIndicators.csv` as the operational label.
- Treat the current JavaScript scorer as a useful reference implementation, but not as the benchmark target for this stage.
- Report all model performance against DB labels only.

## Evaluation Goal

The benchmark should answer four questions:

1. How strong are simple label-policy baselines?
2. How much do text-aware and hybrid models improve over lookup and sparse-text baselines?
3. Which class-imbalance strategy is most stable under temporal drift?
4. Which encoder family is strongest for IOC severity prediction on this dataset?

## Split Policy

- Use temporal evaluation only.
- Train on older rows, validate on the next recent block, test on the newest block.
- Keep the current split logic grouped by canonical `indicator_key` so the same IOC value does not leak across splits.
- Primary split parameters:
  - validation window: `60` days
  - test window: `90` days

## Seeds And Reproducibility

- Use `3` seeds for the main benchmark:
  - `42`
  - `1337`
  - `2027`
- Fix:
  - Python hash seed
  - `random`
  - `numpy`
  - `torch`
  - `torch.cuda`
  - model-level random states where supported
- The benchmark runner is resumable and writes:
- `artifacts/benchmarks/<benchmark_name>/state.json`
  - `events.jsonl`
  - `results.json`
  - `results.csv`
  - `aggregate.csv`
  - `leaderboard.md`

## Primary Metrics

- `macro_f1`
- `weighted_f1`
- `balanced_accuracy`
- `critical` recall
- `critical` F1
- `ece`
- `brier`

Primary ranking metric:

- `macro_f1`

Tie-breakers:

- `balanced_accuracy`
- `critical` recall
- `ece`

## Benchmark Families

### Baselines

- `source_majority`
- `type_majority`
- `source_type_majority`
- `logreg_tfidf_none`
- `logreg_tfidf_effective`

### Hybrid baselines

- `xgboost_hybrid_modernbert_effective`
- `xgboost_hybrid_securebert2_effective`

### Neural main models

- `modernbert_lora_ce_none`
- `modernbert_lora_weighted_ce_effective`
- `modernbert_lora_balanced_softmax_effective`
- `modernbert_lora_cb_focal_effective`

### Encoder ablations

- `securebert2_lora_balanced_softmax_effective`
- `securebert2_lora_cb_focal_effective`
- `cysecbert_lora_balanced_softmax_effective`
- `secbert_lora_balanced_softmax_effective`

### Compute-heavy optional ablation

- `modernbert_full_balanced_softmax_effective`

## Recommended Defaults

### Classical

- TF-IDF logistic regression with `effective-number` weighting

### Hybrid

- XGBoost with retrieval features and `effective-number` sample weighting

### Neural

- `ModernBERT + LoRA + balanced_softmax + effective priors`

### Cyber encoder comparison

- Compare `ModernBERT` against:
  - `SecureBERT 2.0`
  - `CySecBERT`
  - `SecBERT`

## Why These Defaults

- Effective-number weighting is milder and usually more stable than raw inverse-frequency weighting on moderate long-tail data.
- Balanced Softmax / logit-adjusted training is a stronger default than focal-style losses when labels are weak and calibration matters.
- Focal-style losses are still worth testing, but they should be treated as recall-oriented ablations, not the default.
- ModernBERT is the strongest general encoder in the stack; SecureBERT 2.0 is the most relevant cybersecurity-specific comparison.

## Ablation Rules

- Only vary one major axis at a time in the named ablations:
  - imbalance strategy
  - neural loss
  - encoder
  - PEFT vs full fine-tuning
- Keep the split policy, metrics, and seed set fixed across all runs.

## Logging And Resumability

- The benchmark runner must:
  - skip completed experiment-seed pairs when resuming
  - rebuild the dataset only when the manifest limit does not match the requested setup
  - skip embeddings and retrieval features when cached files already exist
  - write an event log entry for every stage transition and failure
- This is implemented in `ml/benchmark.py`.

## Final Research Position

For this project, the strongest defensible benchmark story is:

- DB labels as the canonical supervised target
- temporal evaluation with grouped leakage control
- seeded comparisons across lookup, sparse-text, hybrid, and neural systems
- explicit long-tail handling through effective-number weighting and Balanced Softmax
- encoder ablations between general and cybersecurity-specific BERT families

## References

- Class-Balanced Loss Based on Effective Number of Samples
  - https://arxiv.org/abs/1901.05555
- Long-tail learning via logit adjustment
  - https://arxiv.org/abs/2007.07314
- Balanced Meta-Softmax for Long-Tailed Visual Recognition
  - https://arxiv.org/abs/2007.10740
- Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
  - https://arxiv.org/abs/1906.07413
- On Calibration of Modern Neural Networks
  - https://arxiv.org/abs/1706.04599
- When Does Label Smoothing Help?
  - https://arxiv.org/abs/1906.02629
