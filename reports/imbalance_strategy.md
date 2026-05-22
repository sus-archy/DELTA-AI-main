# Imbalance Strategy Recommendation

## Current Data Signal

- Full DB labels are effectively three-class right now: `medium`, `high`, `critical`.
- The train split is imbalanced but not ultra-extreme. In the current smoke build:
  - `medium`: 1758 (`50.23%`)
  - `high`: 1513 (`43.23%`)
  - `critical`: 229 (`6.54%`)
- The newest temporal slices are more skewed toward `high`, so class imbalance and temporal drift are coupled.
- Labels are weak labels inherited from the rule-based teacher, so aggressive tail amplification can also amplify teacher noise.

## Recommendation

### Classical models

- Keep `inverse-frequency` weighting as the baseline.
- Use `effective-number` weighting as the stronger default for tabular baselines.
- Avoid oversampling and SMOTE for the mainline pipeline.

Why:

- Effective-number weights are milder than raw inverse-frequency weights and are better matched to moderate long-tail settings.
- Synthetic oversampling is a poor fit for mixed IOC text, categorical metadata, and temporal features.
- Resampling would also distort the real feed priors and the time-aware evaluation protocol.

### Neural models

- Use `balanced_softmax` as the default long-tail loss for transformer fine-tuning.
- Keep `class-balanced focal loss` as the recall-heavy ablation.
- Keep plain `cross-entropy` and `weighted cross-entropy` as baselines.
- Apply post-hoc `temperature scaling` on the validation split for any model whose probabilities will be interpreted as confidence.

Why:

- Balanced Softmax and logit-adjusted training are stronger long-tail multiclass choices than plain inverse weighting when the goal includes macro metrics and minority recall.
- Focal-style losses can help the tail, but they focus on hard examples; with weak labels this can over-focus noisy cases, so they are better as ablations than as the default.
- Temperature scaling is still the cleanest first calibration step once the classifier is trained.

## What We Implemented

- `ml/class_balance.py`
  - inverse-frequency weights
  - effective-number weights
  - empirical class priors
- `ml/train.py`
  - `--class-balance none|inverse|effective`
  - logistic regression uses `class_weight`
  - XGBoost uses per-row `sample_weight`
- `ml/finetune.py`
  - `--loss ce|weighted_ce|balanced_softmax|focal|cb_focal`
  - `--class-balance none|inverse|effective`
  - local `Trainer` compatibility fixes
  - run metadata now records the loss and balance strategy

## Smoke Results

These are only smoke-scale indicators, not publication numbers.

- `logreg_tfidf_inverse`
  - macro-F1: `0.2988`
  - balanced accuracy: `0.6104`
  - critical recall: `0.4706`
- `logreg_tfidf_effective`
  - macro-F1: `0.2988`
  - balanced accuracy: `0.6104`
  - critical recall: `0.4706`
  - slightly better Brier score than inverse weighting in this smoke run
- `xgboost_hybrid_modernbert-base_inverse`
  - macro-F1: `0.1398`
  - balanced accuracy: `0.2550`
- `xgboost_hybrid_modernbert-base_effective`
  - macro-F1: `0.1371`
  - balanced accuracy: `0.2455`
- `modernbert-base_lora_balanced_softmax_effective`
  - pipeline runs successfully on smoke
  - the current `64`-row training smoke is too small for meaningful model selection
  - still, it remained much more stable than focal on this toy run
- `modernbert-base_lora_cb_focal_effective`
  - macro-F1: `0.0071`
  - balanced accuracy: `0.1569`
  - critical recall: `0.4706`
  - collapsed toward minority-heavy predictions on the tiny smoke split
  - good as a tail-recall ablation, not a safe default under weak labels

## Recommended Experiment Matrix

1. Classical baselines
   - `logreg_tfidf` with `none`, `inverse`, `effective`
   - `xgboost_hybrid` with `none`, `inverse`, `effective`
2. Transformer long-tail losses
   - `ce`
   - `weighted_ce + effective`
   - `balanced_softmax`
   - `cb_focal + effective`
3. Calibration
   - raw probabilities
   - temperature-scaled probabilities
4. Reporting
   - macro-F1
   - balanced accuracy
   - per-class recall
   - critical recall/F1
   - ECE
   - Brier score

## Literature Used

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

## Bottom Line

- Best current default for this project:
  - classical: `effective-number weighting`
  - transformer: `balanced_softmax`
  - confidence: `temperature scaling`
- Best ablation for tail recall:
  - `class-balanced focal loss`
