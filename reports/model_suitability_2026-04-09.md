# Model Suitability Update (2026-04-09)

## Local evidence

- Active development benchmark: balanced 20k sample recorded in `results/dataset_manifest_20k.json`
- Rows: 20,000
- Classes: `medium`, `high`, `critical` in near-perfect balance
- Sources: 10
- Types: 12
- Mean `text_input` length: 155.14 characters
- Mean description length: 37.49 characters
- Mean IOC `value` length: 29.79 characters

The text is short and highly templated. The 10 most common descriptions dominate a large share of the sample, and many rows are driven by short CTI templates plus structured IOC strings such as domains, URLs, IPs, and hashes.

That makes this task closer to:

- short structured text classification
- lexical pattern detection
- metadata-aware IOC string classification

than to:

- long-form semantic document understanding
- retrieval-heavy report reasoning
- context-heavy narrative CTI analysis

## What fits this data best

### 1. Primary model family: sparse linear text classification

For this dataset, a sparse linear model remains the most suitable primary model family.

Why:

- it is already the top performer on the main 20k benchmark
- it matches the short-template nature of the text
- it can exploit exact lexical triggers very efficiently
- it handles structured IOC strings better when character n-grams are added
- it is cheap enough to scale to much larger samples on current hardware

Current best completed baseline:

- `results/benchmark_20k/aggregate.csv`
- `logreg_tfidf_effective`: macro-F1 `0.98468`

New targeted test:

- exported winning run metrics under `released_models/logreg_tfidf_valuechar_effective_tuned/metrics.json`
- `logreg` + word TF-IDF + IOC `value` char n-grams + validation-tuned `C`
- macro-F1 `0.99434`

This is the clearest signal so far that the model benefits from stronger string-level lexical coverage rather than a larger model family.

### 2. Best neural model family: SecureBERT 2.0

If a neural model is kept, `SecureBERT 2.0` is the right one to keep.

Why:

- it already beat `ModernBERT` in the finished 20k benchmark
- it is domain-specialized for cybersecurity text
- the model card and paper position it for threat intelligence, technical extraction, and related downstream security tasks

Relevant sources:

- [SecureBERT 2.0 paper](https://arxiv.org/abs/2510.00240)
- [SecureBERT 2.0 model card](https://huggingface.co/cisco-ai/SecureBERT2.0-base)

Current completed neural result:

- `results/benchmark_20k/aggregate.csv`
- `securebert2_lora_balanced_softmax_effective`: macro-F1 `0.98334`

Targeted efficiency check already validated locally:

- final neural benchmark summaries under `results/benchmark_20k_round2/aggregate.csv`
- `SecureBERT 2.0`, LoRA, balanced softmax, temperature scaling, max length 192
- macro-F1 `0.98234`

This suggests SecureBERT remains viable, but it is not obviously the best main model for this dataset.

## Research-backed next tests

### Sparse model

Best next sparse tests:

- add character n-grams over IOC `value`
- tune regularization `C` on validation
- keep source/type and numeric metadata

This matches the capabilities exposed by scikit-learn's `TfidfVectorizer`, which supports `word`, `char`, and `char_wb` analyzers:

- [TfidfVectorizer documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

### SecureBERT

Best next SecureBERT tests on this hardware:

- temperature scaling on validation
- shorter sequence length for speed/throughput
- one extra epoch check

Temperature scaling is still the highest-value calibration test for transformer classifiers:

- [Calibration of Pre-trained Transformers](https://arxiv.org/abs/2003.07892)
- [CalibratedClassifierCV docs](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)

## Practical conclusion

Most suitable right now:

- primary model: `logreg` with sparse lexical features
- neural comparator: `SecureBERT 2.0`

Most realistic next step:

- run a small round-2 benchmark on 20k with only high-value variants
- pick the winner
- move only the winner sparse model and the winner SecureBERT setup to a much larger sample
