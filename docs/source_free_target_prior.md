# Source-free target-prior correction

`neureptrace.decoding.source_free_target_prior` adds an optional target-prior
correction wrapper for source-free OpenNeuro Protocol 2 / 2.5 style adaptation.
The correction is intended for held-out target batches where the fixed
source-trained model is biased toward source-fold class priors.

```python
from neureptrace.decoding.source_free_target_prior import (
    fit_source_free_target_prior_predict_proba,
)

result = fit_source_free_target_prior_predict_proba(
    source_model=source_model,
    target_features=X_target_unlabeled,
    confidence_threshold=0.75,
    max_iterations=5,
    target_prior_correction="balanced",
    target_prior_strength=1.0,
)

probabilities = result.probabilities
metadata = result.metadata
```

## Protocol hygiene

The wrapper first runs the existing source-free adapter, then estimates the
marginal predicted class distribution from the same unlabeled target batch. In
`balanced` mode it divides rows by that predicted marginal prior and
renormalizes probabilities.

The correction uses:

- a fitted source model,
- unlabeled target features,
- source-model / source-free predicted target probabilities.

The correction does **not** use:

- original source feature rows during target adaptation,
- source labels during target adaptation,
- target labels.

Metadata records:

- `source_free_target_prior_correction`
- `source_free_target_prior_strength`
- `source_free_target_class_prior`
- `source_free_target_prior_uses_target_labels=False`
- `source_free_target_prior_uses_source_rows=False`
- `source_free_valid_for_benchmark=True`

## Suggested OpenNeuro use

Use this as a lightweight follow-up knob for OpenNeuro LOSO folds where the
source-free adapter collapses to source-prior-dominated predictions or where a
balanced-accuracy endpoint is more important than preserving the raw source
posterior marginal. Keep `target_prior_strength` below `1.0` when the target
class marginal is expected to be genuinely imbalanced.
