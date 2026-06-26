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
- `source_free_target_prior_estimator`
- `source_free_target_prior_smoothing`
- `source_free_target_prior_floor`
- `source_free_target_raw_class_prior`
- `source_free_target_class_prior`
- `source_free_target_prior_uses_target_labels=False`
- `source_free_target_prior_uses_source_rows=False`
- `source_free_valid_for_benchmark=True`

## Robust OpenNeuro settings

The raw predicted target marginal can be noisy on small OpenNeuro LOSO folds,
especially when a source model nearly collapses to one pseudo-class. Use the
robust prior options as a first Protocol 2.5 follow-up before increasing model
complexity:

```python
result = fit_source_free_target_prior_predict_proba(
    source_model=source_model,
    target_features=X_target_unlabeled,
    max_iterations=5,
    pseudo_label_selection="balanced_topk",
    balanced_topk_per_class=4,
    target_prior_correction="balanced",
    target_prior_strength=0.75,
    target_prior_estimator="entropy_weighted",
    target_prior_smoothing=0.25,
    target_prior_floor=0.02,
)
```

Available prior estimators:

- `mean`: average all target posterior rows equally;
- `confidence_weighted`: upweight rows with high maximum posterior;
- `entropy_weighted`: upweight low-entropy target posterior rows.

`target_prior_smoothing` blends the estimated target prior toward a uniform prior
before correction. `target_prior_floor` lower-bounds each class prior before
renormalization. These two guards make the correction less aggressive when a fold
contains an extreme or unreliable unlabeled predicted class marginal.

## Suggested OpenNeuro use

Use this as a lightweight follow-up knob for OpenNeuro LOSO folds where the
source-free adapter collapses to source-prior-dominated predictions or where a
balanced-accuracy endpoint is more important than preserving the raw source
posterior marginal. Keep `target_prior_strength` below `1.0` when the target
class marginal is expected to be genuinely imbalanced.

For Protocol 2.5-style reporting, keep the default `mean`/unsmoothed row as the
baseline and report the robust row separately, because the robust row changes the
unlabeled target-prior estimator and correction prior even though it still does
not use target labels.
