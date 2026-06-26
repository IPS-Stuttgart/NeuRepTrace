# Source-free subject adaptation

NeuRepTrace now includes a source-free target-subject adaptation utility in `neureptrace.decoding.source_free`.

This mode is intended for Protocol 2 evaluations: a source-trained model is already fitted, and the adaptation step receives only unlabeled target-subject features `X_t`. It does not accept source rows, source labels, or target labels during adaptation.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neureptrace.decoding.source_free import fit_source_free_predict_proba

source_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
source_model.fit(X_source, y_source)

result = fit_source_free_predict_proba(
    source_model=source_model,
    target_features=X_target_unlabeled,
    confidence_threshold=0.75,
    max_iterations=5,
)

probabilities = result.probabilities
metadata = result.metadata
```

## Protocol hygiene

The adapter records explicit metadata:

- `source_free_uses_pretrained_source_model=True`
- `source_free_uses_source_features_during_adaptation=False`
- `source_free_uses_source_labels_during_adaptation=False`
- `source_free_uses_target_features=True`
- `source_free_uses_target_labels=False`
- `source_free_valid_for_benchmark=True`

The fitted source model necessarily encodes information learned from source subjects, but the adaptation step itself is source-free: only the fixed source model and unlabeled target features are used.

## Method sketch

1. Run the fitted source model on the unlabeled target batch.
2. Select pseudo-labeled target rows.
3. Estimate target-batch class prototypes from selected rows.
4. Blend frozen source-model posteriors with prototype posteriors.
5. Repeat until the pseudo-label assignment stabilizes or `max_iterations` is reached.

The adapter intentionally has no `target_labels` argument. Any target labels should be used only after adaptation, for held-out scoring.

## Class-balanced pseudo-label selection

OpenNeuro/Protocol-2-style target batches can be class imbalanced or source-biased. The default selector keeps rows whose source-model posterior confidence exceeds `confidence_threshold`. That is conservative, but it can collapse to a single pseudo-class and disable prototype adaptation.

For such runs, use the target-label-free balanced selector:

```python
result = fit_source_free_predict_proba(
    source_model=source_model,
    target_features=X_target_unlabeled,
    confidence_threshold=0.80,
    max_iterations=5,
    min_class_count=2,
    pseudo_label_selection="balanced_topk",
    balanced_topk_per_class=4,
)
```

`balanced_topk` still uses only source-model probabilities on the unlabeled target features. For each predicted pseudo-class, it keeps up to `balanced_topk_per_class` rows with the highest posterior confidence. If fewer than `min_class_count` rows for a predicted pseudo-class exceed the threshold, the selector falls back to that class's top-confidence rows so minority prototypes can remain active. This is useful as a Protocol-2/2.5 candidate for OpenNeuro datasets where the source model is confident mostly on the dominant target pseudo-class.

The metadata records the selector as:

- `source_free_pseudo_label_selection`
- `source_free_balanced_topk_per_class`
- `source_free_class_counts`

Keep the default `confidence` selector as the benchmark-safe baseline, and report `balanced_topk` as a separate adaptation variant.

## Soft target prototypes

For small or difficult OpenNeuro target folds, every target row may be predicted as the same pseudo-class even though the non-argmax posterior still carries usable information. In that case, hard pseudo-label prototypes can stop with `none_selected` or `insufficient_active_classes`.

Use the soft prototype estimator as a Protocol 2.5 follow-up:

```python
result = fit_source_free_predict_proba(
    source_model=source_model,
    target_features=X_target_unlabeled,
    confidence_threshold=0.90,
    max_iterations=5,
    min_class_count=2,
    min_active_classes=2,
    prototype_weight=0.5,
    prototype_estimator="soft_all",
)
```

Available prototype estimators:

- `hard`: default; selected rows contribute only to their hard pseudo-class.
- `soft_selected`: selected rows contribute to every class with posterior weights.
- `soft_all`: every target row contributes to every class with posterior weights.

`soft_all` is still source-free: it uses only the fitted source model and unlabeled target features/probabilities. It is useful when a target fold has collapsed argmax pseudo-labels but nonzero posterior mass for minority classes. Metadata records the chosen estimator as `source_free_prototype_estimator`.
