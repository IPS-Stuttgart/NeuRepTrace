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
2. Select high-confidence pseudo-labeled target rows.
3. Estimate target-batch class prototypes from selected rows.
4. Blend frozen source-model posteriors with prototype posteriors.
5. Repeat until the pseudo-label assignment stabilizes or `max_iterations` is reached.

The adapter intentionally has no `target_labels` argument. Any target labels should be used only after adaptation, for held-out scoring.
