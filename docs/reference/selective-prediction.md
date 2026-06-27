# Selective prediction

`neureptrace.decoding.selective_prediction` converts probability traces into predictions plus an abstention mask.

Fixed confidence, entropy, and margin thresholds are ordinary post-hoc prediction rules. A requested target coverage sets the confidence threshold from the unlabeled probability batch and should be reported as an unlabeled target-adaptive thresholding step.

The public API intentionally has no label-vector argument.

Typical usage:

```python
from neureptrace.decoding.selective_prediction import selective_predict

result = selective_predict(
    probabilities,
    classes=classes,
    confidence_threshold=0.75,
)

predictions = result.predictions
selected = result.selected_mask
```

::: neureptrace.decoding.selective_prediction
    options:
      members:
        - SelectivePredictionResult
        - selective_predict
        - normalize_probability_rows
        - probability_entropy
        - probability_margin
        - confidence_threshold_for_coverage
