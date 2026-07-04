# Target confidence selection

`neureptrace.decoding.target_confidence_selection` selects confident rows from an unlabeled target probability batch.

The protocol is **Category 2 / unlabeled target-adaptive**. It uses target probability rows from a source-trained model, but it does not accept held-out target labels.

Typical usage:

```python
from neureptrace.decoding.target_confidence_selection import select_target_confident_predictions

result = select_target_confident_predictions(
    probabilities=target_probabilities,
    classes=classes,
    config={"min_confidence": 0.8, "min_margin": 0.3},
)

selected_rows = result.selected_indices
pseudo_labels = result.predictions[result.keep_mask]
```

::: neureptrace.decoding.target_confidence_selection
    options:
      members:
        - TargetConfidenceSelectionConfig
        - TargetConfidenceSelectionResult
        - select_target_confident_predictions
        - target_confidence_selection_config
