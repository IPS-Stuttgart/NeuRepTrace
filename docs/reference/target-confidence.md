# Target confidence weighting

`neureptrace.decoding.target_confidence` converts target probability rows from a source-trained model into pseudo-labels, confidence diagnostics, keep masks, and sample weights.

The protocol is **Category 2 / unlabeled target-adaptive**. It uses target probability rows, but it does not accept held-out target labels.

Supported weighting modes:

- `confidence`
- `margin`
- `entropy`
- `mask`

::: neureptrace.decoding.target_confidence
    options:
      members:
        - TargetConfidenceConfig
        - TargetConfidenceResult
        - target_confidence_weights
        - target_confidence_config
        - normalize_weighting_mode
        - probability_margin
        - normalized_entropy
