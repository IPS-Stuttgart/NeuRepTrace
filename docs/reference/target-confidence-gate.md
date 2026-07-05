# Target confidence gate

`neureptrace.decoding.target_confidence_gate` selects confident target probability rows without using target labels.

The protocol is **Category 2 / unlabeled target-adaptive** because it uses target probability rows from a source-trained model, and optional target-batch retain-fraction thresholds. Held-out target labels are not part of the API.

Supported confidence scores:

- `max_probability`
- `margin`
- `normalized_confidence`

Supported threshold modes:

- `fixed`
- `retain_fraction`

::: neureptrace.decoding.target_confidence_gate
    options:
      members:
        - TargetConfidenceGateConfig
        - TargetConfidenceGateResult
        - gate_target_probabilities_by_confidence
        - target_confidence_gate_config
        - target_confidence_scores
        - normalize_score_mode
        - normalize_threshold_mode
