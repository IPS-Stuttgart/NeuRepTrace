# Source confidence weighting

`neureptrace.decoding.source_confidence_weighting` implements strict source-only confidence-based sample weighting.

The protocol is **Category 1 / strict source-only**. Weights are derived from source-row probability estimates and optional source labels. Held-out labels are not part of the API.

Supported modes:

- `confidence`
- `correct_confidence`
- `margin`
- `entropy`

::: neureptrace.decoding.source_confidence_weighting
    options:
      members:
        - SourceConfidenceWeightConfig
        - SourceConfidenceWeightResult
        - compute_source_confidence_weights
        - confidence_scores
        - source_confidence_weight_config
        - normalize_confidence_weight_mode
