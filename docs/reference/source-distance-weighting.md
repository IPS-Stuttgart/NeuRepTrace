# Source distance weighting

`neureptrace.decoding.source_distance_weighting` implements strict source-only distance-based sample weighting.

The protocol is **Category 1 / strict source-only**. Weights are fitted from source features, source labels, and optional source-domain identifiers. Held-out rows and held-out labels are not part of the API.

Supported grouping modes:

- `global`
- `class`
- `domain`
- `class_domain`

::: neureptrace.decoding.source_distance_weighting
    options:
      members:
        - SourceDistanceWeightConfig
        - SourceDistanceWeightResult
        - compute_source_distance_weights
        - source_distance_weight_config
        - normalize_distance_group_mode
