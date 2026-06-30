# Source outlier weighting

`neureptrace.decoding.source_outlier` implements strict source-only class-distance weighting.

The protocol is **Category 1 / strict source-only**. Class centroids, feature scales, per-class thresholds, and row weights are estimated from source features and source labels only.

Supported threshold modes are `quantile` and `mad`. Supported weight modes are `binary`, `linear`, and `soft`.

::: neureptrace.decoding.source_outlier
    options:
      members:
        - SourceOutlierConfig
        - SourceOutlierResult
        - compute_source_outlier_weights
        - source_outlier_config
        - normalize_threshold_mode
        - normalize_weight_mode
