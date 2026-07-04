# Source threshold transform

`neureptrace.decoding.source_threshold` fits feature-wise thresholds from source rows only and applies the fixed threshold map to source and held-out rows.

The protocol is **Category 1 / strict source-only** because held-out rows are transformed but never used to fit thresholds.

Supported threshold modes:

- `median`
- `mean`
- `quantile`
- `zero`

Supported outputs:

- `binary`
- `signed`

::: neureptrace.decoding.source_threshold
    options:
      members:
        - SourceThresholdConfig
        - SourceThresholdMap
        - SourceThresholdResult
        - fit_source_threshold_transform
        - fit_source_threshold_map
        - apply_source_threshold_transform
        - source_threshold_config
        - normalize_threshold_mode
        - normalize_output_mode
