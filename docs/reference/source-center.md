# Source center transform

`neureptrace.decoding.source_center` fits a feature-wise center from source rows only and applies the fixed center to source and held-out rows.

The protocol is **Category 1 / strict source-only** because held-out rows are transformed but never used to fit the center.

Supported center modes:

- `mean`
- `median`
- `zero`

::: neureptrace.decoding.source_center
    options:
      members:
        - SourceCenterConfig
        - SourceCenterMap
        - SourceCenterResult
        - fit_source_center_transform
        - fit_source_center_map
        - apply_source_center_transform
        - source_center_config
        - normalize_center_mode
