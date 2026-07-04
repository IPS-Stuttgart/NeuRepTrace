# Source asinh transform

`neureptrace.decoding.source_asinh` fits optional feature scales from source rows only and applies a signed inverse-hyperbolic-sine compression to source and held-out rows.

The protocol is **Category 1 / strict source-only** because held-out rows are transformed but never used to fit scales.

Supported scale modes:

- `unit`
- `std`
- `mad`
- `iqr`

::: neureptrace.decoding.source_asinh
    options:
      members:
        - SourceAsinhConfig
        - SourceAsinhMap
        - SourceAsinhResult
        - fit_source_asinh_transform
        - fit_source_asinh_map
        - apply_source_asinh_transform
        - source_asinh_config
        - normalize_scale_mode
