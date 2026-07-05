# Source squash transform

`neureptrace.decoding.source_squash` fits optional feature scales from source rows only and applies a bounded odd feature compression to source and held-out rows.

The protocol is **Category 1 / strict source-only** because held-out rows are transformed but never used to fit scales.

Supported scale modes:

- `unit`
- `std`
- `mad`
- `iqr`

::: neureptrace.decoding.source_squash
    options:
      members:
        - SourceSquashConfig
        - SourceSquashMap
        - SourceSquashResult
        - fit_source_squash_transform
        - fit_source_squash_map
        - apply_source_squash_transform
        - source_squash_config
        - normalize_scale_mode
