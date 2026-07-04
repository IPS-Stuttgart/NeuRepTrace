# Source ECDF uniform transform

`neureptrace.decoding.source_ecdf_uniform` fits empirical CDF breakpoints from source rows only and applies the fixed map to source and held-out rows.

The protocol is **Category 1 / strict source-only** because held-out rows are transformed but never used to fit the map.

::: neureptrace.decoding.source_ecdf_uniform
    options:
      members:
        - SourceEcdfConfig
        - SourceEcdfMap
        - SourceEcdfResult
        - fit_source_ecdf_transform
        - fit_source_ecdf_map
        - apply_source_ecdf_transform
        - source_ecdf_config
