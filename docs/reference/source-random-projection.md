# Source random projection

`neureptrace.decoding.source_random_projection` implements strict source-only random projection preprocessing.

The protocol is **Category 1 / strict source-only**. The projection matrix is determined from the source feature width and random seed only. Held-out rows are transformed but are not used for fitting or tuning.

Supported distributions:

- `gaussian`
- `sparse`

::: neureptrace.decoding.source_random_projection
    options:
      members:
        - SourceRandomProjectionConfig
        - SourceRandomProjectionReference
        - SourceRandomProjectionResult
        - fit_source_random_projection_transform
        - fit_source_random_projection_reference
        - apply_source_random_projection
        - source_random_projection_config
        - normalize_projection_distribution
