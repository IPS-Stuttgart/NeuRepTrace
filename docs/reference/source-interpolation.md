# Source interpolation

`neureptrace.decoding.source_interpolation` implements strict source-only interpolation augmentation.

The protocol is **Category 1 / strict source-only**. Synthetic rows are convex combinations of same-class source rows. Optional source-domain identifiers can be used to prefer cross-domain partners, but held-out rows and labels are not part of the API.

Supported pair modes:

- `same_class`
- `same_class_cross_domain`

::: neureptrace.decoding.source_interpolation
    options:
      members:
        - SourceInterpolationConfig
        - SourceInterpolationResult
        - augment_source_with_interpolation
        - interpolate_rows
        - source_interpolation_config
        - normalize_pair_mode
