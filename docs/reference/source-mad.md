# Source MAD normalization

`neureptrace.decoding.source_mad` implements strict source-only median/MAD normalization.

The protocol is **Category 1 / strict source-only**. The robust center and scale are estimated from source rows only. Held-out rows are transformed with the frozen source-fitted statistics and are not used for fitting.

::: neureptrace.decoding.source_mad
    options:
      members:
        - SourceMADConfig
        - SourceMADReference
        - SourceMADResult
        - fit_source_mad_transform
        - fit_source_mad_reference
        - apply_source_mad_transform
        - source_mad_config
