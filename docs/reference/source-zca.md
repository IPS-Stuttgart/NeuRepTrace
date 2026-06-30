# Source ZCA whitening

`neureptrace.decoding.source_zca` implements strict source-only ZCA whitening preprocessing.

The protocol is **Category 1 / strict source-only**. The mean, covariance, and whitening matrix are estimated from source rows only. Held-out rows are transformed but are not used for fitting.

::: neureptrace.decoding.source_zca
    options:
      members:
        - SourceZCAConfig
        - SourceZCAReference
        - SourceZCAResult
        - fit_source_zca_transform
        - fit_source_zca_reference
        - apply_source_zca_transform
        - source_zca_config
