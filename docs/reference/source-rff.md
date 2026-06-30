# Source random Fourier features

`neureptrace.decoding.source_rff` implements strict source-only random Fourier feature preprocessing for RBF-style kernels.

The protocol is **Category 1 / strict source-only**. Gamma and optional standardization statistics are fitted from source rows only. Held-out rows are transformed with the frozen feature map and are not used for fitting.

::: neureptrace.decoding.source_rff
    options:
      members:
        - SourceRFFConfig
        - SourceRFFReference
        - SourceRFFResult
        - fit_source_rff_transform
        - fit_source_rff_reference
        - apply_source_rff
        - source_rff_config
        - normalize_gamma
