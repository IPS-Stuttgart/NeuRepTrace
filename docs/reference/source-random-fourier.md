# Source random Fourier features

`neureptrace.decoding.source_random_fourier` implements strict source-only random Fourier feature preprocessing for RBF-style nonlinear probes.

The protocol is **Category 1 / strict source-only**. The random basis is determined from the source feature width, source-only gamma setting, and random seed. Held-out rows are transformed but are not used for fitting or tuning.

Supported gamma modes:

- `auto`: source-only median-distance heuristic
- fixed positive float

::: neureptrace.decoding.source_random_fourier
    options:
      members:
        - SourceRandomFourierConfig
        - SourceRandomFourierReference
        - SourceRandomFourierResult
        - fit_source_random_fourier_transform
        - fit_source_random_fourier_reference
        - apply_source_random_fourier
        - source_auto_rbf_gamma
        - source_random_fourier_config
