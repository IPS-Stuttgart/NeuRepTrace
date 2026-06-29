# Source polynomial features

`neureptrace.decoding.source_polynomial` implements strict source-only deterministic polynomial feature expansion.

The protocol is **Category 1 / strict source-only**. The feature map is determined from the source feature width and configuration only. Held-out rows are transformed but are not used for fitting or tuning.

Supported blocks:

- bias
- original features
- squared features
- pairwise interactions

::: neureptrace.decoding.source_polynomial
    options:
      members:
        - SourcePolynomialConfig
        - SourcePolynomialReference
        - SourcePolynomialResult
        - fit_source_polynomial_transform
        - fit_source_polynomial_reference
        - apply_source_polynomial_transform
        - source_polynomial_config
