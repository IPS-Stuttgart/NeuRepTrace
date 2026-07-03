# Source PCA

`neureptrace.decoding.source_pca` implements a strict source-only PCA projection.

The protocol is **Category 1 / strict source-only**. PCA means, scales, components, singular values, and whitening factors are fitted from source rows only. Held-out rows are transformed with the fitted projection but are not used for fitting.

::: neureptrace.decoding.source_pca
    options:
      members:
        - SourcePCAConfig
        - SourcePCAProjection
        - SourcePCAResult
        - fit_source_pca
        - fit_source_pca_projection
        - transform_with_source_pca
        - source_pca_config
