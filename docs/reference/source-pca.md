# Source PCA

`neureptrace.decoding.source_pca` fits a PCA projection from source rows only and applies the frozen projection to source and held-out rows.

The protocol is **Category 1 / strict source-only** because held-out rows are transformed but never used to estimate centering, scaling, components, or whitening factors.

::: neureptrace.decoding.source_pca
    options:
      members:
        - SourcePCAConfig
        - SourcePCAProjection
        - SourcePCAResult
        - fit_source_pca
        - fit_source_pca_projection
        - apply_source_pca
        - source_pca_config
