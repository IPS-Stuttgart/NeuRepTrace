# Source PCA projection

`neureptrace.decoding.source_pca` fits a PCA projection from source rows only and applies that source-fitted projection to source and held-out rows.

The protocol is **Category 1 / strict source-only** because held-out rows are transformed but never used to estimate the projection.

::: neureptrace.decoding.source_pca
    options:
      members:
        - SourcePCAConfig
        - SourcePCAProjector
        - SourcePCAResult
        - fit_source_pca_projection
        - fit_source_pca_projector
        - transform_with_source_pca
        - reconstruct_from_source_pca
        - source_pca_config
