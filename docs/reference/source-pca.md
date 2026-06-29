# Source PCA projection

`neureptrace.decoding.source_pca` implements strict source-only PCA projection for feature matrices.

The protocol is **Category 1 / strict source-only**. PCA mean, scale, and components are fitted from source rows only. Held-out rows are transformed with the fixed source-fitted projection but are not used for fitting.

Supported options include centering, optional feature scaling, optional whitening, and capped component selection.

::: neureptrace.decoding.source_pca
    options:
      members:
        - SourcePCAConfig
        - SourcePCAReference
        - SourcePCATransformResult
        - fit_source_pca_transform
        - fit_source_pca_reference
        - apply_source_pca_transform
        - source_pca_config
