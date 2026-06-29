# Source feature selection

`neureptrace.decoding.source_feature_select` implements strict source-only variance feature selection.

The protocol is Category 1 / strict source-only. Feature scores and selected columns are fitted from source rows only; held-out rows are transformed with the fixed selected indices.

::: neureptrace.decoding.source_feature_select
    options:
      members:
        - SourceFeatureSelectResult
        - select_source_variance_features
        - source_variance_feature_indices
