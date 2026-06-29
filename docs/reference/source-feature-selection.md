# Source feature selection

`neureptrace.decoding.source_feature_selection` implements strict source-only univariate feature selection.

The protocol is **Category 1 / strict source-only**. Feature scores are fitted from source rows and source labels only. Held-out rows are transformed by the fixed selected-column mask but are not used for fitting.

Supported score methods:

- `anova`
- `variance`

::: neureptrace.decoding.source_feature_selection
    options:
      members:
        - SourceFeatureSelectionConfig
        - SourceFeatureSelectionResult
        - fit_source_feature_selection
        - source_feature_scores
        - select_top_source_features
        - source_feature_selection_config
        - normalize_score_method
