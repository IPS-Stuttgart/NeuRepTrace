# Source feature scaling

`neureptrace.decoding.source_scaling` implements source-only gain augmentation for feature matrices.

Synthetic rows are scaled copies of source rows. The method uses source rows and labels only.

::: neureptrace.decoding.source_scaling
    options:
      members:
        - SourceFeatureScalingConfig
        - SourceFeatureScalingResult
        - augment_source_with_feature_scaling
        - sample_scaling_factors
        - source_feature_scaling_config
