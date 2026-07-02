# Source feature roll

`neureptrace.decoding.source_roll` implements strict source-only feature roll augmentation for ordered feature matrices.

The protocol is **Category 1 / strict source-only**. Synthetic rows are shifted copies of labeled source rows; held-out data are not used.

::: neureptrace.decoding.source_roll
    options:
      members:
        - SourceFeatureRollConfig
        - SourceFeatureRollResult
        - augment_source_with_feature_roll
        - roll_feature_row
        - sample_roll_shift
        - source_feature_roll_config
        - normalize_roll_mode
