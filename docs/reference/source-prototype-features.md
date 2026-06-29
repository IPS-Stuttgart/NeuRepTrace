# Source prototype features

`neureptrace.decoding.source_prototype_features` implements strict source-only prototype-distance feature transforms.

The protocol is **Category 1 / strict source-only**. Class prototypes and feature scales are fitted from source rows and source labels only. Held-out rows are transformed by the fixed source prototypes and are not used for fitting.

Supported metrics:

- `squared_euclidean`
- `euclidean`
- `cosine`

Supported outputs:

- `distance`
- `negative_distance`
- `rbf_similarity`

::: neureptrace.decoding.source_prototype_features
    options:
      members:
        - SourcePrototypeFeatureConfig
        - SourcePrototypeFeatureResult
        - fit_source_prototype_features
        - class_prototypes
        - prototype_distance_features
        - source_prototype_feature_config
        - normalize_prototype_metric
        - normalize_prototype_output
