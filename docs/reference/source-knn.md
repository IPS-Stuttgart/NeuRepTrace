# Source kNN decoder

`neureptrace.decoding.source_knn` implements a dependency-light strict source-only k-nearest-neighbor decoder.

The protocol is **Category 1 / strict source-only**. Neighbor search is fit from source features and source labels only. Held-out rows are scored, but are not used for fitting or adaptation.

Supported weighting modes:

- `uniform`
- `distance`

::: neureptrace.decoding.source_knn
    options:
      members:
        - SourceKNNConfig
        - SourceKNNReference
        - SourceKNNResult
        - fit_source_knn_decoder
        - fit_source_knn_reference
        - predict_source_knn_probabilities
        - source_knn_config
        - normalize_weight_mode
