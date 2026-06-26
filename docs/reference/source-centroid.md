# Source centroid decoder

`neureptrace.decoding.source_centroid` implements a dependency-light strict source-only nearest-centroid decoder.

The protocol is **Category 1 / strict source-only**. Class centroids are estimated from source features and source labels only; held-out rows are scored by scaled squared distance to those source centroids.

Optional centroid shrinkage moves source class centroids toward the source global mean. This remains source-only because no held-out target rows or labels are used for fitting.

::: neureptrace.decoding.source_centroid
    options:
      members:
        - SourceCentroidConfig
        - SourceCentroidResult
        - fit_source_centroid_decoder
        - source_centroid_config
