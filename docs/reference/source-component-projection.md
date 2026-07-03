# Source component projection

`neureptrace.decoding.source_component_projection` fits an SVD component projection from source rows only and applies that source-fitted projection to source and held-out rows.

The protocol is **Category 1 / strict source-only** because held-out rows are transformed but never used to estimate the projection.

::: neureptrace.decoding.source_component_projection
    options:
      members:
        - SourceComponentConfig
        - SourceComponentProjector
        - SourceComponentResult
        - fit_source_component_projection
        - fit_source_component_projector
        - transform_with_source_components
        - reconstruct_from_source_components
        - source_component_config
