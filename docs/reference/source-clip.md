# Source clip

`neureptrace.decoding.source_clip` fits feature-wise clipping bounds from source rows only and applies them to source and held-out rows.

The protocol is **Category 1 / strict source-only** because only source rows are used to estimate the clipping bounds.

::: neureptrace.decoding.source_clip
    options:
      members:
        - SourceClipConfig
        - SourceClipBounds
        - SourceClipResult
        - fit_source_clip
        - fit_source_clip_bounds
        - apply_source_clip
        - source_clip_config
        - normalize_center_mode
