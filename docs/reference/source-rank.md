# Source rank transform

`neureptrace.decoding.source_rank` implements a strict source-only empirical rank transform.

The reference distribution is fitted from source rows only. Evaluation rows are transformed with that fixed source reference and are not used for fitting.

::: neureptrace.decoding.source_rank
    options:
      members:
        - SourceRankReference
        - SourceRankTransformResult
        - fit_source_rank_reference
        - transform_source_rank_features
        - fit_source_rank_transform
        - normalize_rank_output
