# Source prior adjustment

`neureptrace.decoding.source_prior` implements strict source-only class-prior adjustment for probability rows.

The protocol is **Category 1 / strict source-only**. Class priors are estimated from source labels only. Held-out probability rows may be transformed, but held-out labels and held-out features are not used for fitting.

Supported target priors:

- `uniform`
- `source`

::: neureptrace.decoding.source_prior
    options:
      members:
        - SourcePriorConfig
        - SourcePriorAdjustmentResult
        - estimate_source_class_prior
        - adjust_probabilities_to_source_prior
        - source_prior_config
        - normalize_target_prior
