# Source balancing

`neureptrace.decoding.source_balance` implements strict source-only sample weighting and balanced resampling helpers.

The protocol is **Category 1 / strict source-only**. The helpers use source labels and optional source-domain identifiers only. Held-out target features and labels are not accepted.

Supported strategies:

- `none`
- `class`
- `domain`
- `class_domain`

Supported group targets:

- `max`
- `min`
- `mean`

::: neureptrace.decoding.source_balance
    options:
      members:
        - SourceBalanceConfig
        - SourceBalanceResult
        - SourceResampleResult
        - compute_source_balance_weights
        - resample_source_rows_balanced
        - source_balance_config
        - normalize_balance_strategy
        - normalize_balance_target
