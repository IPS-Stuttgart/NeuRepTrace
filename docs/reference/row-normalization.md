# Row-wise normalization

`neureptrace.decoding.row_normalization` implements deterministic per-row feature normalization.

The protocol is **Category 1 / strict source-only** because no cross-row statistics are fitted from held-out rows. Source and held-out rows are transformed independently with the same row-wise rule.

Supported norms:

- `l2`
- `l1`
- `max`

::: neureptrace.decoding.row_normalization
    options:
      members:
        - RowNormalizationConfig
        - RowNormalizationResult
        - normalize_source_and_test_rows
        - normalize_rows
        - row_norms
        - row_normalization_config
        - normalize_norm_mode
