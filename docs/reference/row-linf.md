# Row L-infinity normalization

`neureptrace.decoding.row_linf` applies deterministic per-row max-absolute-value normalization to train and test feature matrices.

The protocol is strict-source-compatible because it has no fitted parameters and does not inspect labels.

::: neureptrace.decoding.row_linf
    options:
      members:
        - RowLinfConfig
        - RowLinfResult
        - normalize_train_test_rows_linf
        - normalize_rows_linf
        - row_linf_config
