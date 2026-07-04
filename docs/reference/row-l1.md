# Row L1 normalization

`neureptrace.decoding.row_l1` applies deterministic per-row L1 normalization to train and test feature matrices.

The protocol is strict-source-compatible because it has no fitted parameters and does not inspect labels.

::: neureptrace.decoding.row_l1
    options:
      members:
        - RowL1Config
        - RowL1Result
        - normalize_train_test_rows_l1
        - normalize_rows_l1
        - row_l1_config
