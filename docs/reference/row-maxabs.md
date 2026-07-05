# Row max-abs normalization

`neureptrace.decoding.row_maxabs` applies deterministic per-row max-absolute normalization to train and score feature matrices.

The transform has no fitted parameters and does not inspect labels. It is therefore compatible with **Category 1 / strict source-only** benchmarking.

::: neureptrace.decoding.row_maxabs
    options:
      members:
        - RowMaxAbsConfig
        - RowMaxAbsResult
        - normalize_train_score_rows_maxabs
        - normalize_rows_maxabs
        - row_maxabs_config
