# Signed log transform

`neureptrace.decoding.signed_log` applies deterministic signed `log1p` compression to train and held-out feature matrices.

The protocol is strict-source-compatible because the transform has no fitted parameters and does not inspect labels.

::: neureptrace.decoding.signed_log
    options:
      members:
        - SignedLogConfig
        - SignedLogResult
        - transform_train_test_signed_log
        - transform_signed_log
        - signed_log_config
