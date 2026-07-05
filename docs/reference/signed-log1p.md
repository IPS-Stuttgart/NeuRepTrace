# Signed log1p transform

`neureptrace.decoding.signed_log1p` applies a deterministic signed `log1p` compression to train and held-out feature matrices.

The protocol is strict-source-compatible because the transform has no fitted parameters and does not inspect labels.

The transform is:

```text
sign(x) * log1p(abs(x) / scale)
```

::: neureptrace.decoding.signed_log1p
    options:
      members:
        - SignedLog1pConfig
        - SignedLog1pResult
        - transform_train_test_signed_log1p
        - signed_log1p_transform
        - signed_log1p_config
