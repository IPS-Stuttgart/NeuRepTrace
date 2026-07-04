# Signed power transform

`neureptrace.decoding.signed_power` applies a deterministic signed power compression to train and held-out feature matrices.

The protocol is strict-source-compatible because the transform has no fitted parameters and does not inspect labels.

::: neureptrace.decoding.signed_power
    options:
      members:
        - SignedPowerConfig
        - SignedPowerResult
        - transform_train_test_signed_power
        - signed_power_transform
        - signed_power_config
