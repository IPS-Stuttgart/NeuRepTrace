# Source robust scaler

`neureptrace.decoding.source_robust_scaler` implements source-only robust feature scaling.

The protocol is **Category 1 / strict source-only**. Location and scale statistics are estimated from source rows only. Evaluation rows are transformed with the fitted statistics but are not used to fit them.

Typical usage:

```python
from neureptrace.decoding.source_robust_scaler import fit_source_robust_scaler

result = fit_source_robust_scaler(
    source_features=X_source,
    test_features=X_target,
    config={"center": "median", "scale": "iqr"},
)

X_source_scaled = result.train_features
X_target_scaled = result.test_features
```

Supported center modes: `median`, `mean`, `none`.

Supported scale modes: `iqr`, `mad`, `std`, `none`.

::: neureptrace.decoding.source_robust_scaler
    options:
      members:
        - SourceRobustScalerConfig
        - SourceRobustScalerStats
        - SourceRobustScalerResult
        - fit_source_robust_scaler
        - source_robust_scaler_config
        - fit_source_robust_scaler_stats
        - apply_source_robust_scaler
        - normalize_center_mode
        - normalize_scale_mode
