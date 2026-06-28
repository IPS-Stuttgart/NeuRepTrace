# Source feature clipping

`neureptrace.decoding.source_clipping` implements source-only feature clipping for robust cross-subject decoding.

The protocol is **Category 1 / strict source-only**. Lower and upper bounds are estimated from source rows only, then applied to source and held-out feature matrices. Held-out rows are never used to fit the clipping bounds.

Typical usage:

```python
from neureptrace.decoding.source_clipping import fit_source_feature_clipping

result = fit_source_feature_clipping(
    source_features=X_source,
    test_features=X_target,
    config={"lower_quantile": 0.01, "upper_quantile": 0.99},
)

X_source_clip = result.train_features
X_target_clip = result.test_features
```

::: neureptrace.decoding.source_clipping
    options:
      members:
        - SourceFeatureClippingConfig
        - SourceFeatureClippingResult
        - fit_source_feature_clipping
        - source_feature_clipping_config
        - source_feature_clipping_bounds
        - apply_feature_clipping
