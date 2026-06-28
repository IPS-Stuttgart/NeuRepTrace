# Source variance filter

`neureptrace.decoding.source_variance_filter` implements source-only variance-based feature filtering.

The protocol is **Category 1 / strict source-only**. Feature variances and the selected feature mask are estimated from source rows only. Evaluation rows are transformed with the fitted mask but are not used to fit it.

Typical usage:

```python
from neureptrace.decoding.source_variance_filter import fit_source_variance_filter

result = fit_source_variance_filter(
    source_features=X_source,
    test_features=X_target,
    config={"variance_threshold": 0.0, "top_k": 128},
)

X_source_filtered = result.train_features
X_target_filtered = result.test_features
```

::: neureptrace.decoding.source_variance_filter
    options:
      members:
        - SourceVarianceFilterConfig
        - SourceVarianceFilterResult
        - fit_source_variance_filter
        - source_variance_filter_config
        - source_feature_variances
        - select_variance_features
