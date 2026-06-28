# Source correlation filter

`neureptrace.decoding.source_correlation_filter` implements source-only correlation-based feature filtering.

The protocol is **Category 1 / strict source-only**. Feature correlations, feature importances, and the selected feature mask are estimated from source rows only. Evaluation rows are transformed with the fitted mask but are not used to fit it.

Typical usage:

```python
from neureptrace.decoding.source_correlation_filter import fit_source_correlation_filter

result = fit_source_correlation_filter(
    source_features=X_source,
    test_features=X_target,
    config={"max_abs_correlation": 0.98, "max_features": 256},
)

X_source_filtered = result.train_features
X_target_filtered = result.test_features
```

::: neureptrace.decoding.source_correlation_filter
    options:
      members:
        - SourceCorrelationFilterConfig
        - SourceCorrelationFilterResult
        - fit_source_correlation_filter
        - source_correlation_filter_config
        - source_feature_correlation
        - source_feature_importance
        - select_uncorrelated_features
