# Domain importance weighting

`neureptrace.decoding.domain_importance` estimates source sample weights from a binary source-versus-target domain classifier.

This is a Category 2 / unlabeled target-adaptive protocol. It uses source features and unlabeled target features to estimate source-row weights. It does not use target labels.

Typical usage:

```python
from neureptrace.decoding.domain_importance import fit_domain_classifier_importance_weights

result = fit_domain_classifier_importance_weights(
    source_features=X_source,
    target_features=X_target,
)

sample_weight = result.sample_weights
```

The returned weights can be passed to any downstream classifier that supports `sample_weight`.

::: neureptrace.decoding.domain_importance
    options:
      members:
        - DomainImportanceConfig
        - DomainImportanceResult
        - fit_domain_classifier_importance_weights
        - domain_importance_config
        - apply_domain_importance_weights
