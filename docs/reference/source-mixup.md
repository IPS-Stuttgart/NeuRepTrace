# Source MixUp

`neureptrace.decoding.source_mixup` implements dependency-light feature-space MixUp for source-only domain generalization.

The protocol is **Category 1 / strict source-only**. It uses source features, source labels, and optional source-domain identifiers. It does **not** accept target features or target labels.

Synthetic rows are convex combinations of source rows. The result contains both hard labels for existing scikit-learn style pipelines and soft label distributions for consumers that support probabilistic targets.

Typical usage:

```python
from neureptrace.decoding.source_mixup import augment_source_with_mixup

result = augment_source_with_mixup(
    X_source,
    y_source,
    source_domains=subject_ids,
    config={
        "synthetic_per_class": 8,
        "same_class_partner": True,
        "cross_domain_partner": True,
        "random_state": 13,
    },
)

X_aug = result.features
y_aug = result.labels
```

::: neureptrace.decoding.source_mixup
    options:
      members:
        - SourceMixUpConfig
        - SourceMixUpResult
        - augment_source_with_mixup
        - mixup_rows
        - source_mixup_config
        - normalize_hard_label_policy
