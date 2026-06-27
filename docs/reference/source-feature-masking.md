# Source feature masking

`neureptrace.decoding.source_masking` implements strict source-only feature masking augmentation for M/EEG feature matrices.

The protocol is **Category 1 / strict source-only**. It uses source features, source labels, and optional source-domain identifiers for provenance. It does not accept held-out target data.

Supported masking modes:

- `feature`: randomly selected feature columns
- `block`: one contiguous feature block

Supported fill modes:

- `feature_mean`: source-only column means
- `row_mean`: the sampled row mean
- `zero`: zeros

Typical usage:

```python
from neureptrace.decoding.source_masking import augment_source_with_feature_masking

result = augment_source_with_feature_masking(
    X_source,
    y_source,
    source_domains=subject_ids,
    config={
        "synthetic_per_class": 8,
        "mask_fraction": 0.15,
        "mask_mode": "feature",
        "fill_mode": "feature_mean",
    },
)

X_aug = result.features
y_aug = result.labels
```

::: neureptrace.decoding.source_masking
    options:
      members:
        - SourceFeatureMaskingConfig
        - SourceFeatureMaskingResult
        - augment_source_with_feature_masking
        - feature_mask_indices
        - source_feature_masking_config
        - normalize_mask_mode
        - normalize_fill_mode
