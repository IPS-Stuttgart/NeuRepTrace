# Generative augmentation protocols

NeuRepTrace supports lightweight feature-space generative augmentation for fold-local
decoding experiments through `neureptrace.decoding.generative_augmentation`.  The
implementation is intentionally dependency-light: it samples class-conditional
Gaussian feature rows after feature extraction, rather than training a neural GAN
or diffusion model inside every LOSO fold.

The same API exposes three protocol-explicit modes:

| Method | Uses source \(X_s, y_s\)? | Uses target \(X_t\)? | Uses target labels? | Protocol |
| --- | --- | --- | --- | --- |
| `source_gaussian` | yes | no | no | Category 1: strict source-only augmentation |
| `target_style_gaussian` | yes | yes, unlabeled | no | Category 2: unlabeled target-adaptive augmentation |
| `target_calibrated_gaussian` | yes | calibration subset | calibration subset only | Category 3: supervised/calibrated augmentation |

`target_labels` are rejected by the augmentation API.  Category-3 experiments must
pass a disjoint calibration subset through `target_calibration_features` and
`target_calibration_labels`; scored target labels should never be passed into the
augmentation step.

## Direct feature-matrix use

```python
from neureptrace.decoding.generative_augmentation import (
    augment_training_features,
    generative_augmentation_config,
)

config = generative_augmentation_config(
    method="source_gaussian",
    synthetic_per_class=8,
    random_state=13,
)

augmented = augment_training_features(train_features, train_labels, config=config)
model.fit(augmented.features, augmented.labels)
```

For a category-2 target-style run, pass unlabeled target features explicitly:

```python
config = generative_augmentation_config(
    method="target_style_gaussian",
    synthetic_per_class=8,
)

augmented = augment_training_features(
    train_features,
    train_labels,
    config=config,
    target_features=unlabeled_target_features,
)
```

This is a transductive/adaptive protocol if `unlabeled_target_features` are the
held-out evaluation batch.  For online deployment, use a separate unlabeled
calibration block and freeze the synthetic augmentation setup before scoring.

For category-3 few-shot calibration:

```python
config = generative_augmentation_config(
    method="target_calibrated_gaussian",
    synthetic_per_class=8,
    target_calibration_weight=0.5,
)

augmented = augment_training_features(
    train_features,
    train_labels,
    config=config,
    target_calibration_features=target_calibration_features,
    target_calibration_labels=target_calibration_labels,
)
```

## Transfer helper integration

`cross_validate_feature_decoding` accepts source-only generative augmentation:

```python
cross_validate_feature_decoding(
    stimulus_features,
    labels,
    generative_augmentation={
        "method": "source_gaussian",
        "synthetic_per_class": 4,
        "random_state": 13,
    },
)
```

`evaluate_feature_transfer` can also run category-2 or category-3 protocols when
the target/calibration arrays are supplied explicitly:

```python
evaluate_feature_transfer(
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    generative_augmentation={
        "method": "target_style_gaussian",
        "synthetic_per_class": 4,
    },
    generative_target_features=unlabeled_target_features,
)
```

The returned augmentation metadata records the method, synthetic row count, and
protocol category when using `augment_training_features` directly or
`make_generative_augmented_fit_model`.
