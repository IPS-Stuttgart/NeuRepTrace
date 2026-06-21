# Generative augmentation protocols

NeuRepTrace supports feature-space generative augmentation for fold-local decoding experiments through `neureptrace.decoding.generative_augmentation`.

The original Gaussian modes remain dependency-light baselines. The new `*_gan` and `*_diffusion` modes train fold-local neural generators on extracted feature matrices and require the optional `torch` extra. These models synthesize feature rows, not raw MEG time series, and must be fitted inside each LOSO fold to avoid leakage.

## Protocol categories

| Method | Target data use | Target label use | Protocol |
| --- | --- | --- | --- |
| `source_gaussian` | no | no | Category 1: strict source-only augmentation |
| `source_gan` | no | no | Category 1: strict source-only augmentation |
| `source_diffusion` | no | no | Category 1: strict source-only augmentation |
| `target_style_gaussian` | unlabeled target features | no | Category 2: unlabeled target-adaptive augmentation |
| `target_style_gan` | unlabeled target features | no | Category 2: unlabeled target-adaptive augmentation |
| `target_style_diffusion` | unlabeled target features | no | Category 2: unlabeled target-adaptive augmentation |
| `target_calibrated_gaussian` | disjoint calibration subset | calibration subset only | Category 3: supervised/calibrated augmentation |
| `target_calibrated_gan` | disjoint calibration subset | calibration subset only | Category 3: supervised/calibrated augmentation |
| `target_calibrated_diffusion` | disjoint calibration subset | calibration subset only | Category 3: supervised/calibrated augmentation |

`target_labels` are rejected by the augmentation API. Category-3 experiments must pass a disjoint calibration subset through `target_calibration_features` and `target_calibration_labels`; scored target labels should never be passed into the augmentation step.

## Neural generator controls

The following options apply to `*_gan` and `*_diffusion` methods.

| Option | Meaning |
| --- | --- |
| `neural_epochs` | Fold-local training epochs for the generator. |
| `neural_hidden_dim` | Width of the generator, discriminator, or denoiser MLP. |
| `neural_batch_size` | Mini-batch size, capped at the current fold's row count. |
| `neural_learning_rate` | Adam learning rate. |
| `gan_latent_dim` | Latent noise dimension for conditional GAN methods. |
| `gan_discriminator_steps` | Discriminator updates per GAN mini-batch. |
| `diffusion_steps` | Number of DDPM-style denoising steps for diffusion methods. |

All neural modes standardize the current fold's generator-training features before training and unstandardize synthetic rows afterward. Category-2 target-style methods then apply unlabeled mean/covariance matching to the generated rows. Category-3 calibrated neural modes train on source rows plus disjoint target calibration rows, then shift generated class means toward calibration class means according to `target_calibration_weight`.

## Minimal examples

Source-only GAN augmentation:

```python
config = generative_augmentation_config(
    method="source_gan",
    synthetic_per_class=8,
    neural_epochs=200,
    random_state=13,
)
augmented = augment_training_features(train_features, train_labels, config=config)
```

Target-style diffusion augmentation:

```python
config = generative_augmentation_config(method="target_style_diffusion", synthetic_per_class=8)
augmented = augment_training_features(
    train_features,
    train_labels,
    config=config,
    target_features=unlabeled_target_features,
)
```

Few-shot calibrated GAN augmentation:

```python
config = generative_augmentation_config(method="target_calibrated_gan", synthetic_per_class=8)
augmented = augment_training_features(
    train_features,
    train_labels,
    config=config,
    target_calibration_features=target_calibration_features,
    target_calibration_labels=target_calibration_labels,
)
```

`evaluate_feature_transfer` can run Category 2 or Category 3 protocols when the target or calibration arrays are supplied explicitly through the existing `generative_target_features`, `generative_target_calibration_features`, and `generative_target_calibration_labels` arguments.

The returned augmentation metadata records the method, synthetic row count, protocol category, and whether a neural generator was used.
