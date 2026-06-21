# Reconstruction Encoder

`neureptrace.decoding.reconstruction_encoder` provides label-free reconstruction baselines for cross-subject decoding. An encoder/decoder is fit with a reconstruction objective, and the downstream classifier is trained only on source labels in the learned latent space.

Two protocol scopes are exposed:

- `source_only`: fit the encoder on source rows only. This is Protocol 1 / strict source-only.
- `source_plus_target`: fit the encoder on source rows plus unlabeled target rows. This is Protocol 2 / unlabeled target-adaptive. If no separate `target_encoder_features` block is provided, the test feature matrix is used as a transductive unlabeled target batch and this is recorded in metadata.

Two encoder families are available:

- `encoder_kind="linear"` uses the deterministic closed-form linear autoencoder/PCA baseline.
- `encoder_kind="masked_autoencoder"` uses an optional PyTorch nonlinear masked autoencoder. During training, random feature entries are masked and optional Gaussian noise is added; the network reconstructs the original clean feature vector and exposes the encoder bottleneck as the latent representation. Aliases include `"deep"`, `"nonlinear"`, `"deep_masked_autoencoder"`, `"torch_masked_autoencoder"`, and `"mae"`.

Target labels are not accepted by either the latent-space helper or the classifier helper. Passing `target_labels` raises a `ValueError`.

```python
from neureptrace.decoding.reconstruction_encoder import (
    fit_reconstruction_latent_classifier,
    reconstruction_encoder_config,
)

config = reconstruction_encoder_config(
    encoder_kind="masked_autoencoder",
    fit_scope="source_plus_target",
    n_components=32,
    hidden_units=(128, 64),
    mask_fraction=0.25,
    max_epochs=100,
    batch_size=128,
    standardize=True,
    device="auto",
)
result = fit_reconstruction_latent_classifier(
    train_features=X_source,
    train_labels=y_source,
    test_features=X_target_test,
    target_encoder_features=X_target_unlabeled_calibration,
    config=config,
)

print(result.metadata["representation_protocol"])
print(result.metadata["representation_method"])
print(result.predictions)
```

Relevant metadata fields include `representation_protocol`, `representation_method`, `representation_encoder_kind`, `representation_is_nonlinear`, `representation_uses_unlabeled_target_data`, `representation_target_labels_used`, `representation_target_feature_source`, and reconstruction MSE summaries for the fit, source-train, and target-test matrices. Masked-autoencoder runs also report `representation_mask_fraction`, `representation_hidden_units`, `representation_epochs_run`, `representation_validation_reconstruction_mse`, and `representation_masked_training_reconstruction_mse`.

::: neureptrace.decoding.reconstruction_encoder
