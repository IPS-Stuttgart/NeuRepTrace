# Reconstruction Encoder

`neureptrace.decoding.reconstruction_encoder` provides a label-free reconstruction baseline for cross-subject decoding. The encoder/decoder is a deterministic linear autoencoder/PCA fit by squared reconstruction loss, and the downstream classifier is trained only on source labels in the latent space.

Two protocol scopes are exposed:

- `source_only`: fit the encoder on source rows only. This is Protocol 1 / strict source-only.
- `source_plus_target`: fit the encoder on source rows plus unlabeled target rows. This is Protocol 2 / unlabeled target-adaptive. If no separate `target_encoder_features` block is provided, the test feature matrix is used as a transductive unlabeled target batch and this is recorded in metadata.

Target labels are not accepted by either the latent-space helper or the classifier helper. Passing `target_labels` raises a `ValueError`.

```python
from neureptrace.decoding.reconstruction_encoder import (
    fit_reconstruction_latent_classifier,
    reconstruction_encoder_config,
)

config = reconstruction_encoder_config(
    fit_scope="source_plus_target",
    n_components=32,
    standardize=True,
)
result = fit_reconstruction_latent_classifier(
    train_features=X_source,
    train_labels=y_source,
    test_features=X_target_test,
    target_encoder_features=X_target_unlabeled_calibration,
    config=config,
)

print(result.metadata["representation_protocol"])
print(result.predictions)
```

Relevant metadata fields include `representation_protocol`, `representation_uses_unlabeled_target_data`, `representation_target_labels_used`, `representation_target_feature_source`, and reconstruction MSE summaries for the fit, source-train, and target-test matrices.

::: neureptrace.decoding.reconstruction_encoder
