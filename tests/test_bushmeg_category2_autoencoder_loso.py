from __future__ import annotations

import numpy as np

from neureptrace.bushmeg_category2_autoencoder_loso import (
    Category2AutoencoderConfig,
    WindowSpec,
    _classification_metrics,
    _fit_autoencoder_latents,
    _predict_source_classifier,
)


def test_category2_autoencoder_protocol_uses_unlabeled_target_features() -> None:
    rng = np.random.default_rng(13)
    x_source = np.vstack([rng.normal(loc=class_idx, scale=0.2, size=(8, 12)) for class_idx in range(3)])
    y_source = np.repeat(np.arange(3), 8)
    x_target = np.vstack([rng.normal(loc=class_idx, scale=0.2, size=(3, 12)) for class_idx in range(3)])
    y_target_for_metrics_only = np.repeat(np.arange(3), 3)

    cfg = Category2AutoencoderConfig(
        windows=(WindowSpec(center=0.184, width=0.100),),
        temporal_bins=4,
        feature_kind="evoked_dct",
        covariance_max_channels=64,
        autoencoder="linear_pca",
        latent_dim=4,
        feature_scaling="standard",
        classifier_c=1.0,
        classifier_class_weight="balanced",
        classifier_max_iter=200,
        random_seed=13,
        mlp_activation="relu",
        mlp_alpha=1e-4,
        mlp_learning_rate_init=1e-3,
        mlp_max_iter=10,
        mlp_batch_size="auto",
        mlp_early_stopping=False,
        mlp_validation_fraction=0.1,
        mlp_tol=1e-4,
    )

    latent = _fit_autoencoder_latents(x_source, x_target, cfg)
    probabilities = _predict_source_classifier(latent.z_source, y_source, latent.z_target, cfg, n_classes=3)
    metrics = _classification_metrics(probabilities, y_target_for_metrics_only, n_classes=3)

    assert latent.z_source.shape == (24, 4)
    assert latent.z_target.shape == (9, 4)
    assert probabilities.shape == (9, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(latent.reconstruction_mse_all)
    assert np.isfinite(latent.reconstruction_mse_source)
    assert np.isfinite(latent.reconstruction_mse_target)
    assert np.isfinite(metrics["balanced_accuracy"])
