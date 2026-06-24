from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from neureptrace.bushmeg_category2_autoencoder_loso import _category2_config


def _base_config() -> dict[str, Any]:
    return {
        "category2_autoencoder_loso": {
            "window_size": 0.1,
            "window_centers": [0.184],
            "temporal_bins": 4,
            "feature_kind": "evoked_dct",
            "covariance_max_channels": 64,
            "autoencoder": "linear_pca",
            "latent_dim": 4,
            "feature_scaling": "standard",
            "classifier_c": 1.0,
            "classifier_class_weight": "balanced",
            "classifier_max_iter": 200,
            "random_seed": 13,
            "mlp_activation": "relu",
            "mlp_alpha": 1e-4,
            "mlp_learning_rate_init": 1e-3,
            "mlp_max_iter": 10,
            "mlp_batch_size": "auto",
            "mlp_early_stopping": False,
            "mlp_validation_fraction": 0.1,
            "mlp_tol": 1e-4,
        }
    }


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("latent_dim", True, "latent_dim must be an integer"),
        ("temporal_bins", np.bool_(True), "temporal_bins must be an integer"),
        ("classifier_c", True, "classifier_c must be a finite floating-point value"),
        ("window_centers", [np.bool_(True)], "window_centers must be a finite floating-point value"),
    ],
)
def test_category2_config_rejects_boolean_numeric_values(key: str, value: Any, message: str) -> None:
    config = _base_config()
    config["category2_autoencoder_loso"][key] = value

    with pytest.raises(ValueError, match=message):
        _category2_config(config)
