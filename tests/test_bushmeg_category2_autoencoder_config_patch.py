from __future__ import annotations

import importlib.abc
from typing import Any

import numpy as np
import pytest

from neureptrace._bushmeg_category2_autoencoder_config_patch import _Category2AutoencoderConfigPatchLoader
from neureptrace.bushmeg_category2_autoencoder_loso import _category2_config


class _RunpyCompatibleLoader(importlib.abc.Loader):
    def __init__(self) -> None:
        self.requested_fullname: str | None = None

    def create_module(self, spec):  # type: ignore[override]
        return None

    def exec_module(self, module) -> None:
        module.loaded_by_dummy = True

    def get_code(self, fullname: str):
        self.requested_fullname = fullname
        return compile("loaded_value = 7", "<dummy-category2-loader>", "exec")


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


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("latent_dim", 1.5, "latent_dim must be an integer"),
        ("temporal_bins", np.float64(2.5), "temporal_bins must be an integer"),
        ("classifier_max_iter", 200.5, "classifier_max_iter must be an integer"),
        ("mlp_batch_size", 8.25, "mlp_batch_size must be an integer"),
    ],
)
def test_category2_config_rejects_fractional_integer_values(key: str, value: Any, message: str) -> None:
    config = _base_config()
    config["category2_autoencoder_loso"][key] = value

    with pytest.raises(ValueError, match=message):
        _category2_config(config)


def test_category2_config_accepts_integer_like_float_values() -> None:
    config = _base_config()
    section = config["category2_autoencoder_loso"]
    section["temporal_bins"] = 4.0
    section["latent_dim"] = np.float64(3.0)
    section["classifier_max_iter"] = 201.0
    section["mlp_batch_size"] = 8.0

    parsed = _category2_config(config)

    assert parsed.temporal_bins == 4
    assert parsed.latent_dim == 3
    assert parsed.classifier_max_iter == 201
    assert parsed.mlp_batch_size == 8


def test_category2_config_patch_loader_delegates_get_code_for_module_execution() -> None:
    wrapped = _RunpyCompatibleLoader()
    loader = _Category2AutoencoderConfigPatchLoader(wrapped)

    code = loader.get_code("neureptrace.bushmeg_category2_autoencoder_loso")

    assert wrapped.requested_fullname == "neureptrace.bushmeg_category2_autoencoder_loso"
    assert code is not None
