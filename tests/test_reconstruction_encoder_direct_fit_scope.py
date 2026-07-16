import numpy as np
import pytest

from neureptrace.decoding.reconstruction_encoder import (
    RECONSTRUCTION_SOURCE_ONLY,
    RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL,
    ReconstructionEncoderConfig,
    fit_reconstruction_latent_space,
)


def _source_target_features():
    source = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    target = np.array(
        [
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )
    return source, target


def test_direct_reconstruction_config_normalizes_source_only_alias():
    source, target = _source_target_features()

    result = fit_reconstruction_latent_space(
        train_features=source,
        test_features=target,
        config=ReconstructionEncoderConfig(
            n_components=1,
            fit_scope="strict-source-only",
        ),
    )

    assert result.metadata["representation_fit_scope"] == RECONSTRUCTION_SOURCE_ONLY
    assert result.metadata["representation_protocol"] == RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL
    assert result.metadata["representation_uses_unlabeled_target_data"] is False
    assert result.metadata["representation_fit_rows"] == source.shape[0]


def test_direct_reconstruction_config_rejects_unknown_fit_scope():
    source, target = _source_target_features()

    with pytest.raises(ValueError, match="Unknown reconstruction fit scope"):
        fit_reconstruction_latent_space(
            train_features=source,
            test_features=target,
            config=ReconstructionEncoderConfig(
                n_components=1,
                fit_scope="typo_source_only",
            ),
        )
