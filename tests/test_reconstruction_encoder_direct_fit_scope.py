import numpy as np
import pytest

from neureptrace.decoding.reconstruction_encoder import (
    RECONSTRUCTION_LINEAR_ENCODER,
    RECONSTRUCTION_SOURCE_ONLY,
    RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL,
    ReconstructionEncoderConfig,
    fit_reconstruction_latent_classifier,
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


def test_direct_reconstruction_config_normalizes_encoder_kind_alias():
    source, target = _source_target_features()

    result = fit_reconstruction_latent_space(
        train_features=source,
        test_features=target,
        config=ReconstructionEncoderConfig(
            n_components=1,
            fit_scope=RECONSTRUCTION_SOURCE_ONLY,
            encoder_kind="pca",
        ),
    )

    assert result.metadata["representation_encoder_kind"] == RECONSTRUCTION_LINEAR_ENCODER


def test_direct_reconstruction_config_normalizes_standardize_false_string():
    source, target = _source_target_features()

    result = fit_reconstruction_latent_space(
        train_features=source,
        test_features=target,
        config=ReconstructionEncoderConfig(
            n_components=1,
            fit_scope=RECONSTRUCTION_SOURCE_ONLY,
            standardize="false",
        ),
    )

    assert result.metadata["representation_standardized"] is False
    np.testing.assert_array_equal(result.encoder.scale_, np.ones(source.shape[1]))


def test_direct_reconstruction_classifier_config_normalizes_numeric_fields():
    source, target = _source_target_features()

    result = fit_reconstruction_latent_classifier(
        train_features=source,
        train_labels=np.array([0, 1, 0]),
        test_features=target,
        config=ReconstructionEncoderConfig(
            n_components=1,
            fit_scope=RECONSTRUCTION_SOURCE_ONLY,
            classifier_max_iter="25",
            classifier_C="2.5",
            random_state="7",
        ),
    )

    assert result.classifier.max_iter == 25
    assert result.classifier.C == pytest.approx(2.5)
    assert result.classifier.random_state == 7


def test_direct_reconstruction_config_rejects_invalid_standardize_value():
    source, target = _source_target_features()

    with pytest.raises(ValueError, match="standardize must be a boolean value"):
        fit_reconstruction_latent_space(
            train_features=source,
            test_features=target,
            config=ReconstructionEncoderConfig(
                n_components=1,
                fit_scope=RECONSTRUCTION_SOURCE_ONLY,
                standardize="sometimes",
            ),
        )


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
