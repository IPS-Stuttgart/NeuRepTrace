import numpy as np
import pytest

from neureptrace.decoding.reconstruction_encoder import (
    DEEP_MASKED_RECONSTRUCTION_ENCODER_METHOD,
    RECONSTRUCTION_MASKED_AUTOENCODER,
    RECONSTRUCTION_SOURCE_ONLY,
    RECONSTRUCTION_SOURCE_PLUS_TARGET,
    RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL,
    RECONSTRUCTION_UNLABELED_TARGET_PROTOCOL,
    TorchMaskedReconstructionEncoder,
    fit_reconstruction_latent_classifier,
    fit_reconstruction_latent_space,
    normalize_reconstruction_encoder_kind,
    normalize_reconstruction_fit_scope,
    reconstruction_encoder_config,
)


def _toy_source_target(seed=3):
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], 16)
    prototypes = np.array([[2.0, 0.0, 0.3, 0.0], [0.0, 2.0, 0.0, 0.3]])
    source = prototypes[labels] + 0.05 * rng.normal(size=(labels.size, 4))
    target = prototypes[labels] + np.array([4.0, -2.0, 1.0, 0.5]) + 0.05 * rng.normal(size=(labels.size, 4))
    return source, labels, target


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("strict-source-only", RECONSTRUCTION_SOURCE_ONLY),
        ("category_1", RECONSTRUCTION_SOURCE_ONLY),
        ("all-data", RECONSTRUCTION_SOURCE_PLUS_TARGET),
        ("unlabeled_target", RECONSTRUCTION_SOURCE_PLUS_TARGET),
        ("protocol_2", RECONSTRUCTION_SOURCE_PLUS_TARGET),
    ],
)
def test_reconstruction_fit_scope_aliases(alias, expected):
    assert normalize_reconstruction_fit_scope(alias) == expected


@pytest.mark.parametrize("alias", ["deep", "nonlinear", "masked", "deep_masked_autoencoder", "torch_masked_autoencoder", "mae"])
def test_reconstruction_encoder_kind_aliases(alias):
    assert normalize_reconstruction_encoder_kind(alias) == RECONSTRUCTION_MASKED_AUTOENCODER


def test_source_only_reconstruction_is_strict_protocol():
    source, labels, target = _toy_source_target()

    result = fit_reconstruction_latent_space(
        train_features=source,
        test_features=target,
        config=reconstruction_encoder_config(fit_scope="source_only", n_components=2),
    )

    assert result.train_latent.shape == (source.shape[0], 2)
    assert result.test_latent.shape == (target.shape[0], 2)
    assert result.metadata["representation_protocol"] == RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL
    assert result.metadata["representation_uses_unlabeled_target_data"] is False
    assert result.metadata["representation_target_labels_used"] is False
    assert result.metadata["representation_valid_for_strict_source_only"] is True
    assert result.metadata["representation_fit_rows"] == source.shape[0]
    with pytest.raises(ValueError, match="target_encoder_features"):
        fit_reconstruction_latent_space(
            train_features=source,
            test_features=target,
            target_encoder_features=target,
            config=reconstruction_encoder_config(fit_scope="source_only", n_components=2),
        )
    assert labels.shape[0] == source.shape[0]


def test_source_plus_target_reconstruction_is_category_2_without_target_labels():
    source, labels, target = _toy_source_target(seed=5)

    result = fit_reconstruction_latent_space(
        train_features=source,
        test_features=target,
        config=reconstruction_encoder_config(fit_scope="source_plus_target", n_components=2, standardize=True),
    )

    assert result.train_latent.shape == (source.shape[0], 2)
    assert result.test_latent.shape == (target.shape[0], 2)
    assert result.metadata["representation_protocol"] == RECONSTRUCTION_UNLABELED_TARGET_PROTOCOL
    assert result.metadata["representation_uses_unlabeled_target_data"] is True
    assert result.metadata["representation_target_labels_used"] is False
    assert result.metadata["representation_valid_for_strict_source_only"] is False
    assert result.metadata["representation_valid_for_benchmark"] is False
    assert result.metadata["representation_target_feature_source"] == "test_features_transductive"
    assert result.metadata["representation_fit_rows"] == source.shape[0] + target.shape[0]
    assert np.isfinite(result.metadata["representation_train_reconstruction_mse"])
    assert np.isfinite(result.metadata["representation_test_reconstruction_mse"])

    with pytest.raises(ValueError, match="target labels"):
        fit_reconstruction_latent_space(
            train_features=source,
            test_features=target,
            target_labels=labels,
            config=reconstruction_encoder_config(fit_scope="source_plus_target", n_components=2),
        )


def test_source_plus_target_accepts_separate_unlabeled_encoder_block():
    source, _labels, target = _toy_source_target(seed=7)
    calibration_block = target[:8]

    result = fit_reconstruction_latent_space(
        train_features=source,
        test_features=target[8:],
        target_encoder_features=calibration_block,
        config=reconstruction_encoder_config(fit_scope="source_plus_target", n_components="all"),
    )

    assert result.metadata["representation_target_feature_source"] == "target_encoder_features"
    assert result.metadata["representation_fit_rows"] == source.shape[0] + calibration_block.shape[0]
    assert result.metadata["representation_n_components"] == min(source.shape[0] + calibration_block.shape[0], source.shape[1])


def test_reconstruction_latent_classifier_trains_only_on_source_labels():
    source, labels, target = _toy_source_target(seed=11)

    result = fit_reconstruction_latent_classifier(
        train_features=source,
        train_labels=labels,
        test_features=target,
        config=reconstruction_encoder_config(fit_scope="source_plus_target", n_components=2),
    )

    assert result.predictions.shape == (target.shape[0],)
    assert result.probabilities is not None
    assert result.probabilities.shape == (target.shape[0], 2)
    assert result.classes.tolist() == [0, 1]
    assert result.metadata["classifier_label_source"] == "source_train_labels"
    assert result.metadata["classifier_target_labels_used"] is False
    assert result.metadata["representation_protocol"] == RECONSTRUCTION_UNLABELED_TARGET_PROTOCOL

    with pytest.raises(ValueError, match="target labels"):
        fit_reconstruction_latent_classifier(
            train_features=source,
            train_labels=labels,
            test_features=target,
            target_labels=labels,
            config=reconstruction_encoder_config(fit_scope="source_plus_target", n_components=2),
        )


def _requires_torch():
    return pytest.importorskip("torch")


def test_masked_autoencoder_source_plus_target_is_category_2_without_target_labels():
    _requires_torch()
    source, labels, target = _toy_source_target(seed=17)

    result = fit_reconstruction_latent_space(
        train_features=source,
        test_features=target,
        config=reconstruction_encoder_config(
            encoder_kind="masked_autoencoder",
            fit_scope="source_plus_target",
            n_components=3,
            hidden_units=(12,),
            mask_fraction=0.35,
            max_epochs=6,
            batch_size=16,
            validation_fraction=0.2,
            patience=3,
            dropout=0.0,
            random_state=123,
            device="cpu",
        ),
    )

    assert isinstance(result.encoder, TorchMaskedReconstructionEncoder)
    assert result.train_latent.shape == (source.shape[0], 3)
    assert result.test_latent.shape == (target.shape[0], 3)
    assert result.metadata["representation_method"] == DEEP_MASKED_RECONSTRUCTION_ENCODER_METHOD
    assert result.metadata["representation_encoder_kind"] == RECONSTRUCTION_MASKED_AUTOENCODER
    assert result.metadata["representation_is_nonlinear"] is True
    assert result.metadata["representation_masked_modeling"] is True
    assert result.metadata["representation_mask_fraction"] == pytest.approx(0.35)
    assert result.metadata["representation_uses_unlabeled_target_data"] is True
    assert result.metadata["representation_target_labels_used"] is False
    assert result.metadata["representation_protocol"] == RECONSTRUCTION_UNLABELED_TARGET_PROTOCOL
    assert np.isfinite(result.metadata["representation_fit_reconstruction_mse"])

    with pytest.raises(ValueError, match="target labels"):
        fit_reconstruction_latent_space(
            train_features=source,
            test_features=target,
            target_labels=labels,
            config=reconstruction_encoder_config(encoder_kind="masked_autoencoder", fit_scope="source_plus_target", n_components=2, max_epochs=1),
        )


def test_masked_autoencoder_classifier_uses_only_source_labels():
    _requires_torch()
    source, labels, target = _toy_source_target(seed=19)

    result = fit_reconstruction_latent_classifier(
        train_features=source,
        train_labels=labels,
        test_features=target,
        config=reconstruction_encoder_config(
            encoder_kind="deep",
            fit_scope="source_only",
            n_components=2,
            hidden_units=10,
            mask_fraction=0.2,
            max_epochs=5,
            batch_size=16,
            validation_fraction=0.0,
            patience=2,
            dropout=0.0,
            random_state=5,
            device="cpu",
        ),
    )

    assert result.predictions.shape == (target.shape[0],)
    assert result.probabilities is not None
    assert result.metadata["representation_protocol"] == RECONSTRUCTION_STRICT_SOURCE_ONLY_PROTOCOL
    assert result.metadata["representation_uses_unlabeled_target_data"] is False
    assert result.metadata["classifier_target_labels_used"] is False
    assert result.metadata["classifier_label_source"] == "source_train_labels"
