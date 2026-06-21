import numpy as np
import pytest

from neureptrace.decoding.cdan import (
    CDAN_CONDITIONAL_MMD_PROTOCOL,
    CDAN_PROTOCOL,
    _boolean,
    fit_cdan_predict_proba,
)


def _toy_domain_shift(seed=19):
    rng = np.random.default_rng(seed)
    source_features = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(10, 3)),
            rng.normal(loc=1.0, scale=0.2, size=(10, 3)),
        ]
    )
    source_labels = np.repeat([0, 1], 10)
    target_features = np.vstack(
        [
            rng.normal(loc=-0.75, scale=0.2, size=(5, 3)),
            rng.normal(loc=0.75, scale=0.2, size=(5, 3)),
        ]
    )
    return source_features, source_labels, target_features


def test_cdan_predict_proba_marks_protocol_2_without_target_labels():
    pytest.importorskip("torch")
    source_features, source_labels, target_features = _toy_domain_shift()

    result = fit_cdan_predict_proba(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=2,
        batch_size=8,
        patience=1,
        cdan_loss_weight=0.2,
        cdan_entropy_conditioning=True,
        random_state=7,
        device="cpu",
    )

    assert result.probabilities.shape == (10, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.metadata["cdan_protocol"] == CDAN_PROTOCOL
    assert result.metadata["cdan_uses_target_features"] is True
    assert result.metadata["cdan_uses_target_labels"] is False
    assert result.metadata["cdan_valid_for_benchmark"] is True
    assert result.metadata["cdan_target_rows"] == 10
    assert result.metadata["cdan_entropy_conditioning"] is True


def test_cdan_can_combine_randomized_multilinear_map_and_conditional_mmd():
    pytest.importorskip("torch")
    source_features, source_labels, target_features = _toy_domain_shift(seed=23)

    result = fit_cdan_predict_proba(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=2,
        batch_size=8,
        patience=1,
        cdan_loss_weight=0.2,
        cdan_randomized_dim=7,
        conditional_mmd_loss_weight=0.1,
        mmd_kernel_num=3,
        random_state=11,
        device="cpu",
    )

    assert result.probabilities.shape == (10, 2)
    assert result.metadata["cdan_protocol"] == CDAN_CONDITIONAL_MMD_PROTOCOL
    assert result.metadata["cdan_randomized_dim"] == 7
    assert result.metadata["cdan_input_dim"] == 7
    assert result.metadata["cdan_conditional_mmd_loss_weight"] == 0.1
    assert result.metadata["cdan_mmd_adaptation"] is True


def test_cdan_rejects_target_dimension_mismatch():
    with pytest.raises(ValueError, match="same feature dimension"):
        fit_cdan_predict_proba(
            source_features=np.zeros((4, 3)),
            source_labels=np.array([0, 1, 0, 1]),
            target_features=np.zeros((2, 2)),
            max_epochs=1,
            device="cpu",
        )


def test_cdan_boolean_parameter_rejects_ambiguous_values():
    assert _boolean("yes", "cdan_entropy_conditioning") is True
    assert _boolean("off", "cdan_entropy_conditioning") is False
    with pytest.raises(ValueError, match="cdan_entropy_conditioning must be a boolean"):
        _boolean("maybe", "cdan_entropy_conditioning")
