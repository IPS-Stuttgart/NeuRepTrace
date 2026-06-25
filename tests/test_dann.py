import numpy as np
import pytest

from neureptrace.decoding.dann import (
    DANN_PROTOCOL,
    _bounded_float,
    _integer,
    _nonnegative_float,
    _positive_float,
    _positive_int,
    fit_dann_predict_proba,
)



def test_dann_helpers_reject_fractional_integer_parameters():
    with pytest.raises(ValueError, match="dann_max_epochs must be an integer"):
        _positive_int(1.5, "dann_max_epochs")

    with pytest.raises(ValueError, match="dann_random_state must be an integer"):
        _integer(13.5, "dann_random_state")



def test_dann_helpers_reject_bool_numeric_parameters():
    with pytest.raises(ValueError, match="dann_batch_size must be an integer"):
        _positive_int(True, "dann_batch_size")

    with pytest.raises(ValueError, match="dann_learning_rate"):
        _positive_float(True, "dann_learning_rate")

    with pytest.raises(ValueError, match="dann_weight_decay"):
        _nonnegative_float(True, "dann_weight_decay")

    with pytest.raises(ValueError, match="dann_validation_fraction"):
        _bounded_float(True, "dann_validation_fraction", lower=0.0, upper=1.0)



def test_fit_dann_predict_proba_marks_unlabeled_target_adaptation():
    pytest.importorskip("torch")
    rng = np.random.default_rng(13)
    source_features = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(12, 3)),
            rng.normal(loc=1.0, scale=0.2, size=(12, 3)),
        ]
    )
    source_labels = np.repeat([0, 1], 12)
    target_features = np.vstack(
        [
            rng.normal(loc=-0.8, scale=0.2, size=(5, 3)),
            rng.normal(loc=0.8, scale=0.2, size=(5, 3)),
        ]
    )

    result = fit_dann_predict_proba(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=4,
        batch_size=8,
        patience=2,
        random_state=7,
        device="cpu",
    )

    assert result.probabilities.shape == (10, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.metadata["dann_protocol"] == DANN_PROTOCOL
    assert result.metadata["dann_uses_target_features"] is True
    assert result.metadata["dann_uses_target_labels"] is False
    assert result.metadata["dann_valid_for_benchmark"] is True
    assert result.metadata["dann_target_rows"] == 10



def test_fit_dann_accepts_single_column_source_labels():
    pytest.importorskip("torch")
    rng = np.random.default_rng(23)
    source_features = np.vstack(
        [
            rng.normal(loc=-1.0, scale=0.2, size=(8, 3)),
            rng.normal(loc=1.0, scale=0.2, size=(8, 3)),
        ]
    )
    source_labels = np.repeat([0, 1], 8).reshape(-1, 1)
    target_features = np.vstack(
        [
            rng.normal(loc=-0.8, scale=0.2, size=(3, 3)),
            rng.normal(loc=0.8, scale=0.2, size=(3, 3)),
        ]
    )

    result = fit_dann_predict_proba(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=3,
        batch_size=4,
        patience=1,
        random_state=7,
        device="cpu",
    )

    assert result.probabilities.shape == (6, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)



def test_fit_dann_rejects_multi_column_source_labels():
    with pytest.raises(ValueError, match="source_labels must be one-dimensional"):
        fit_dann_predict_proba(
            source_features=np.zeros((4, 3)),
            source_labels=np.zeros((4, 2)),
            target_features=np.zeros((2, 3)),
            max_epochs=1,
            device="cpu",
        )



def test_fit_dann_rejects_feature_dimension_mismatch():
    with pytest.raises(ValueError, match="same feature dimension"):
        fit_dann_predict_proba(
            source_features=np.zeros((4, 3)),
            source_labels=np.array([0, 1, 0, 1]),
            target_features=np.zeros((2, 2)),
            max_epochs=1,
            device="cpu",
        )
