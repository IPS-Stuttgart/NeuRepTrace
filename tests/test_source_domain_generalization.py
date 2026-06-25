import numpy as np
import pytest

from neureptrace.decoding.source_domain_generalization import (
    SOURCE_ADVERSARIAL_PROTOCOL,
    SOURCE_GROUP_DRO_PROTOCOL,
    fit_source_adversarial_predict_proba,
    fit_source_domain_generalization_predict_proba,
    normalize_source_domain_generalization_strategy,
)


def _toy_source_problem(seed=13):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.repeat([0, 1], 6), 3)
    domains = np.repeat(["sub-01", "sub-02", "sub-03"], 12)
    class_signal = np.where(labels == 0, -1.0, 1.0).reshape(-1, 1)
    domain_offsets = {"sub-01": -0.5, "sub-02": 0.0, "sub-03": 0.5}
    domain_signal = np.asarray([domain_offsets[domain] for domain in domains]).reshape(-1, 1)
    source_features = np.hstack(
        [
            class_signal + rng.normal(scale=0.05, size=(labels.shape[0], 1)),
            domain_signal + rng.normal(scale=0.05, size=(labels.shape[0], 1)),
            rng.normal(scale=0.05, size=(labels.shape[0], 2)),
        ]
    )
    test_features = rng.normal(size=(7, source_features.shape[1]))
    return source_features, labels, domains, test_features


def test_fit_source_adversarial_predict_proba_marks_source_only_protocol():
    pytest.importorskip("torch")
    source_features, labels, domains, test_features = _toy_source_problem()

    result = fit_source_adversarial_predict_proba(
        source_features=source_features,
        source_labels=labels,
        source_domains=domains,
        test_features=test_features,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=4,
        batch_size=8,
        patience=2,
        random_state=7,
        device="cpu",
    )

    assert result.probabilities.shape == (7, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.metadata["source_adversarial_protocol"] == SOURCE_ADVERSARIAL_PROTOCOL
    assert result.metadata["source_adversarial_uses_target_features"] is False
    assert result.metadata["source_adversarial_uses_target_labels"] is False
    assert result.metadata["source_adversarial_valid_for_benchmark"] is True
    assert result.metadata["source_adversarial_source_domains"] == 3
    assert result.metadata["source_adversarial_test_rows"] == 7
    assert result.metadata["source_adversarial_source_validation_mode"] == "heldout_source_domain"
    assert result.metadata["source_domain_generalization_protocol"] == SOURCE_ADVERSARIAL_PROTOCOL
    assert result.metadata["source_domain_generalization_uses_target_features"] is False


def test_source_domain_generalization_accepts_single_column_source_labels():
    pytest.importorskip("torch")
    source_features, labels, domains, test_features = _toy_source_problem(seed=29)

    result = fit_source_adversarial_predict_proba(
        source_features=source_features,
        source_labels=labels.reshape(-1, 1),
        source_domains=domains,
        test_features=test_features,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=3,
        batch_size=8,
        patience=1,
        random_state=7,
        device="cpu",
    )

    assert result.probabilities.shape == (7, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.metadata["source_domain_generalization_source_rows"] == labels.shape[0]


def test_fit_group_dro_predict_proba_marks_source_only_protocol():
    pytest.importorskip("torch")
    source_features, labels, domains, test_features = _toy_source_problem(seed=17)

    result = fit_source_domain_generalization_predict_proba(
        strategy="group-dro",
        source_features=source_features,
        source_labels=labels,
        source_domains=domains,
        test_features=test_features,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=4,
        batch_size=8,
        patience=2,
        group_dro_eta=0.1,
        random_state=11,
        device="cpu",
    )

    assert result.probabilities.shape == (7, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.metadata["source_domain_generalization"] is True
    assert result.metadata["source_domain_generalization_strategy"] == "group_dro"
    assert result.metadata["source_domain_generalization_protocol"] == SOURCE_GROUP_DRO_PROTOCOL
    assert result.metadata["source_domain_generalization_uses_target_features"] is False
    assert result.metadata["source_domain_generalization_uses_target_labels"] is False
    assert result.metadata["source_domain_generalization_valid_for_benchmark"] is True
    assert result.metadata["source_domain_generalization_validation_mode"] == "heldout_source_domain"
    assert result.metadata["group_dro_source_domains"] == 3
    assert result.metadata["group_dro_final_group_weights"]


def test_source_domain_generalization_erm_uses_same_protocol_contract():
    pytest.importorskip("torch")
    source_features, labels, domains, test_features = _toy_source_problem(seed=19)

    result = fit_source_domain_generalization_predict_proba(
        strategy="neural-erm",
        source_features=source_features,
        source_labels=labels,
        source_domains=domains,
        test_features=test_features,
        hidden_units=8,
        embedding_dim=4,
        max_epochs=3,
        batch_size=8,
        patience=2,
        random_state=3,
        device="cpu",
    )

    assert result.probabilities.shape == (7, 2)
    assert result.metadata["source_domain_generalization_strategy"] == "erm"
    assert result.metadata["source_domain_generalization_uses_target_features"] is False
    assert result.metadata["source_domain_generalization_uses_target_labels"] is False


def test_source_domain_generalization_rejects_multi_column_source_labels():
    with pytest.raises(ValueError, match="source_labels must be one-dimensional"):
        fit_source_adversarial_predict_proba(
            source_features=np.zeros((6, 3)),
            source_labels=np.zeros((6, 2)),
            source_domains=np.array(["a", "a", "b", "b", "c", "c"]),
            test_features=np.zeros((2, 3)),
            max_epochs=1,
            device="cpu",
        )


def test_source_adversarial_rejects_target_feature_dimension_mismatch():
    with pytest.raises(ValueError, match="same feature dimension"):
        fit_source_adversarial_predict_proba(
            source_features=np.zeros((6, 3)),
            source_labels=np.array([0, 1, 0, 1, 0, 1]),
            source_domains=np.array(["a", "a", "b", "b", "c", "c"]),
            test_features=np.zeros((2, 2)),
            max_epochs=1,
            device="cpu",
        )


def test_source_adversarial_rejects_single_source_domain():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="at least two source domains"):
        fit_source_adversarial_predict_proba(
            source_features=np.zeros((6, 3)),
            source_labels=np.array([0, 1, 0, 1, 0, 1]),
            source_domains=np.array(["sub-01"] * 6),
            test_features=np.zeros((2, 3)),
            max_epochs=1,
            device="cpu",
        )


def test_source_adversarial_rejects_missing_source_domain():
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="source_domains must not contain missing values"):
        fit_source_adversarial_predict_proba(
            source_features=np.zeros((6, 3)),
            source_labels=np.array([0, 1, 0, 1, 0, 1]),
            source_domains=np.array(["sub-01", "sub-01", "sub-02", "sub-02", None, "sub-03"], dtype=object),
            test_features=np.zeros((2, 3)),
            max_epochs=1,
            device="cpu",
        )


def test_normalize_source_domain_generalization_strategy_aliases():
    assert normalize_source_domain_generalization_strategy("source-adversarial") == "subject_adversarial"
    assert normalize_source_domain_generalization_strategy("groupdro") == "group_dro"
    assert normalize_source_domain_generalization_strategy("source-erm") == "erm"
    with pytest.raises(ValueError, match="Unknown source-domain generalization strategy"):
        normalize_source_domain_generalization_strategy("target-adaptive")
