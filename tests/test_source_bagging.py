from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_bagging import SOURCE_BAGGING_CATEGORY, SourceBaggingConfig, fit_source_bagging_decoder, source_bagging_config


def test_source_bagging_predicts_separated_classes() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "left", "right", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"n_estimators": 5, "random_state": 7},
    )

    assert result.predictions.tolist() == ["left", "right"]
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.n_estimators == 5
    assert result.metadata["source_bagging_protocol_category"] == SOURCE_BAGGING_CATEGORY
    assert result.metadata["source_bagging_valid_for_strict_source_only"] is True


def test_source_bagging_preserves_composite_labels() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [1.5], [2.0]], dtype=float)
    source_labels = [["face", "early"], ["face", "early"], ["tool", "late"], ["tool", "late"]]
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"n_estimators": 5, "random_state": 7},
    )

    assert result.classes.tolist() == [("face", "early"), ("tool", "late")]
    assert result.predictions.tolist() == [("face", "early"), ("tool", "late")]
    assert result.probabilities.shape == (2, 2)
    assert result.metadata["source_bagging_n_classes"] == 2


def test_source_bagging_feature_fraction_subsamples_features() -> None:
    source_features = np.asarray([[-2.0, 0.0, 1.0, 0.0], [-1.5, 0.2, 1.1, 0.0], [1.5, 0.1, -1.0, 0.0], [2.0, -0.1, -1.1, 0.0]], dtype=float)
    source_labels = np.asarray([0, 0, 1, 1], dtype=object)
    test_features = np.asarray([[-1.8, 0.0, 1.0, 0.0], [1.8, 0.0, -1.0, 0.0]], dtype=float)

    result = fit_source_bagging_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"n_estimators": 3, "feature_fraction": 0.5, "random_state": 11},
    )

    assert len(result.feature_indices) == 3
    assert all(indices.size == 2 for indices in result.feature_indices)


def test_source_bagging_config_validation() -> None:
    cfg = source_bagging_config(n_estimators="3", sample_fraction="0.75", feature_fraction="0.5")
    assert cfg.n_estimators == 3
    assert cfg.sample_fraction == 0.75
    assert cfg.feature_fraction == 0.5

    with pytest.raises(ValueError, match="n_estimators"):
        source_bagging_config(n_estimators=0)


@pytest.mark.parametrize("value", [np.asarray(3), np.asarray([3]), np.asarray(True)])
def test_source_bagging_rejects_array_valued_n_estimators(value: object) -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        source_bagging_config(n_estimators=value)


@pytest.mark.parametrize("option_name", ["sample_fraction", "feature_fraction"])
def test_source_bagging_rejects_fraction_above_one(option_name: str) -> None:
    with pytest.raises(ValueError, match=option_name):
        source_bagging_config(**{option_name: 1.01})


def test_source_bagging_rejects_direct_config_fraction_above_one() -> None:
    source_features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    with pytest.raises(ValueError, match="feature_fraction"):
        fit_source_bagging_decoder(
            source_features=source_features,
            source_labels=source_labels,
            test_features=test_features,
            config=SourceBaggingConfig(n_estimators=1, feature_fraction=1.5),
        )


def test_source_bagging_rejects_direct_config_invalid_n_estimators() -> None:
    source_features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    with pytest.raises(ValueError, match="n_estimators"):
        fit_source_bagging_decoder(
            source_features=source_features,
            source_labels=source_labels,
            test_features=test_features,
            config=SourceBaggingConfig(n_estimators=0),
        )


@pytest.mark.parametrize("value", [None, "", " none ", "NULL"])
def test_source_bagging_random_state_accepts_none_like_values(value: object) -> None:
    cfg = source_bagging_config(random_state=value)

    assert cfg.random_state is None


def test_source_bagging_random_state_accepts_scalar_numpy_seed() -> None:
    cfg = source_bagging_config(random_state=np.asarray(7))

    assert cfg.random_state == 7


@pytest.mark.parametrize("value", [[7], {"seed": 7}, {7}, np.asarray([7])])
def test_source_bagging_random_state_rejects_non_scalar_values(value: object) -> None:
    with pytest.raises(ValueError, match="random_state"):
        source_bagging_config(random_state=value)


def test_source_bagging_epsilon_accepts_scalar_numpy_value() -> None:
    cfg = source_bagging_config(epsilon=np.asarray(1.0e-6))

    assert cfg.epsilon == pytest.approx(1.0e-6)


@pytest.mark.parametrize("value", [True, np.bool_(True), np.asarray(True), np.asarray([1.0e-6]), [1.0e-6], {"eps": 1.0e-6}])
def test_source_bagging_epsilon_rejects_boolean_and_array_values(value: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        source_bagging_config(epsilon=value)


def test_source_bagging_rejects_direct_config_invalid_epsilon() -> None:
    source_features = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    source_labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    with pytest.raises(ValueError, match="epsilon"):
        fit_source_bagging_decoder(
            source_features=source_features,
            source_labels=source_labels,
            test_features=test_features,
            config=SourceBaggingConfig(n_estimators=1, epsilon=np.asarray(True)),
        )


def test_source_bagging_boolean_string_config() -> None:
    cfg = source_bagging_config(bootstrap_rows="false", bootstrap_features="yes", class_balanced="0")

    assert cfg.bootstrap_rows is False
    assert cfg.bootstrap_features is True
    assert cfg.class_balanced is False

    with pytest.raises(ValueError, match="bootstrap_rows"):
        source_bagging_config(bootstrap_rows="not-a-bool")
