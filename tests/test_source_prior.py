from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding.source_prior import (
    SOURCE_PRIOR_CATEGORY,
    SourcePriorConfig,
    adjust_probabilities_to_source_prior,
    estimate_source_class_prior,
    normalize_target_prior,
    source_prior_config,
)


def test_estimate_source_class_prior_empirical_order() -> None:
    prior, classes = estimate_source_class_prior(["b", "a", "b", "b"])

    assert classes.tolist() == ["b", "a"]
    assert np.allclose(prior, np.asarray([0.75, 0.25]))


def test_uniform_prior_adjustment_reweights_probabilities() -> None:
    probabilities = np.asarray([[0.75, 0.25], [0.50, 0.50]], dtype=float)

    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=["major", "major", "major", "minor"],
        classes=["major", "minor"],
        config={"target_prior": "uniform"},
    )

    assert result.probabilities.shape == probabilities.shape
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.probabilities[0, 1] > probabilities[0, 1]
    assert np.allclose(result.source_prior, np.asarray([0.75, 0.25]))
    assert np.allclose(result.target_prior, np.asarray([0.5, 0.5]))
    assert result.metadata["source_prior_protocol_category"] == SOURCE_PRIOR_CATEGORY
    assert result.metadata["source_prior_uses_source_labels"] is True
    assert result.metadata["source_prior_uses_heldout_features"] is False
    assert result.metadata["source_prior_uses_heldout_labels"] is False
    assert result.metadata["source_prior_valid_for_strict_source_only"] is True


def test_source_prior_accepts_rectangular_numpy_composite_values() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    source_labels = np.asarray([("left", 1), ("right", 2), ("left", 1), ("right", 2)], dtype=object)
    classes = np.asarray([("left", 1), ("right", 2)], dtype=object)

    prior, inferred_classes = estimate_source_class_prior(source_labels, classes=classes)
    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=source_labels,
        classes=classes,
        config={"target_prior": "source"},
    )

    assert inferred_classes.tolist() == [("left", 1), ("right", 2)]
    assert result.classes.tolist() == [("left", 1), ("right", 2)]
    assert np.allclose(prior, np.asarray([0.5, 0.5]))
    assert np.allclose(result.source_prior, np.asarray([0.5, 0.5]))
    assert np.allclose(result.probabilities, probabilities)
    assert result.metadata["source_prior_n_classes"] == 2


def test_source_prior_treats_nan_labels_as_matching_class_values() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    source_labels = np.asarray(["seen", np.nan, "seen", np.nan], dtype=object)
    classes = np.asarray(["seen", np.nan], dtype=object)

    prior, inferred_classes = estimate_source_class_prior(source_labels, classes=classes)
    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=source_labels,
        classes=classes,
        config={"target_prior": "source"},
    )

    assert inferred_classes[0] == "seen"
    assert np.isnan(inferred_classes[1])
    assert result.classes[0] == "seen"
    assert np.isnan(result.classes[1])
    np.testing.assert_allclose(prior, np.asarray([0.5, 0.5]))
    np.testing.assert_allclose(result.source_prior, np.asarray([0.5, 0.5]))
    np.testing.assert_allclose(result.probabilities, probabilities)


def test_source_prior_treats_pandas_missing_labels_as_matching_class_values() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    source_labels = pd.Series(["seen", pd.NA, "seen", pd.NA], dtype="object")
    classes = pd.Index(["seen", pd.NA], dtype="object")

    prior, inferred_classes = estimate_source_class_prior(source_labels, classes=classes)
    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=source_labels,
        classes=classes,
        config={"target_prior": "source"},
    )

    assert inferred_classes[0] == "seen"
    assert pd.isna(inferred_classes[1])
    assert result.classes[0] == "seen"
    assert pd.isna(result.classes[1])
    np.testing.assert_allclose(prior, np.asarray([0.5, 0.5]))
    np.testing.assert_allclose(result.source_prior, np.asarray([0.5, 0.5]))
    np.testing.assert_allclose(result.probabilities, probabilities)


def test_source_prior_treats_composite_pandas_missing_labels_as_matching_class_values() -> None:
    source_labels = [(pd.NA, "cue"), ("seen", "cue"), (pd.NA, "cue")]
    classes = [(pd.NA, "cue"), ("seen", "cue")]

    prior, inferred_classes = estimate_source_class_prior(source_labels, classes=classes)

    assert pd.isna(inferred_classes[0][0])
    assert inferred_classes[0][1] == "cue"
    assert inferred_classes[1] == ("seen", "cue")
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))


def test_source_prior_keeps_none_and_missing_labels_distinct() -> None:
    prior, inferred_classes = estimate_source_class_prior([None, pd.NA, None], classes=[None, pd.NA])

    assert inferred_classes[0] is None
    assert pd.isna(inferred_classes[1])
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))


@pytest.mark.parametrize("nat_value", [np.datetime64("NaT"), np.timedelta64("NaT")])
def test_source_prior_keeps_none_and_numpy_nat_labels_distinct(nat_value) -> None:
    prior, inferred_classes = estimate_source_class_prior([None, nat_value, None], classes=[None, nat_value])

    assert inferred_classes[0] is None
    assert np.isnat(inferred_classes[1])
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))


def test_source_prior_preserves_numpy_datetime64_nat_array_labels() -> None:
    source_labels = np.asarray(["NaT", "2020-01-01", "NaT"], dtype="datetime64[D]")
    classes = np.asarray(["NaT", "2020-01-01"], dtype="datetime64[D]")

    prior, inferred_classes = estimate_source_class_prior(source_labels, classes=classes)

    assert np.isnat(inferred_classes[0])
    assert inferred_classes[1] == np.datetime64("2020-01-01", "D")
    np.testing.assert_allclose(prior, np.asarray([2.0 / 3.0, 1.0 / 3.0]))


def test_source_prior_target_source_is_identity_after_normalization() -> None:
    probabilities = np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=float)

    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=[0, 0, 1],
        classes=[0, 1],
        config={"target_prior": "source"},
    )

    assert np.allclose(result.probabilities, probabilities)


def test_source_prior_smoothing_and_aliases() -> None:
    assert normalize_target_prior("balanced") == "uniform"
    assert normalize_target_prior("empirical") == "source"
    cfg = source_prior_config(target_prior="source-prior", smoothing="1.0")
    assert cfg.target_prior == "source"
    assert cfg.smoothing == 1.0

    prior, _classes = estimate_source_class_prior(["a", "a", "b"], classes=["a", "b"], smoothing=1.0)
    assert np.allclose(prior, np.asarray([0.6, 0.4]))


def test_direct_source_prior_config_normalizes_like_mapping_config() -> None:
    direct_config = SourcePriorConfig(target_prior="source-prior", smoothing=np.asarray(1.0), epsilon=np.float64(1e-6))

    direct = adjust_probabilities_to_source_prior(
        [[0.2, 0.8], [0.7, 0.3]],
        source_labels=[0, 0, 1],
        classes=[0, 1],
        config=direct_config,
    )
    mapping = adjust_probabilities_to_source_prior(
        [[0.2, 0.8], [0.7, 0.3]],
        source_labels=[0, 0, 1],
        classes=[0, 1],
        config={"target_prior": "source-prior", "smoothing": np.asarray(1.0), "epsilon": np.float64(1e-6)},
    )

    assert direct_config.target_prior == "source"
    assert direct_config.smoothing == 1.0
    assert direct_config.epsilon == pytest.approx(1e-6)
    assert np.allclose(direct.probabilities, mapping.probabilities)
    assert np.allclose(direct.source_prior, mapping.source_prior)
    assert np.allclose(direct.target_prior, mapping.target_prior)
    assert direct.metadata["source_prior_target_prior"] == "source"


@pytest.mark.parametrize("smoothing", [True, False, np.bool_(True), np.bool_(False), np.asarray(True), np.asarray(False)])
def test_source_prior_rejects_boolean_smoothing(smoothing) -> None:
    with pytest.raises(ValueError, match="smoothing must be non-negative and finite"):
        source_prior_config(smoothing=smoothing)


@pytest.mark.parametrize("epsilon", [True, False, np.bool_(True), np.bool_(False), np.asarray(True), np.asarray(False)])
def test_source_prior_rejects_boolean_epsilon(epsilon) -> None:
    with pytest.raises(ValueError, match="epsilon must be positive and finite"):
        source_prior_config(epsilon=epsilon)


@pytest.mark.parametrize(
    "probabilities",
    [
        [True, False],
        np.asarray([[True, False]], dtype=bool),
        np.asarray([[True, 0.0]], dtype=object),
        [[np.bool_(True), np.bool_(False)]],
    ],
)
def test_source_prior_rejects_boolean_probability_rows(probabilities) -> None:
    with pytest.raises(ValueError, match="boolean"):
        adjust_probabilities_to_source_prior(probabilities, source_labels=[0, 1], classes=[0, 1])


@pytest.mark.parametrize("field", ["smoothing", "epsilon"])
def test_direct_source_prior_config_rejects_boolean_numeric_controls(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SourcePriorConfig(**{field: True})


@pytest.mark.parametrize("field", ["smoothing", "epsilon"])
def test_direct_source_prior_config_rejects_vector_numeric_controls(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SourcePriorConfig(**{field: np.asarray([1.0, 2.0])})


def test_source_prior_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="target_prior"):
        normalize_target_prior("bad")

    with pytest.raises(ValueError, match="absent from classes"):
        estimate_source_class_prior(["a", "b"], classes=["a"])

    with pytest.raises(ValueError, match="shape"):
        adjust_probabilities_to_source_prior([[0.5, 0.5, 0.0]], source_labels=[0, 1], classes=[0, 1])

    with pytest.raises(ValueError, match="positive mass"):
        adjust_probabilities_to_source_prior([[0.0, 0.0]], source_labels=[0, 1], classes=[0, 1])


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        adjust_probabilities_to_source_prior(
            [[0.5, 0.5]],
            source_labels=[0, 1],
            heldout_labels=[0],  # type: ignore[call-arg]
        )
