from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_temperature import (
    SOURCE_TEMPERATURE_CATEGORY,
    SourceTemperatureConfig,
    apply_temperature,
    fit_source_temperature_scaling,
    negative_log_likelihood,
    source_temperature_config,
)


def test_fit_source_temperature_scaling_selects_source_temperature_and_scales_test_rows() -> None:
    source_probabilities = np.asarray(
        [
            [0.70, 0.30],
            [0.65, 0.35],
            [0.35, 0.65],
            [0.30, 0.70],
        ],
        dtype=float,
    )
    source_labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    test_probabilities = np.asarray([[0.60, 0.40], [0.20, 0.80]], dtype=float)

    result = fit_source_temperature_scaling(
        source_probabilities=source_probabilities,
        source_labels=source_labels,
        test_probabilities=test_probabilities,
        classes=["a", "b"],
        config={"temperatures": [0.5, 1.0, 2.0]},
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.temperature in {0.5, 1.0, 2.0}
    assert set(result.source_losses) == {0.5, 1.0, 2.0}
    assert result.metadata["source_temperature_protocol_category"] == SOURCE_TEMPERATURE_CATEGORY
    assert result.metadata["source_temperature_uses_test_probabilities_for_fitting"] is False
    assert result.metadata["source_temperature_uses_test_labels"] is False
    assert result.metadata["source_temperature_valid_for_strict_source_only"] is True


def test_apply_temperature_sharpens_and_smooths_probabilities() -> None:
    probabilities = np.asarray([[0.75, 0.25]], dtype=float)

    sharpened = apply_temperature(probabilities, temperature=0.5)
    smoothed = apply_temperature(probabilities, temperature=2.0)

    assert sharpened[0, 0] > probabilities[0, 0]
    assert smoothed[0, 0] < probabilities[0, 0]
    assert np.allclose(sharpened.sum(axis=1), 1.0)
    assert np.allclose(smoothed.sum(axis=1), 1.0)


def test_negative_log_likelihood_validates_labels() -> None:
    with pytest.raises(ValueError, match="outside"):
        negative_log_likelihood([[0.5, 0.5]], [2])


def test_negative_log_likelihood_accepts_integer_like_column_labels() -> None:
    value = negative_log_likelihood([[0.25, 0.75], [0.8, 0.2]], [[1.0], [0]])

    assert np.isclose(value, -np.mean(np.log([0.75, 0.8])))


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        ([[0.5, 0.5]], [True], "integer"),
        ([[0.5, 0.5]], [np.bool_(False)], "integer"),
        ([[0.5, 0.5]], [0.5], "integer"),
        ([[0.5, 0.5]], [np.nan], "integer"),
        ([[0.5, 0.5]], [np.inf], "integer"),
        ([[0.5, 0.5], [0.4, 0.6]], [[0, 1]], "one value"),
    ],
)
def test_negative_log_likelihood_rejects_malformed_label_indices(probabilities, labels, message) -> None:
    with pytest.raises(ValueError, match=message):
        negative_log_likelihood(probabilities, labels)


def test_source_temperature_config_parses_grid() -> None:
    cfg = source_temperature_config(temperatures="0.5,1,2", epsilon="1e-9")

    assert cfg.temperatures == (0.5, 1.0, 2.0)
    assert np.isclose(cfg.epsilon, 1e-9)

    with pytest.raises(ValueError, match="temperatures"):
        source_temperature_config(temperatures=[])


def test_source_temperature_config_accepts_numpy_numeric_scalars() -> None:
    cfg = source_temperature_config(temperatures=[np.float64(0.5), np.int64(1)], epsilon=np.float64(1e-9))

    assert cfg.temperatures == (0.5, 1.0)
    assert np.isclose(cfg.epsilon, 1e-9)


@pytest.mark.parametrize(
    "call",
    [
        lambda: source_temperature_config(temperatures=[True]),
        lambda: source_temperature_config(temperatures=[np.bool_(True)]),
        lambda: source_temperature_config(temperatures=[np.asarray([1.0])]),
        lambda: source_temperature_config(epsilon=True),
        lambda: source_temperature_config(epsilon=np.asarray([1e-9])),
        lambda: apply_temperature([[0.5, 0.5]], temperature=True),
        lambda: apply_temperature([[0.5, 0.5]], temperature=np.asarray([1.0])),
    ],
)
def test_source_temperature_rejects_boolean_and_array_scalar_controls(call) -> None:
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize("bad_epsilon", [0.0, -1e-12, 1.0, 2.0, np.nan, np.inf])
def test_source_temperature_rejects_invalid_probability_floor(bad_epsilon) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        source_temperature_config(epsilon=bad_epsilon)

    with pytest.raises(ValueError, match="epsilon"):
        apply_temperature([[0.5, 0.5]], temperature=1.0, epsilon=bad_epsilon)

    with pytest.raises(ValueError, match="epsilon"):
        negative_log_likelihood([[0.5, 0.5]], [0], epsilon=bad_epsilon)


@pytest.mark.parametrize(
    "bad_config",
    [
        SourceTemperatureConfig(temperatures=(True,)),  # type: ignore[arg-type]
        SourceTemperatureConfig(temperatures=(np.asarray([1.0]),)),  # type: ignore[arg-type]
        SourceTemperatureConfig(temperatures=(1.0,), epsilon=np.asarray([1e-9])),  # type: ignore[arg-type]
        SourceTemperatureConfig(temperatures=(1.0,), epsilon=1.0),
    ],
)
def test_fit_source_temperature_scaling_revalidates_direct_config_objects(bad_config) -> None:
    with pytest.raises(ValueError):
        fit_source_temperature_scaling(
            source_probabilities=[[0.6, 0.4], [0.4, 0.6]],
            source_labels=[0, 1],
            test_probabilities=[[0.5, 0.5]],
            config=bad_config,
        )


def test_source_temperature_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="class width"):
        fit_source_temperature_scaling(
            source_probabilities=[[0.5, 0.5]],
            source_labels=[0],
            test_probabilities=[[0.3, 0.3, 0.4]],
        )


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_temperature_scaling(
            source_probabilities=[[0.5, 0.5], [0.4, 0.6]],
            source_labels=[0, 1],
            test_probabilities=[[0.5, 0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )


def test_fit_source_temperature_scaling_preserves_tuple_labels_and_classes() -> None:
    source_probabilities = np.asarray(
        [
            [0.90, 0.10],
            [0.80, 0.20],
            [0.20, 0.80],
            [0.10, 0.90],
        ],
        dtype=float,
    )
    source_labels = [("face", 1), ("face", 1), ("scene", 2), ("scene", 2)]
    test_probabilities = np.asarray([[0.65, 0.35], [0.25, 0.75]], dtype=float)

    result = fit_source_temperature_scaling(
        source_probabilities=source_probabilities,
        source_labels=source_labels,
        test_probabilities=test_probabilities,
        classes=[("face", 1), ("scene", 2)],
        config={"temperatures": [0.5, 1.0, 2.0]},
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert set(result.source_losses) == {0.5, 1.0, 2.0}
