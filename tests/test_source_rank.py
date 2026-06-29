from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_rank import (
    SOURCE_RANK_CATEGORY,
    fit_source_rank_reference,
    fit_source_rank_transform,
    normalize_rank_output,
    transform_source_rank_features,
)


def test_source_rank_reference_transforms_source_and_eval_rows() -> None:
    source = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]], dtype=float)
    eval_features = np.asarray([[1.5, 11.5], [5.0, 8.0]], dtype=float)

    result = fit_source_rank_transform(source_features=source, eval_features=eval_features, output="uniform")

    assert result.train_features.shape == source.shape
    assert result.eval_features.shape == eval_features.shape
    assert np.all(result.train_features > 0.0)
    assert np.all(result.train_features < 1.0)
    assert result.metadata["source_rank_protocol_category"] == SOURCE_RANK_CATEGORY
    assert result.metadata["source_rank_uses_source_features"] is True
    assert result.metadata["source_rank_uses_eval_features_for_fitting"] is False
    assert result.metadata["source_rank_uses_eval_labels"] is False
    assert result.metadata["source_rank_valid_for_strict_source_only"] is True


def test_centered_output_is_in_minus_one_one_range() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    eval_features = np.asarray([[-1.0], [1.5], [4.0]], dtype=float)

    result = fit_source_rank_transform(source_features=source, eval_features=eval_features, output="centered")

    assert np.all(result.eval_features >= -1.0)
    assert np.all(result.eval_features <= 1.0)
    assert result.metadata["source_rank_output"] == "centered"


def test_transform_source_rank_features_handles_ties_by_midrank() -> None:
    reference = fit_source_rank_reference([[0.0], [1.0], [1.0], [2.0]], clip_extremes=False)

    transformed = transform_source_rank_features([[1.0]], reference)

    assert np.allclose(transformed, np.asarray([[0.5]], dtype=np.float32))


def test_clip_extremes_string_false_keeps_unclipped_rank_extremes() -> None:
    result = fit_source_rank_transform(source_features=[[0.0], [1.0]], eval_features=[[-1.0], [2.0]], clip_extremes="false")

    assert result.reference.clip_extremes is False
    assert result.metadata["source_rank_clip_extremes"] is False
    assert np.allclose(result.eval_features, np.asarray([[0.0], [1.0]], dtype=np.float32))


def test_clip_extremes_rejects_ambiguous_config_values() -> None:
    with pytest.raises(ValueError, match="clip_extremes"):
        fit_source_rank_reference([[0.0], [1.0]], clip_extremes="disabled")

    with pytest.raises(ValueError, match="clip_extremes"):
        fit_source_rank_reference([[0.0], [1.0]], clip_extremes=2)


def test_epsilon_rejects_array_valued_config_values() -> None:
    for bad_epsilon in (np.asarray(1e-6), np.asarray([1e-6]), np.asarray([[1e-6]])):
        with pytest.raises(ValueError, match="epsilon"):
            fit_source_rank_reference([[0.0], [1.0]], epsilon=bad_epsilon)


def test_epsilon_accepts_numpy_numeric_scalar() -> None:
    reference = fit_source_rank_reference([[0.0], [1.0]], epsilon=np.float64(1e-5))

    assert reference.epsilon == pytest.approx(1e-5)


def test_rank_output_aliases_and_validation() -> None:
    assert normalize_rank_output("percentile") == "uniform"
    assert normalize_rank_output("signed") == "centered"

    with pytest.raises(ValueError, match="rank output"):
        normalize_rank_output("bad")

    with pytest.raises(ValueError, match="epsilon"):
        fit_source_rank_reference([[0.0], [1.0]], epsilon=0.8)


def test_rank_transform_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_rank_transform(source_features=[[0.0, 1.0]], eval_features=[[0.0]])


def test_eval_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_rank_transform(source_features=[[0.0], [1.0]], eval_features=[[0.5]], eval_labels=[0])  # type: ignore[call-arg]
