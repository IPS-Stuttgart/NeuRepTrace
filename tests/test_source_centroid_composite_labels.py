from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_centroid import fit_source_centroid_decoder


def test_source_centroid_preserves_composite_labels() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [1.5], [2.0]], dtype=float)
    source_labels = [["face", "early"], ["face", "early"], ["tool", "late"], ["tool", "late"]]
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"use_diagonal_scale": False},
    )

    assert result.classes.tolist() == [("face", "early"), ("tool", "late")]
    assert result.predictions.tolist() == [("face", "early"), ("tool", "late")]
    assert result.probabilities.shape == (2, 2)
    assert result.metadata["source_centroid_n_classes"] == 2


def test_source_centroid_canonicalizes_dict_label_order() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [1.5], [2.0]], dtype=float)
    source_labels = [
        {"stimulus": "face", "stage": "early"},
        {"stage": "early", "stimulus": "face"},
        {"stimulus": "tool", "stage": "late"},
        {"stage": "late", "stimulus": "tool"},
    ]
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"use_diagonal_scale": False},
    )

    assert result.classes.tolist() == [
        (("stage", "early"), ("stimulus", "face")),
        (("stage", "late"), ("stimulus", "tool")),
    ]
    assert result.predictions.tolist() == result.classes.tolist()
    assert result.metadata["source_centroid_n_classes"] == 2


def test_source_centroid_groups_repeated_scalar_nan_labels() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [1.5], [2.0]], dtype=float)
    source_labels = [float("nan"), float("nan"), "seen", "seen"]
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"use_diagonal_scale": False},
    )

    assert result.classes.shape == (2,)
    assert np.isnan(result.classes[0])
    assert result.classes[1] == "seen"
    assert np.isnan(result.predictions[0])
    assert result.predictions[1] == "seen"
    assert result.metadata["source_centroid_n_classes"] == 2
    assert np.allclose(result.centroids.ravel(), np.asarray([-1.75, 1.75], dtype=float))


def test_source_centroid_groups_nan_inside_composite_labels() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [1.5], [2.0]], dtype=float)
    source_labels = [("face", float("nan")), ("face", float("nan")), ("tool", "late"), ("tool", "late")]
    test_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"use_diagonal_scale": False},
    )

    assert result.classes.shape == (2,)
    assert result.classes[0][0] == "face"
    assert np.isnan(result.classes[0][1])
    assert result.classes[1] == ("tool", "late")
    assert result.predictions[0][0] == "face"
    assert np.isnan(result.predictions[0][1])
    assert result.predictions[1] == ("tool", "late")
    assert result.metadata["source_centroid_n_classes"] == 2
