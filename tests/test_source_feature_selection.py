from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_feature_selection import (
    SOURCE_FEATURE_SELECTION_CATEGORY,
    SourceFeatureSelectionConfig,
    fit_source_feature_selection,
    normalize_score_method,
    select_top_source_features,
    source_feature_scores,
    source_feature_selection_config,
)


def test_anova_feature_selection_selects_discriminative_columns() -> None:
    source = np.asarray(
        [
            [0.0, 10.0, 0.1],
            [0.2, 10.1, 0.2],
            [5.0, 10.2, 0.1],
            [5.2, 10.3, 0.2],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    test = np.asarray([[0.1, 9.9, 0.0], [5.1, 10.4, 0.3]], dtype=float)

    result = fit_source_feature_selection(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config={"method": "anova", "k": 1},
    )

    assert result.selected_indices.tolist() == [0]
    assert result.train_features.shape == (4, 1)
    assert result.test_features.shape == (2, 1)
    assert result.metadata["source_feature_selection_protocol_category"] == SOURCE_FEATURE_SELECTION_CATEGORY
    assert result.metadata["source_feature_selection_uses_test_features_for_fitting"] is False
    assert result.metadata["source_feature_selection_uses_test_labels"] is False
    assert result.metadata["source_feature_selection_valid_for_strict_source_only"] is True


def test_anova_feature_selection_preserves_composite_label_rows() -> None:
    source = np.asarray(
        [
            [0.0, 10.0, 0.1],
            [0.2, 10.1, 0.2],
            [5.0, 10.2, 0.1],
            [5.2, 10.3, 0.2],
        ],
        dtype=float,
    )
    labels = np.asarray(
        [
            ["animal", "seen"],
            ["animal", "seen"],
            ["tool", "imagined"],
            ["tool", "imagined"],
        ],
        dtype=object,
    )
    test = np.asarray([[0.1, 9.9, 0.0], [5.1, 10.4, 0.3]], dtype=float)

    scores = source_feature_scores(source, labels, method="anova")
    result = fit_source_feature_selection(
        source_features=source,
        source_labels=[("animal", "seen"), ("animal", "seen"), ("tool", "imagined"), ("tool", "imagined")],
        test_features=test,
        config={"method": "anova", "k": 1},
    )

    assert scores[0] > scores[1]
    assert result.selected_indices.tolist() == [0]
    assert result.train_features.shape == (4, 1)
    assert result.test_features.shape == (2, 1)


def test_variance_scores_do_not_require_multiple_classes() -> None:
    source = np.asarray([[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]], dtype=float)
    scores = source_feature_scores(source, ["same", "same", "same"], method="variance")

    assert np.allclose(scores, np.asarray([0.0, 4.0]))


def test_select_top_features_honors_min_score() -> None:
    selected = select_top_source_features([0.1, 10.0, 5.0, 0.0], k=3, min_score=1.0)

    assert selected.tolist() == [1, 2]


def test_aliases_and_validation() -> None:
    assert normalize_score_method("f-score") == "anova"
    assert normalize_score_method("var") == "variance"

    with pytest.raises(ValueError, match="score method"):
        normalize_score_method("bad")

    with pytest.raises(ValueError, match="No features selected"):
        select_top_source_features([0.1, 0.2], k=2, min_score=1.0)


def test_feature_selection_config_rejects_malformed_numeric_controls() -> None:
    for kwargs, match in [
        ({"k": True}, "k"),
        ({"k": np.asarray([1])}, "k"),
        ({"min_score": True}, "min_score"),
        ({"min_score": np.asarray([0.1])}, "min_score"),
        ({"epsilon": False}, "epsilon"),
        ({"epsilon": np.asarray([1e-12])}, "epsilon"),
    ]:
        with pytest.raises(ValueError, match=match):
            source_feature_selection_config(**kwargs)


def test_feature_selection_config_preserves_numpy_numeric_scalars() -> None:
    cfg = source_feature_selection_config(k=np.int64(2), min_score=np.float64(0.1), epsilon=np.float64(1e-9))

    assert cfg.k == 2
    assert cfg.min_score == pytest.approx(0.1)
    assert cfg.epsilon == pytest.approx(1e-9)


def test_select_top_features_rejects_malformed_numeric_controls() -> None:
    with pytest.raises(ValueError, match="k"):
        select_top_source_features([0.1, 0.2], k=True)

    with pytest.raises(ValueError, match="min_score"):
        select_top_source_features([0.1, 0.2], min_score=np.asarray([0.1]))


def test_direct_feature_selection_config_is_revalidated() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        fit_source_feature_selection(
            source_features=[[0.0], [1.0]],
            source_labels=[0, 1],
            test_features=[[0.5]],
            config=SourceFeatureSelectionConfig(epsilon=True),
        )


def test_source_feature_selection_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_feature_selection(
            source_features=[[0.0, 1.0], [1.0, 0.0]],
            source_labels=[0, 1],
            test_features=[[0.0]],
        )


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_feature_selection(
            source_features=[[0.0], [1.0]],
            source_labels=[0, 1],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )
