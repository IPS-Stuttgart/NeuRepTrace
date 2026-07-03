from __future__ import annotations

from neureptrace.decoding.source_feature_selection import fit_source_feature_selection, source_feature_scores


def test_source_feature_selection_accepts_one_pass_iterables() -> None:
    source_rows = ([value, 1.0] for value in [0.0, 0.2, 5.0, 5.2])
    labels = (label for label in ["a", "a", "b", "b"])
    test_rows = ([value, 1.0] for value in [0.1, 5.1])

    result = fit_source_feature_selection(
        source_features=source_rows,
        source_labels=labels,
        test_features=test_rows,
        config={"method": "anova", "k": 1},
    )

    assert result.selected_indices.tolist() == [0]
    assert result.train_features.shape == (4, 1)
    assert result.test_features.shape == (2, 1)


def test_source_feature_scores_accepts_one_pass_label_iterable() -> None:
    scores = source_feature_scores(
        [[0.0, 1.0], [0.2, 1.0], [5.0, 1.0], [5.2, 1.0]],
        (label for label in ["a", "a", "b", "b"]),
        method="anova",
    )

    assert scores[0] > scores[1]
