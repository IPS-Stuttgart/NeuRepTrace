from __future__ import annotations

import numpy as np

from neureptrace.decoding.conditional_subspace import fit_jda


def _nested_label(side: str, run: int):
    return (side, (part for part in ("run", run)))


def test_fit_jda_materializes_nested_generator_backed_class_labels() -> None:
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ],
        dtype=float,
    )
    source_labels = [
        _nested_label("left", 0),
        _nested_label("left", 0),
        _nested_label("right", 1),
        _nested_label("right", 1),
    ]
    target_features = np.asarray(
        [
            [0.05, 0.0],
            [3.05, 3.0],
        ],
        dtype=float,
    )

    result = fit_jda(
        source_features,
        source_labels,
        target_features,
        n_components=1,
        max_iterations=3,
    )

    assert result.pseudo_labels.tolist() == [
        ("left", ("run", 0)),
        ("right", ("run", 1)),
    ]
    assert all(not hasattr(label[1], "__next__") for label in result.pseudo_labels)
