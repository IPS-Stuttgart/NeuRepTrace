from __future__ import annotations

import numpy as np

from neureptrace.decoding.precomputed_foundation import (
    align_precomputed_foundation_features,
    load_precomputed_foundation_features,
    make_precomputed_foundation_feature_table,
)


def test_make_table_preserves_composite_row_ids_for_alignment() -> None:
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    row_ids = [
        ("sub-01", "trial-0"),
        ("sub-01", "trial-1"),
        ("sub-02", "trial-0"),
    ]

    table = make_precomputed_foundation_feature_table(features, row_ids=row_ids)
    aligned = align_precomputed_foundation_features(
        table,
        np.asarray(
            [
                ["sub-02", "trial-0"],
                ["sub-01", "trial-0"],
            ]
        ),
    )

    assert table.row_ids == tuple(row_ids)
    assert np.allclose(aligned, np.asarray([[2.0, 2.0], [1.0, 0.0]], dtype=np.float32))


def test_align_accepts_bare_composite_row_id_for_single_lookup() -> None:
    table = make_precomputed_foundation_feature_table(
        [[1.0, 0.0], [0.0, 1.0]],
        row_ids=[("sub-01", "trial-0"), ("sub-01", "trial-1")],
    )

    aligned = align_precomputed_foundation_features(table, ("sub-01", "trial-1"))

    assert aligned.shape == (1, 2)
    assert np.allclose(aligned, np.asarray([[0.0, 1.0]], dtype=np.float32))


def test_npz_loader_preserves_matrix_encoded_composite_row_ids(tmp_path) -> None:
    path = tmp_path / "features_with_composite_row_ids.npz"
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    row_id_matrix = np.asarray(
        [
            ["sub-01", "trial-0"],
            ["sub-01", "trial-1"],
            ["sub-02", "trial-0"],
        ]
    )
    np.savez(path, features=features, row_ids=row_id_matrix)

    table = load_precomputed_foundation_features(path)
    aligned = align_precomputed_foundation_features(table, [("sub-01", "trial-1"), ("sub-02", "trial-0")])

    assert table.row_ids == (
        ("sub-01", "trial-0"),
        ("sub-01", "trial-1"),
        ("sub-02", "trial-0"),
    )
    assert np.allclose(aligned, np.asarray([[0.0, 1.0], [2.0, 2.0]], dtype=np.float32))
