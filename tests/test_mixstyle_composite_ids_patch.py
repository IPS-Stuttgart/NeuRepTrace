from __future__ import annotations

import numpy as np


def test_package_import_preserves_feature_mixstyle_composite_labels() -> None:
    import neureptrace
    from neureptrace.decoding.mixstyle import augment_source_mixstyle

    assert neureptrace.__version__

    result = augment_source_mixstyle(
        np.array(
            [
                [0.0, 1.0],
                [0.5, 1.5],
                [2.0, 3.0],
                [2.5, 3.5],
            ],
            dtype=float,
        ),
        source_labels=[("class_a", "view_a"), ("class_a", "view_a"), ("class_b", "view_b"), ("class_b", "view_b")],
        source_domains=[("domain", 1), ("domain", 2), ("domain", 1), ("domain", 2)],
        augmentations_per_row=0,
        include_original=True,
    )

    assert result.labels.dtype == object
    assert result.domains.dtype == object
    assert result.labels.tolist() == [("class_a", "view_a"), ("class_a", "view_a"), ("class_b", "view_b"), ("class_b", "view_b")]
    assert result.domains.tolist() == [("domain", 1), ("domain", 2), ("domain", 1), ("domain", 2)]
