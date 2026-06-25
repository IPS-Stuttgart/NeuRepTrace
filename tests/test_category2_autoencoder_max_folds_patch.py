from __future__ import annotations

from neureptrace._category2_autoencoder_max_folds_patch import _FoldLimitedSubjectMap


def test_fold_limited_subject_map_preserves_sources_after_outer_fold_selection() -> None:
    subject_map = _FoldLimitedSubjectMap({"s01": 1, "s02": 2, "s03": 3}, max_folds=1)

    assert dict(subject_map) == {"s01": 1, "s02": 2, "s03": 3}
    assert sorted(subject_map) == ["s01"]
    assert sorted(subject_map) == ["s01", "s02", "s03"]
