from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from neureptrace.decoding import source_weighting as _source_weighting

_PATCH_FLAG = "_neureptrace_feature_width_validation_patch_installed"
_ORIGINAL_ATTR = "_neureptrace_original_target_similarity_scores"


def _patched_target_similarity_scores(
    source_features: Mapping[Hashable, Sequence[Sequence[float]] | Sequence[float] | np.ndarray],
    target_features: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    *,
    groups: Sequence[Hashable] | None = None,
) -> dict[Hashable, float]:
    """Return centroid similarities after explicit feature-width validation."""

    group_list = _source_weighting._group_list(groups, source_features=source_features)  # noqa: SLF001
    target_centroid = _source_weighting._feature_centroid(target_features)  # noqa: SLF001
    target = _source_weighting._unit_centered_vector(target_centroid)  # noqa: SLF001
    scores: dict[Hashable, float] = {}
    for group in group_list:
        if group not in source_features:
            raise ValueError(f"Missing source features for group {group!r}.")
        source_centroid = _source_weighting._feature_centroid(source_features[group])  # noqa: SLF001
        if source_centroid.shape != target_centroid.shape:
            raise ValueError(
                "source and target feature centroids must have the same feature width: "
                f"group {group!r} has {source_centroid.size}, target has {target_centroid.size}."
            )
        source = _source_weighting._unit_centered_vector(source_centroid)  # noqa: SLF001
        scores[group] = float(np.clip(np.dot(source, target), -1.0, 1.0))
    return scores


def install() -> None:
    """Install explicit feature-width validation for target-similarity weighting."""

    if getattr(_source_weighting, _PATCH_FLAG, False):
        return
    setattr(_source_weighting, _ORIGINAL_ATTR, _source_weighting.target_similarity_scores)
    _source_weighting.target_similarity_scores = _patched_target_similarity_scores
    setattr(_source_weighting, _PATCH_FLAG, True)
