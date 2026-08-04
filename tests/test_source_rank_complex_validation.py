from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_rank import (
    SourceRankReference,
    fit_source_rank_reference,
    fit_source_rank_transform,
    transform_source_rank_features,
)


def test_source_rank_rejects_complex_source_features() -> None:
    source = np.asarray([[0.0 + 1.0j], [1.0 + 0.0j]], dtype=np.complex128)

    with pytest.raises(
        ValueError,
        match="source_features must contain real-valued numeric values",
    ):
        fit_source_rank_reference(source)


def test_source_rank_rejects_complex_eval_features() -> None:
    eval_features = np.asarray([[0.5 + 1.0j]], dtype=np.complex128)

    with pytest.raises(
        ValueError,
        match="eval_features must contain real-valued numeric values",
    ):
        fit_source_rank_transform(
            source_features=[[0.0], [1.0]],
            eval_features=eval_features,
        )


def test_source_rank_rejects_complex_direct_reference_values() -> None:
    reference = SourceRankReference(
        sorted_values=np.asarray([[0.0 + 1.0j], [1.0 + 0.0j]], dtype=np.complex128)
    )

    with pytest.raises(
        ValueError,
        match="sorted_values must contain real-valued numeric values",
    ):
        transform_source_rank_features([[0.5]], reference)
