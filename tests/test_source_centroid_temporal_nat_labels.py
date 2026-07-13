from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_centroid import fit_source_centroid_decoder


@pytest.mark.parametrize("temporal_nat", [np.datetime64("NaT"), np.timedelta64("NaT")])
def test_source_centroid_keeps_temporal_nat_separate_from_none(temporal_nat: object) -> None:
    source_features = np.asarray([[0.0], [0.2], [9.8], [10.0]], dtype=float)
    source_labels = np.empty(4, dtype=object)
    source_labels[:] = [None, None, temporal_nat, temporal_nat]

    result = fit_source_centroid_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=np.asarray([[0.1], [9.9]], dtype=float),
        config={"use_diagonal_scale": False},
    )

    assert result.classes.shape == (2,)
    assert result.classes[0] is None
    assert type(result.classes[1]) is type(temporal_nat)
    assert np.isnat(result.classes[1])
    assert result.predictions[0] is None
    assert type(result.predictions[1]) is type(temporal_nat)
    assert np.isnat(result.predictions[1])
    assert result.metadata["source_centroid_n_classes"] == 2
