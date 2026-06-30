from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_mixstyle import SourceMixStyleConfig, augment_source_domains_mixstyle


def test_source_mixstyle_dataclass_string_false_excludes_original_rows() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [10.0, 10.0],
            [11.0, 10.5],
            [10.5, 11.0],
        ],
        dtype=float,
    )
    labels = np.asarray(["a", "b", "a", "a", "b", "a"], dtype=object)
    domains = np.asarray(["s1", "s1", "s1", "s2", "s2", "s2"], dtype=object)
    config = SourceMixStyleConfig(mixes_per_row=1, include_original="false", random_state=5)  # type: ignore[arg-type]

    result = augment_source_domains_mixstyle(features, labels, domains, config=config)

    assert result.features.shape == features.shape
    assert result.n_original == 0
    assert result.n_synthetic == features.shape[0]
    assert np.all(result.synthetic_mask)
    assert result.metadata["source_mixstyle_include_original"] is False
