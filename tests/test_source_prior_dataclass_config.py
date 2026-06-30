from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_prior import SourcePriorConfig, adjust_probabilities_to_source_prior


def test_source_prior_accepts_direct_dataclass_aliases() -> None:
    probabilities = np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=float)
    cfg = SourcePriorConfig(target_prior="source-prior", smoothing="1.0", epsilon="1e-6")

    result = adjust_probabilities_to_source_prior(
        probabilities,
        source_labels=[0, 0, 1],
        classes=[0, 1],
        config=cfg,
    )

    assert np.allclose(result.probabilities, probabilities)
    assert result.metadata["source_prior_target_prior"] == "source"
