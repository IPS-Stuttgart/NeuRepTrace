from __future__ import annotations

import numpy as np

from neureptrace.decoding.transfer_components import TRANSFER_COMPONENT_CATEGORY, fit_transfer_component_features


def test_linear_tca_returns_category2_latent_features() -> None:
    source = np.asarray([[-2.0, 0.0], [-1.5, 0.2], [2.0, 0.0], [1.7, 0.2]])
    target = source + np.asarray([0.7, 1.5])

    result = fit_transfer_component_features(source_features=source, target_features=target, config={"n_components": 1})

    assert result.source_features.shape == (4, 1)
    assert result.target_features.shape == (4, 1)
    assert result.metadata["transfer_component_protocol_category"] == TRANSFER_COMPONENT_CATEGORY
    assert result.metadata["transfer_component_uses_target_features"] is True
    assert result.metadata["transfer_component_uses_target_labels"] is False
