from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.conditional_coral import fit_pseudo_label_conditional_coral


def _features() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ],
        dtype=float,
    )


def _labels() -> np.ndarray:
    return np.asarray([0, 0, 1, 1])


@pytest.mark.parametrize("feature_name", ["source_features", "target_features"])
def test_conditional_coral_rejects_complex_feature_inputs(feature_name: str) -> None:
    source = _features()
    target = _features()
    complex_features = source.astype(complex)
    complex_features[0, 0] += 1.0j
    if feature_name == "source_features":
        source = complex_features
    else:
        target = complex_features

    with pytest.raises(ValueError, match=rf"{feature_name} must contain real-valued feature values"):
        fit_pseudo_label_conditional_coral(
            source_features=source,
            source_labels=_labels(),
            target_features=target,
            target_pseudo_labels=_labels(),
        )


def test_conditional_coral_rejects_complex_target_probabilities() -> None:
    probabilities = np.asarray(
        [
            [0.9 + 0.1j, 0.1 - 0.1j],
            [0.8 + 0.1j, 0.2 - 0.1j],
            [0.2 + 0.1j, 0.8 - 0.1j],
            [0.1 + 0.1j, 0.9 - 0.1j],
        ],
        dtype=complex,
    )

    with pytest.raises(ValueError, match="target_probabilities must contain real-valued probability values"):
        fit_pseudo_label_conditional_coral(
            source_features=_features(),
            source_labels=_labels(),
            target_features=_features(),
            target_probabilities=probabilities,
        )
