from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_temperature import negative_log_likelihood


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0.0 + 0.0j], dtype=np.complex64),
        np.asarray([0.0 + 0.5j], dtype=np.complex128),
        np.asarray([np.complex128(0.0 + 0.5j)], dtype=object),
    ],
)
def test_negative_log_likelihood_rejects_complex_class_indices(labels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="labels must contain finite integer class indices"):
        negative_log_likelihood([[0.8, 0.2]], labels)


def test_negative_log_likelihood_rejects_complex_one_pass_labels() -> None:
    labels = (label for label in [np.complex128(0.0 + 0.5j)])

    with pytest.raises(ValueError, match="labels must contain finite integer class indices"):
        negative_log_likelihood([[0.8, 0.2]], labels)
