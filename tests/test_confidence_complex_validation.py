from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.confidence import accepted_probability_rows, confidence_scores, select_confident_rows


def test_confidence_entry_points_reject_complex_probability_arrays() -> None:
    probabilities = np.asarray([[0.8 + 0.2j, 0.2 - 0.2j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="real-valued scores"):
        confidence_scores(probabilities)
    with pytest.raises(ValueError, match="real-valued scores"):
        select_confident_rows(probabilities)

    selection = select_confident_rows([[0.8, 0.2]])
    with pytest.raises(ValueError, match="real-valued scores"):
        accepted_probability_rows(probabilities, selection=selection)


def test_confidence_rejects_nested_and_one_pass_complex_scores() -> None:
    object_probabilities = np.asarray([[0.8, np.complex128(0.2 + 0.1j)]], dtype=object)
    generator_probabilities = (row for row in ([0.8, 0.2], [0.6, 0.4 + 0.1j]))

    with pytest.raises(ValueError, match="real-valued scores"):
        confidence_scores(object_probabilities)
    with pytest.raises(ValueError, match="real-valued scores"):
        confidence_scores(generator_probabilities)


def test_confidence_rejects_complex_numpy_thresholds() -> None:
    probabilities = [[0.8, 0.2]]

    for name, value in (
        ("min_confidence", np.complex128(0.5 + 0.1j)),
        ("min_margin", np.complex64(0.2 + 0.1j)),
        ("max_entropy", np.asarray(0.8 + 0.1j)),
    ):
        kwargs = {name: value}
        with pytest.raises(ValueError, match=name):
            select_confident_rows(probabilities, **kwargs)
