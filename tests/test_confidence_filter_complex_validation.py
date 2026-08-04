from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.confidence_filter import confidence_filter, probability_entropy


@pytest.mark.parametrize(
    "operation",
    [confidence_filter, probability_entropy],
)
@pytest.mark.parametrize(
    "probabilities",
    [
        np.asarray([[0.8 + 0.1j, 0.2 - 0.1j]], dtype=np.complex128),
        np.asarray(
            [[np.complex128(0.8 + 0.1j), np.complex128(0.2 - 0.1j)]],
            dtype=object,
        ),
    ],
)
def test_confidence_helpers_reject_complex_probabilities(
    operation: Callable[..., object],
    probabilities: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="real-valued probability values, not complex values",
    ):
        operation(probabilities)


def _one_pass_complex_probability_rows():
    return (iter(row) for row in ((0.8 + 0.1j, 0.2 - 0.1j),))


@pytest.mark.parametrize(
    "operation",
    [confidence_filter, probability_entropy],
)
def test_confidence_helpers_reject_one_pass_complex_probabilities(
    operation: Callable[..., object],
) -> None:
    with pytest.raises(
        ValueError,
        match="real-valued probability values, not complex values",
    ):
        operation(_one_pass_complex_probability_rows())


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("min_confidence", np.complex128(0.5 + 1.0j)),
        ("min_margin", np.complex128(0.1 + 1.0j)),
        ("max_entropy", np.complex128(0.5 + 1.0j)),
        ("min_confidence", np.asarray(0.5 + 1.0j)),
        ("max_entropy", np.asarray(0.5 + 1.0j)),
    ],
)
def test_confidence_filter_rejects_complex_thresholds(
    keyword: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=keyword):
        confidence_filter([[0.8, 0.2]], **{keyword: value})
