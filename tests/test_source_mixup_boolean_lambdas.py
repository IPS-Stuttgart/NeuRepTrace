from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mixup import mixup_rows


@pytest.mark.parametrize(
    ("lambdas", "rows", "partners"),
    [
        (True, [[0.0]], [[1.0]]),
        ([False], [[0.0]], [[1.0]]),
        (np.asarray([True]), [[0.0]], [[1.0]]),
        (np.asarray([0.25, np.bool_(True)], dtype=object), [[0.0], [1.0]], [[1.0], [2.0]]),
    ],
)
def test_mixup_rows_rejects_boolean_lambdas(lambdas, rows, partners) -> None:
    with pytest.raises(ValueError, match="lambdas"):
        mixup_rows(rows, partners, lambdas=lambdas)


def test_mixup_rows_accepts_numeric_endpoint_lambdas() -> None:
    mixed = mixup_rows([[0.0], [10.0]], [[2.0], [20.0]], lambdas=[0, 1])

    assert np.allclose(mixed, [[2.0], [10.0]])
