from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mixup import augment_source_with_mixup, mixup_rows


@pytest.mark.parametrize("complex_argument", ["content", "partner"])
def test_mixup_rows_rejects_complex_feature_inputs(complex_argument: str) -> None:
    content = np.asarray([[1.0, 2.0]])
    partner = np.asarray([[3.0, 4.0]])
    if complex_argument == "content":
        content = content.astype(complex) + 1.0j
    else:
        partner = partner.astype(complex) + 1.0j

    with pytest.raises(ValueError, match="real-valued feature values"):
        mixup_rows(content, partner, lambdas=[0.5])


def test_mixup_rows_rejects_complex_lambda_inputs() -> None:
    with pytest.raises(ValueError, match="lambdas must contain real-valued values"):
        mixup_rows(
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            lambdas=np.asarray([0.5 + 0.25j]),
        )


def test_disabled_source_mixup_rejects_complex_source_features() -> None:
    with pytest.raises(ValueError, match="source_features must contain real-valued feature values"):
        augment_source_with_mixup(
            np.asarray([[1.0 + 1.0j, 2.0], [3.0, 4.0]]),
            ["left", "right"],
            config={"synthetic_per_class": 0},
        )


def test_mixup_rows_preserves_real_generator_lambdas() -> None:
    result = mixup_rows(
        [[1.0, 2.0]],
        [[3.0, 4.0]],
        lambdas=(value for value in [0.25]),
    )

    np.testing.assert_allclose(result, [[2.5, 3.5]])
