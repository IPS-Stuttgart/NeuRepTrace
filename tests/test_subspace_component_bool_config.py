from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.subspace_adaptation import fit_subspace_adaptation, subspace_adaptation_config


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        np.asarray(True),
        np.asarray([True]),
    ],
)
def test_subspace_config_rejects_boolean_component_counts(value: object) -> None:
    with pytest.raises(ValueError, match="n_components"):
        subspace_adaptation_config(n_components=value)  # type: ignore[arg-type]


def test_fit_subspace_adaptation_rejects_boolean_component_count() -> None:
    source = np.asarray([[-1.0, 0.0], [-0.5, 0.2], [0.5, -0.2], [1.0, 0.0]])
    target = source + np.asarray([2.0, 0.25])

    with pytest.raises(ValueError, match="n_components"):
        fit_subspace_adaptation(source, target, n_components=True)


def test_subspace_config_still_accepts_numpy_integer_component_count() -> None:
    config = subspace_adaptation_config(n_components=np.int64(1))

    assert config.n_components == 1
