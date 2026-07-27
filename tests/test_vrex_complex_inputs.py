from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.vrex import LinearVRExClassifier


def _source_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [10.0, 2.0],
                [11.0, 2.5],
                [12.0, 3.0],
                [13.0, 3.5],
            ]
        ),
        np.asarray(["left", "right", "left", "right"], dtype=object),
        np.asarray(["s1", "s1", "s2", "s2"], dtype=object),
    )


@pytest.mark.parametrize(
    "feature_factory",
    [
        lambda: np.asarray(
            [[10.0 + 0.0j, 2.0], [11.0, 2.5], [12.0, 3.0 + 1.0j], [13.0, 3.5]],
            dtype=np.complex128,
        ),
        lambda: np.asarray(
            [[10.0, 2.0], [11.0, 2.5], [12.0, 3.0 + 1.0j], [13.0, 3.5]],
            dtype=object,
        ),
        lambda: (iter(row) for row in ([10.0, 2.0], [11.0, 2.5], [12.0, 3.0 + 1.0j], [13.0, 3.5])),
    ],
    ids=["complex-array", "object-array", "nested-generator"],
)
def test_vrex_rejects_complex_fit_features(feature_factory: Callable[[], object]) -> None:
    _, labels, domains = _source_table()
    model = LinearVRExClassifier(max_iter=3, tol=1e-4)

    with pytest.raises(ValueError, match="source_features.*complex"):
        model.fit(feature_factory(), labels, source_domains=domains)


@pytest.mark.parametrize(
    "feature_factory",
    [
        lambda: np.asarray([[10.5 + 1.0j, 2.2]], dtype=np.complex128),
        lambda: (iter(row) for row in ([10.5, 2.2 + 1.0j],)),
    ],
    ids=["complex-array", "nested-generator"],
)
def test_vrex_rejects_complex_prediction_features(feature_factory: Callable[[], object]) -> None:
    features, labels, domains = _source_table()
    model = LinearVRExClassifier(max_iter=3, tol=1e-4)
    model.fit(features, labels, source_domains=domains)

    with pytest.raises(ValueError, match="features.*complex"):
        model.predict_proba(feature_factory())


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("max_iter", np.complex128(3 + 1j), "max_iter must be a positive integer"),
        ("max_iter", np.asarray(3 + 1j), "max_iter must be a positive integer"),
        ("tol", np.complex128(1e-4 + 1j), "tol must be positive and finite"),
        ("penalty_weight", np.complex128(1.0 + 1j), "penalty_weight must be finite and non-negative"),
        ("l2", np.asarray([1e-4 + 1j]), "l2 must be finite and non-negative"),
    ],
)
def test_vrex_rejects_numpy_complex_numeric_hyperparameters(parameter: str, value: object, message: str) -> None:
    features, labels, domains = _source_table()
    model = LinearVRExClassifier(**{parameter: value})

    with pytest.raises(ValueError, match=message):
        model.fit(features, labels, source_domains=domains)
