from __future__ import annotations

from neureptrace.decoding.source_whitening import apply_source_whitening, fit_source_whitening


def test_source_whitening_accepts_nested_one_pass_iterables() -> None:
    source_features = (iter(row) for row in [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    heldout_features = (iter(row) for row in [[6.0, 7.0], [8.0, 9.0]])

    result = fit_source_whitening(source_features=source_features, test_features=heldout_features, config={"mode": "diagonal", "regularization": 0.0})
    transformed = apply_source_whitening((iter(row) for row in [[10.0, 11.0]]), result.transform)

    assert result.train_features.shape == (3, 2)
    assert result.test_features.shape == (2, 2)
    assert transformed.shape == (1, 2)
