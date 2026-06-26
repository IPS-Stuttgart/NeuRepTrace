from __future__ import annotations

import numpy as np
import pytest

from neureptrace.bushmeg_source_loso_ensemble import (
    _fit_stacking_weights,
    _normalize_rerank_top_k,
    _normalize_temperature,
    _parse_float_grid,
    run_bushmeg_source_loso_ensemble,
)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False), 2.5, np.float64(1.25)])
def test_rerank_top_k_rejects_boolean_and_fractional_values(value) -> None:
    with pytest.raises(ValueError, match="source_loso.rerank_top_k must be an integer"):
        _normalize_rerank_top_k(value)


def test_rerank_top_k_preserves_disable_aliases_and_integral_values() -> None:
    assert _normalize_rerank_top_k(None) == 0
    assert _normalize_rerank_top_k("off") == 0
    assert _normalize_rerank_top_k("false") == 0
    assert _normalize_rerank_top_k(2.0) == 2
    assert _normalize_rerank_top_k(np.int64(3)) == 3


def test_softmax_temperature_rejects_boolean_values() -> None:
    with pytest.raises(ValueError, match="source_loso.ensemble_temperature must be a finite floating-point value"):
        _normalize_temperature(True, "softmax")


def test_rerank_alpha_grid_rejects_boolean_values() -> None:
    with pytest.raises(ValueError, match="Reranker alpha grid must contain numeric values"):
        _parse_float_grid([0.0, np.bool_(True)], (0.0, 0.25))


def test_stacking_weights_reject_fractional_or_boolean_iteration_counts() -> None:
    probability_cube = np.asarray(
        [
            [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]],
            [[0.7, 0.3], [0.4, 0.6], [0.55, 0.45], [0.25, 0.75]],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=int)

    for bad_max_iter in (True, 1.5):
        with pytest.raises(ValueError, match="Stacking max_iter must be an integer"):
            _fit_stacking_weights(probability_cube, labels, n_classes=2, max_iter=bad_max_iter)

    weights = _fit_stacking_weights(probability_cube, labels, n_classes=2, max_iter=np.float64(2.0))
    assert weights.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("source_loso:\n  ensemble_top_k: true\n", "source_loso.ensemble_top_k must be an integer"),
        ("source_loso:\n  ensemble_top_k: 1.5\n", "source_loso.ensemble_top_k must be an integer"),
        ("source_loso:\n  rerank_top_k: 2.5\n", "source_loso.rerank_top_k must be an integer"),
        ("source_loso:\n  ensemble_rerank_top_k: true\n", "source_loso.rerank_top_k must be an integer"),
        ("source_loso:\n  ensemble_temperature: true\n", "source_loso.ensemble_temperature must be a finite floating-point value"),
        ("source_loso:\n  rerank_alpha_grid: [0.0, true]\n", "Reranker alpha grid must contain numeric values"),
        ("decoding:\n  max_iter: true\n", "decoding.max_iter must be an integer"),
        ("decoding:\n  max_iter: 2.5\n", "decoding.max_iter must be an integer"),
    ],
)
def test_source_loso_ensemble_runner_rejects_ambiguous_numeric_config(tmp_path, yaml_text: str, message: str) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_bushmeg_source_loso_ensemble(config_path)
