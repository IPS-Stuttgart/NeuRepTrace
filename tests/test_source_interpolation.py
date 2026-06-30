from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_interpolation import (
    SOURCE_INTERPOLATION_CATEGORY,
    augment_source_with_interpolation,
    interpolate_rows,
    normalize_pair_mode,
    source_interpolation_config,
)


def test_source_interpolation_appends_same_class_rows() -> None:
    features = np.asarray([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)

    result = augment_source_with_interpolation(features, labels, config={"synthetic_per_class": 2, "random_state": 7})

    assert result.features.shape == (8, 2)
    assert result.synthetic_mask.tolist() == [False, False, False, False, True, True, True, True]
    assert result.n_synthetic == 4
    assert result.metadata["source_interpolation_protocol_category"] == SOURCE_INTERPOLATION_CATEGORY
    assert result.metadata["source_interpolation_uses_heldout_features"] is False
    assert result.metadata["source_interpolation_uses_heldout_labels"] is False
    assert np.all(result.labels[result.synthetic_mask] == labels[result.content_indices])
    assert np.all(result.labels[result.content_indices] == labels[result.partner_indices])


def test_cross_domain_mode_prefers_other_domain() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    labels = np.asarray(["a", "a", "a", "a"], dtype=object)
    domains = np.asarray(["s1", "s1", "s2", "s2"], dtype=object)

    result = augment_source_with_interpolation(
        features,
        labels,
        source_domains=domains,
        config={"synthetic_per_class": 4, "pair_mode": "same_class_cross_domain", "random_state": 3},
    )

    assert np.all(domains[result.content_indices] != domains[result.partner_indices])


def test_interpolate_rows_and_config_validation() -> None:
    assert np.allclose(interpolate_rows([0.0, 10.0], [10.0, 0.0], 0.25), np.asarray([7.5, 2.5]))
    assert normalize_pair_mode("within-class") == "same_class"
    assert normalize_pair_mode("cross-domain") == "same_class_cross_domain"
    assert source_interpolation_config(synthetic_per_class="2").synthetic_per_class == 2

    with pytest.raises(ValueError, match="pair mode"):
        normalize_pair_mode("bad")
    with pytest.raises(ValueError, match="alpha"):
        source_interpolation_config(alpha=0.0)
    with pytest.raises(ValueError, match="lam"):
        interpolate_rows([0.0], [1.0], 1.5)


def test_source_interpolation_accepts_null_random_state_strings() -> None:
    cfg = source_interpolation_config(random_state="NULL")
    assert cfg.random_state is None

    features = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float)
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    result = augment_source_with_interpolation(features, labels, config={"synthetic_per_class": 1, "random_state": "null"})

    assert result.n_synthetic == 2
    assert result.metadata["source_interpolation_random_state"] == ""


@pytest.mark.parametrize("bad_seed", [True, np.asarray([7])])
def test_source_interpolation_rejects_invalid_random_state_values(bad_seed: object) -> None:
    with pytest.raises(ValueError, match="random_state"):
        source_interpolation_config(random_state=bad_seed)  # type: ignore[arg-type]


def test_disabled_interpolation_returns_original_rows() -> None:
    features = np.asarray([[0.0], [1.0]])
    labels = np.asarray([0, 1], dtype=object)

    result = augment_source_with_interpolation(features, labels)

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == [0, 1]
    assert result.n_synthetic == 0
