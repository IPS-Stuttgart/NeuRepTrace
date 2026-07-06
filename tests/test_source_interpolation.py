from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_interpolation import (
    SOURCE_INTERPOLATION_CATEGORY,
    SourceInterpolationConfig,
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


def test_source_interpolation_accepts_one_pass_feature_label_and_domain_iterables() -> None:
    feature_rows = [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]]
    labels = ["a", "a", "b", "b"]
    domains = ["s1", "s2", "s1", "s2"]

    result = augment_source_with_interpolation(
        (row for row in feature_rows),
        (label for label in labels),
        source_domains=(domain for domain in domains),
        config={"synthetic_per_class": 1, "pair_mode": "same_class_cross_domain", "random_state": 5},
    )

    assert result.features.shape == (6, 2)
    assert result.labels.tolist()[:4] == labels
    assert result.labels.tolist()[4:] == ["a", "b"]
    assert result.n_synthetic == 2
    assert np.all(np.asarray(domains, dtype=object)[result.content_indices] != np.asarray(domains, dtype=object)[result.partner_indices])


def test_source_interpolation_config_normalizes_direct_dataclass_values() -> None:
    cfg = SourceInterpolationConfig(
        synthetic_per_class="2",  # type: ignore[arg-type]
        pair_mode="cross-domain",
        alpha=np.asarray("0.5"),  # type: ignore[arg-type]
        preserve_original="false",  # type: ignore[arg-type]
        random_state=np.asarray("none"),  # type: ignore[arg-type]
    )

    assert cfg.synthetic_per_class == 2
    assert isinstance(cfg.synthetic_per_class, int)
    assert cfg.pair_mode == "same_class_cross_domain"
    assert cfg.alpha == pytest.approx(0.5)
    assert cfg.preserve_original is False
    assert cfg.random_state is None
    assert cfg.enabled is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"synthetic_per_class": True}, "synthetic_per_class must be a non-negative integer"),
        ({"synthetic_per_class": np.asarray([1])}, "synthetic_per_class must be a non-negative integer"),
        ({"alpha": False}, "alpha must be positive and finite"),
        ({"preserve_original": np.asarray([True])}, "preserve_original must be a boolean value"),
        ({"random_state": True}, "random_state must be a non-negative integer"),
        ({"random_state": np.asarray([7])}, "random_state must be a non-negative integer"),
    ],
)
def test_source_interpolation_config_rejects_invalid_direct_dataclass_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SourceInterpolationConfig(**kwargs)  # type: ignore[arg-type]


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


@pytest.mark.parametrize("bad_value", [True, np.bool_(False), np.asarray([1])])
def test_source_interpolation_rejects_invalid_synthetic_counts(bad_value: object) -> None:
    with pytest.raises(ValueError, match="synthetic_per_class"):
        source_interpolation_config(synthetic_per_class=bad_value)  # type: ignore[arg-type]


def test_disabled_interpolation_returns_original_rows() -> None:
    features = np.asarray([[0.0], [1.0]])
    labels = np.asarray([0, 1], dtype=object)

    result = augment_source_with_interpolation(features, labels)

    assert np.allclose(result.features, features)
    assert result.labels.tolist() == [0, 1]
    assert result.n_synthetic == 0
