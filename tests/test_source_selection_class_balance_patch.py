from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_selection import select_source_domains_by_target_similarity


def _class_balance_fixture():
    source_features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.0, 0.2],
            [6.0, 6.0],
            [6.1, 6.0],
        ],
        dtype=float,
    )
    source_domains = np.asarray(["near", "near", "near", "near", "near", "far", "far"], dtype=object)
    source_labels = np.asarray(["major", "major", "major", "major", "minor", "major", "minor"], dtype=object)
    target_features = np.asarray([[0.0, 0.0], [0.2, 0.1]], dtype=float)
    return source_features, source_domains, source_labels, target_features


@pytest.mark.parametrize("value", ["false", "False", "0", "off", "no", 0, False, np.bool_(False)])
def test_source_selection_class_balance_false_aliases_do_not_require_source_labels(value) -> None:
    source_features, source_domains, _source_labels, target_features = _class_balance_fixture()

    result = select_source_domains_by_target_similarity(
        source_features,
        source_domains,
        target_features,
        metric="mean",
        top_k=1,
        class_balance=value,
    )

    assert result.metadata["source_selection_uses_source_labels"] is False
    assert result.metadata["source_selection_class_balance"] is False


@pytest.mark.parametrize("value", ["true", "True", "1", "on", "yes", 1, True, np.bool_(True)])
def test_source_selection_class_balance_true_aliases_enable_balancing(value) -> None:
    source_features, source_domains, source_labels, target_features = _class_balance_fixture()

    result = select_source_domains_by_target_similarity(
        source_features,
        source_domains,
        target_features,
        metric="mean",
        top_k=1,
        source_labels=source_labels,
        class_balance=value,
    )

    selected_major_mass = float(np.sum(result.sample_weights[(source_labels == "major") & result.selected_mask]))
    selected_minor_mass = float(np.sum(result.sample_weights[(source_labels == "minor") & result.selected_mask]))
    assert result.metadata["source_selection_uses_source_labels"] is True
    assert result.metadata["source_selection_class_balance"] is True
    assert np.isclose(selected_major_mass, selected_minor_mass)


@pytest.mark.parametrize("value", ["", "maybe", 2, -1])
def test_source_selection_class_balance_rejects_ambiguous_values(value) -> None:
    source_features, source_domains, _source_labels, target_features = _class_balance_fixture()

    with pytest.raises(ValueError, match="class_balance must be a boolean value"):
        select_source_domains_by_target_similarity(
            source_features,
            source_domains,
            target_features,
            metric="mean",
            top_k=1,
            class_balance=value,
        )
