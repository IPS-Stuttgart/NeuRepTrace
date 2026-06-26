from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_vrex import TorchVRExClassifier, _domain_balanced_batch


def _data():
    rows, labels, domains = [], [], []
    for domain_index, offset in enumerate((-0.4, 0.0, 0.5)):
        for repeat in range(4):
            rows.extend(([-1.0 + offset, 0.1 * repeat], [1.0 + offset, -0.1 * repeat]))
            labels.extend(("left", "right"))
            domains.extend((f"s{domain_index}", f"s{domain_index}"))
    return np.asarray(rows), np.asarray(labels), np.asarray(domains)


def test_vrex_probabilities_and_metadata() -> None:
    pytest.importorskip("torch")
    features, labels, domains = _data()
    model = TorchVRExClassifier(
        hidden_units=8,
        embedding_dim=4,
        max_epochs=3,
        batch_size=12,
        penalty_weight=0.5,
        penalty_anneal_epochs=0,
        validation_fraction=0.34,
        patience=2,
        random_state=3,
        device="cpu",
    ).fit(features, labels, source_domains=domains)
    probabilities = model.predict_proba([[-0.8, 0.0], [0.9, 0.0]])
    metadata = model.metadata(test_rows=2)
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert metadata["source_vrex_uses_target_features"] is False
    assert metadata["source_vrex_uses_target_labels"] is False
    assert metadata["source_vrex_valid_for_strict_source_only"] is True
    assert metadata["source_vrex_source_domains"] == 3


def test_balanced_batch_contains_all_domains() -> None:
    indices = np.arange(12)
    domains = np.asarray([0] * 4 + [1] * 4 + [2] * 4)
    batch = _domain_balanced_batch(indices, domains, batch_size=9, rng=np.random.default_rng(7))
    assert batch.shape == (9,)
    assert set(domains[batch].tolist()) == {0, 1, 2}


def test_vrex_requires_multiple_domains() -> None:
    pytest.importorskip("torch")
    features, labels, domains = _data()
    with pytest.raises(ValueError, match="at least two source domains"):
        TorchVRExClassifier(max_epochs=1, patience=1, device="cpu").fit(
            features[:8], labels[:8], source_domains=domains[:8]
        )
