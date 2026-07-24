import numpy as np

from neureptrace.decoding.source_domain_generalization import _source_domain_validation_split


def test_source_domain_generalization_row_fallback_handles_small_fraction_multiclass():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    domains = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)

    train_idx, valid_idx, mode = _source_domain_validation_split(
        labels,
        domains,
        validation_fraction=0.1,
        random_state=3,
    )

    assert mode == "stratified_row_fallback"
    assert train_idx.shape == (3,)
    assert valid_idx.shape == (3,)
    assert set(labels[train_idx].tolist()) == {0, 1, 2}
    assert set(labels[valid_idx].tolist()) == {0, 1, 2}
    assert not set(train_idx.tolist()) & set(valid_idx.tolist())


def test_source_domain_generalization_holds_out_requested_domain_fraction():
    domains = np.repeat(np.arange(6, dtype=np.int64), 2)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 6)

    train_idx, valid_idx, mode = _source_domain_validation_split(
        labels,
        domains,
        validation_fraction=0.5,
        random_state=7,
    )

    assert mode == "heldout_source_domain"
    assert np.unique(domains[valid_idx]).shape[0] == 3
    assert np.unique(domains[train_idx]).shape[0] == 3
    assert set(labels[train_idx].tolist()) == {0, 1}
    assert set(labels[valid_idx].tolist()) == {0, 1}
    assert not set(train_idx.tolist()) & set(valid_idx.tolist())
