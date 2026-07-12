"""Keep source-domain generalization row-validation splits valid for small folds."""

from __future__ import annotations

import importlib

import numpy as np

_PATCH_MARKER = "_neureptrace_source_domain_generalization_validation_split_patch_installed"


def _stratified_row_fallback_split(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stratified row split that keeps at least one row per class on both sides."""

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    _classes, encoded = np.unique(labels, return_inverse=True)
    class_counts = np.bincount(encoded)
    n_classes = int(class_counts.shape[0])
    max_validation_rows = int(np.sum(class_counts - 1))
    desired_validation_rows = int(np.ceil(labels.shape[0] * float(validation_fraction)))
    n_validation_rows = min(max(n_classes, desired_validation_rows), max_validation_rows)

    validation_counts = np.ones(n_classes, dtype=np.int64)
    remaining = int(n_validation_rows - n_classes)
    capacities = class_counts.astype(np.int64, copy=True) - 2
    while remaining > 0 and np.any(capacities > 0):
        candidates = np.flatnonzero(capacities > 0)
        deficits = class_counts[candidates] * float(validation_fraction) - validation_counts[candidates]
        best_deficit = float(np.max(deficits))
        tied = candidates[np.flatnonzero(deficits == best_deficit)]
        chosen = int(tied[np.argmax(capacities[tied])])
        validation_counts[chosen] += 1
        capacities[chosen] -= 1
        remaining -= 1

    rng = np.random.default_rng(random_state)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for code in range(n_classes):
        class_indices = indices[encoded == code]
        permuted = rng.permutation(class_indices)
        n_class_validation = int(validation_counts[code])
        validation_parts.append(permuted[:n_class_validation])
        train_parts.append(permuted[n_class_validation:])

    train_idx = rng.permutation(np.concatenate(train_parts)).astype(np.int64, copy=False)
    validation_idx = rng.permutation(np.concatenate(validation_parts)).astype(np.int64, copy=False)
    return train_idx, validation_idx


def install() -> None:
    """Patch the source-domain generalization validation splitter."""

    module = importlib.import_module("neureptrace.decoding.source_domain_generalization")
    if getattr(module, _PATCH_MARKER, False):
        return

    def _source_domain_validation_split(labels: np.ndarray, domains: np.ndarray, *, validation_fraction: float, random_state: int | None):
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        domains_array = np.asarray(domains, dtype=np.int64).reshape(-1)
        indices = np.arange(labels_array.shape[0])
        unique_domains = np.unique(domains_array)
        n_classes = int(np.unique(labels_array).shape[0])
        fraction = float(validation_fraction)
        rng = np.random.default_rng(random_state)

        if 0.0 < fraction < 1.0 and unique_domains.shape[0] >= 2:
            n_valid_domains = max(1, int(round(unique_domains.shape[0] * fraction)))
            n_valid_domains = min(n_valid_domains, unique_domains.shape[0] - 1)
            subsets = [(domain,) for domain in rng.permutation(unique_domains).tolist()]
            if n_valid_domains > 1:
                for _ in range(min(32, 4 * unique_domains.shape[0])):
                    subset = tuple(sorted(rng.choice(unique_domains, size=n_valid_domains, replace=False).tolist()))
                    if subset not in subsets:
                        subsets.append(subset)
            for valid_domains in subsets:
                valid_mask = np.isin(domains_array, valid_domains)
                train_idx = indices[~valid_mask]
                valid_idx = indices[valid_mask]
                if train_idx.size and valid_idx.size:
                    train_has_all_classes = np.unique(labels_array[train_idx]).shape[0] == n_classes
                    valid_has_two_classes = np.unique(labels_array[valid_idx]).shape[0] >= 2
                    if train_has_all_classes and valid_has_two_classes:
                        return train_idx, valid_idx, "heldout_source_domain"

        class_counts = np.bincount(labels_array, minlength=n_classes)
        can_row_validate = 0.0 < fraction < 1.0 and labels_array.shape[0] >= 2 * n_classes and np.min(class_counts) >= 2
        if can_row_validate:
            train_idx, valid_idx = _stratified_row_fallback_split(
                indices,
                labels_array,
                validation_fraction=fraction,
                random_state=random_state,
            )
            return train_idx, valid_idx, "stratified_row_fallback"
        return indices, indices, "training_loss_fallback"

    module._source_domain_validation_split = _source_domain_validation_split
    setattr(module, _PATCH_MARKER, True)


__all__ = ["install"]
