"""Exact inference and training losses for small permutation-valued trials."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np

from neureptrace.decoding._progressive_sequence_core import _positive_float, _torch


@dataclass(frozen=True, slots=True)
class ExactPermutationDecodingResult:
    """Exact posterior marginals and MAP assignments over class permutations."""

    marginals: np.ndarray
    map_assignments: np.ndarray
    log_partition: np.ndarray
    permutation_probabilities: np.ndarray


@lru_cache(maxsize=None)
def permutation_indices(n_classes: int) -> np.ndarray:
    """Return every permutation of ``range(n_classes)`` in lexicographic order."""

    count = int(n_classes)
    if count < 2:
        raise ValueError("Permutation inference requires at least two classes.")
    result = np.asarray(tuple(permutations(range(count))), dtype=np.int64)
    result.setflags(write=False)
    return result


def _square_probability_tensor(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError(
            "probabilities must have shape (trials, events, classes)."
        )
    if values.shape[1] != values.shape[2] or values.shape[1] < 2:
        raise ValueError(
            "Exact permutation inference requires the same number of events and classes."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    row_mass = values.sum(axis=2, keepdims=True)
    if np.any(row_mass <= 0.0):
        raise ValueError("Every event row must contain positive probability mass.")
    return values / row_mass


def exact_permutation_decode(
    probabilities: np.ndarray,
    *,
    temperature: float = 1.0,
) -> ExactPermutationDecodingResult:
    """Enumerate the exact permutation posterior for each trial.

    The input may contain independently normalized event probabilities. Row-wise
    normalizing constants cancel from every permutation score, so the resulting
    posterior is identical to one computed from the corresponding logits.
    """

    values = _square_probability_tensor(probabilities)
    inverse_temperature = 1.0 / _positive_float(temperature, "temperature")
    n_trials, n_events, _n_classes = values.shape
    candidates = permutation_indices(n_events)
    n_permutations = candidates.shape[0]
    log_values = np.log(np.clip(values, np.finfo(np.float64).tiny, 1.0))
    expanded = np.broadcast_to(
        log_values[:, None, :, :],
        (n_trials, n_permutations, n_events, n_events),
    )
    selected = np.take_along_axis(
        expanded,
        candidates[None, :, :, None],
        axis=3,
    ).squeeze(3)
    scores = inverse_temperature * selected.sum(axis=2)
    score_max = np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores - score_max)
    normalizer = weights.sum(axis=1, keepdims=True)
    posterior = weights / normalizer
    log_partition = score_max[:, 0] + np.log(normalizer[:, 0])

    marginals = np.zeros_like(values)
    for permutation_index, candidate in enumerate(candidates):
        weight = posterior[:, permutation_index]
        for event_index, class_index in enumerate(candidate.tolist()):
            marginals[:, event_index, class_index] += weight
    map_assignments = candidates[np.argmax(scores, axis=1)].copy()
    return ExactPermutationDecodingResult(
        marginals=marginals,
        map_assignments=map_assignments,
        log_partition=log_partition,
        permutation_probabilities=posterior,
    )


def torch_exact_permutation_nll(logits, labels, *, temperature: float = 1.0):
    """Return exact mean negative log-likelihood over all class permutations."""

    torch = _torch()
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must have shapes (trials, events, classes) and "
            "(trials, events)."
        )
    n_events = int(logits.shape[1])
    n_classes = int(logits.shape[2])
    if n_events != n_classes or n_events < 2:
        raise ValueError(
            "Exact permutation loss requires the same number of events and classes."
        )
    if labels.numel() and (
        bool(torch.any(labels < 0)) or bool(torch.any(labels >= n_classes))
    ):
        raise ValueError("labels contain an out-of-range class index.")
    sorted_labels = torch.sort(labels, dim=1).values
    expected = torch.arange(n_classes, device=labels.device).expand_as(sorted_labels)
    if labels.numel() and not bool(torch.all(sorted_labels == expected)):
        raise ValueError("Every label row must contain each class exactly once.")

    scale = 1.0 / _positive_float(temperature, "temperature")
    candidates = torch.as_tensor(
        permutation_indices(n_classes).copy(),
        dtype=torch.long,
        device=logits.device,
    )
    batch_size = int(logits.shape[0])
    n_permutations = int(candidates.shape[0])
    expanded = logits.unsqueeze(1).expand(
        batch_size,
        n_permutations,
        n_events,
        n_classes,
    )
    gather_indices = candidates.view(1, n_permutations, n_events, 1).expand(
        batch_size,
        n_permutations,
        n_events,
        1,
    )
    permutation_scores = (
        expanded.gather(3, gather_indices).squeeze(3).sum(dim=2) * scale
    )
    true_scores = (
        logits.gather(2, labels.unsqueeze(2)).squeeze(2).sum(dim=1) * scale
    )
    return (torch.logsumexp(permutation_scores, dim=1) - true_scores).mean()


__all__ = (
    "ExactPermutationDecodingResult",
    "exact_permutation_decode",
    "permutation_indices",
    "torch_exact_permutation_nll",
)
