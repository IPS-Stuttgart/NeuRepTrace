"""Candidate-grid helpers for decoding model-selection workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CandidateGrid:
    """Declarative product grid for benchmark candidate configurations."""

    dimensions: Mapping[str, Iterable[Any]]
    fixed: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "dimensions", normalize_candidate_grid_dimensions(self.dimensions))
        object.__setattr__(self, "fixed", dict(self.fixed))
        overlapping = set(self.dimensions).intersection(self.fixed)
        if overlapping:
            raise ValueError(f"Candidate-grid fields cannot be both fixed and varied: {sorted(overlapping)}")

    def expand(self, *, factory: Callable[..., T] = dict) -> tuple[T, ...]:
        """Expand this grid into immutable candidate objects."""

        return expand_candidate_grid(self.dimensions, fixed=self.fixed, factory=factory)

    def rows(self, *, start_index: int = 1, index_key: str = "candidate_index") -> tuple[dict[str, Any], ...]:
        """Expand the grid into dictionaries that include a candidate index."""

        return candidate_grid_rows(self.dimensions, fixed=self.fixed, start_index=start_index, index_key=index_key)


def expand_candidate_grid(
    dimensions: Mapping[str, Iterable[Any]] | None = None,
    *,
    fixed: Mapping[str, Any] | None = None,
    factory: Callable[..., T] = dict,
    **dimension_kwargs: Iterable[Any],
) -> tuple[T, ...]:
    """Build the Cartesian product of candidate dimensions.

    ``dimensions`` and ``dimension_kwargs`` are merged in insertion order.  String
    values are treated as scalar grid values rather than iterables of characters.
    The default ``factory=dict`` returns dictionaries; dataclasses or project
    config classes can be passed directly when their constructors accept the grid
    field names as keyword arguments.
    """

    merged_dimensions: dict[str, Iterable[Any]] = {}
    if dimensions is not None:
        merged_dimensions.update(dimensions)
    merged_dimensions.update(dimension_kwargs)
    fixed_values = {} if fixed is None else dict(fixed)
    normalized = normalize_candidate_grid_dimensions(merged_dimensions)
    overlapping = set(normalized).intersection(fixed_values)
    if overlapping:
        raise ValueError(f"Candidate-grid fields cannot be both fixed and varied: {sorted(overlapping)}")
    keys = tuple(normalized)
    candidates: list[T] = []
    for values in _iter_product(tuple(normalized[key] for key in keys)):
        kwargs = {**fixed_values, **dict(zip(keys, values))}
        candidates.append(factory(**kwargs))
    return tuple(candidates)


def candidate_grid_rows(
    dimensions: Mapping[str, Iterable[Any]] | None = None,
    *,
    fixed: Mapping[str, Any] | None = None,
    start_index: int = 1,
    index_key: str = "candidate_index",
    **dimension_kwargs: Iterable[Any],
) -> tuple[dict[str, Any], ...]:
    """Expand a candidate grid into dictionaries with stable one-based indices."""

    if not index_key:
        raise ValueError("index_key must be a non-empty string.")
    start_index = int(start_index)
    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    rows = []
    for offset, row in enumerate(expand_candidate_grid(dimensions, fixed=fixed, **dimension_kwargs)):
        rows.append({index_key: start_index + offset, **row})
    return tuple(rows)


def normalize_candidate_grid_dimensions(dimensions: Mapping[str, Iterable[Any]]) -> dict[str, tuple[Any, ...]]:
    """Validate and tuple-normalize candidate-grid dimensions."""

    if dimensions is None:
        return {}
    normalized: dict[str, tuple[Any, ...]] = {}
    for key, values in dimensions.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("Candidate-grid field names must be non-empty strings.")
        normalized_values = tuple(_dimension_values(values))
        if not normalized_values:
            raise ValueError(f"Candidate-grid field {normalized_key!r} must contain at least one value.")
        normalized[normalized_key] = normalized_values
    return normalized


def _dimension_values(values: Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        return (values,)
    try:
        iterator = iter(values)
    except TypeError:
        return (values,)
    return tuple(iterator)


def _iter_product(dimensions: tuple[tuple[Any, ...], ...]):
    if not dimensions:
        yield ()
        return
    first, *rest = dimensions
    for value in first:
        for suffix in _iter_product(tuple(rest)):
            yield (value, *suffix)
