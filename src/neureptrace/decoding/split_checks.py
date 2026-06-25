from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence, Hashable
import numpy as np


@dataclass(frozen=True, slots=True)
class SplitCheckResult:
    ok: bool
    first_groups: tuple[Hashable, ...]
    second_groups: tuple[Hashable, ...]
    shared_groups: tuple[Hashable, ...]


def disjoint_group_check(first_groups: Sequence[Hashable], second_groups: Sequence[Hashable]) -> SplitCheckResult:
    first = tuple(dict.fromkeys(list(first_groups)))
    second = tuple(dict.fromkeys(list(second_groups)))
    second_set = set(second)
    shared = tuple(value for value in first if value in second_set)
    return SplitCheckResult(ok=len(shared) == 0, first_groups=first, second_groups=second, shared_groups=shared)


def mask_group_check(groups: Sequence[Hashable], first_mask: Sequence[bool], second_mask: Sequence[bool]) -> SplitCheckResult:
    group_values = np.asarray(list(groups), dtype=object).reshape(-1)
    first = np.asarray(first_mask, dtype=bool).reshape(-1)
    second = np.asarray(second_mask, dtype=bool).reshape(-1)
    if group_values.shape[0] != first.shape[0] or group_values.shape[0] != second.shape[0]:
        raise ValueError("groups and masks must have the same length.")
    return disjoint_group_check(group_values[first].tolist(), group_values[second].tolist())
