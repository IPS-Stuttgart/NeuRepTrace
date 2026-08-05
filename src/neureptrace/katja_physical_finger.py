"""Physical-finger semantics for the Katja button-press benchmark.

The Julia-comparable target uses four participant-local classes even though the
recorded button code is a global five-finger identity.  Participants omit their
fixed first finger from the four variable events, so sorted local classes can
refer to different physical fingers across participants.  This module provides
validated mappings between the two label spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class ParticipantPhysicalFingerMap:
    """One participant's four variable codes and complementary fixed code."""

    participant: str
    variable_codes: tuple[Any, ...]
    fixed_code: Any


def _sorted_unique(values: Sequence[Any] | np.ndarray) -> tuple[Any, ...]:
    items = np.unique(np.asarray(values)).tolist()
    try:
        return tuple(sorted(items))
    except TypeError:
        return tuple(sorted(items, key=str))


def infer_global_physical_codes(
    finger_codes: Sequence[Any] | np.ndarray,
    *,
    included_mask: Sequence[bool] | np.ndarray | None = None,
    expected_codes: int = 5,
) -> tuple[Any, ...]:
    """Infer and validate the global physical-finger code universe."""

    codes = np.asarray(finger_codes)
    if codes.ndim != 1:
        raise ValueError("finger_codes must be one-dimensional.")
    mask = (
        np.ones(codes.shape[0], dtype=bool)
        if included_mask is None
        else np.asarray(included_mask, dtype=bool)
    )
    if mask.shape != codes.shape:
        raise ValueError("included_mask must match finger_codes.")
    if not np.any(mask):
        raise ValueError("included_mask selects no finger codes.")
    result = _sorted_unique(codes[mask])
    if len(result) != int(expected_codes):
        raise ValueError(
            f"Expected {expected_codes} global physical codes; got {result!r}."
        )
    return result


def participant_physical_finger_maps(
    subjects: Sequence[Any] | np.ndarray,
    finger_codes: Sequence[Any] | np.ndarray,
    *,
    included_mask: Sequence[bool] | np.ndarray | None = None,
    global_codes: Sequence[Any] | np.ndarray | None = None,
    expected_variable_codes: int = 4,
) -> dict[str, ParticipantPhysicalFingerMap]:
    """Build participant-local-to-physical mappings from retained event rows."""

    subject_vector = np.asarray(subjects).astype(str)
    code_vector = np.asarray(finger_codes)
    if subject_vector.ndim != 1 or code_vector.ndim != 1:
        raise ValueError("subjects and finger_codes must be one-dimensional.")
    if subject_vector.shape != code_vector.shape:
        raise ValueError("subjects and finger_codes must contain the same rows.")
    mask = (
        np.ones(subject_vector.shape[0], dtype=bool)
        if included_mask is None
        else np.asarray(included_mask, dtype=bool)
    )
    if mask.shape != subject_vector.shape:
        raise ValueError("included_mask must match subjects.")
    universe = (
        infer_global_physical_codes(code_vector, included_mask=mask)
        if global_codes is None
        else tuple(global_codes)
    )
    expected = int(expected_variable_codes)
    if expected < 1 or len(universe) != expected + 1:
        raise ValueError(
            "global_codes must contain exactly one more code than the variable set."
        )

    result: dict[str, ParticipantPhysicalFingerMap] = {}
    for participant in dict.fromkeys(subject_vector[mask].tolist()):
        participant_mask = mask & (subject_vector == participant)
        variable_codes = _sorted_unique(code_vector[participant_mask])
        if len(variable_codes) != expected:
            raise ValueError(
                f"Participant {participant!r} has {len(variable_codes)} variable "
                f"codes; expected {expected}: {variable_codes!r}."
            )
        fixed_codes = tuple(code for code in universe if code not in variable_codes)
        if len(fixed_codes) != 1:
            raise ValueError(
                f"Participant {participant!r} has ambiguous fixed-code complement "
                f"{fixed_codes!r}."
            )
        result[participant] = ParticipantPhysicalFingerMap(
            participant=participant,
            variable_codes=variable_codes,
            fixed_code=fixed_codes[0],
        )
    return result


def physical_probabilities_to_local(
    probabilities: Sequence | np.ndarray,
    *,
    model_classes: Sequence[Any] | np.ndarray,
    variable_codes: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Mask the fixed physical finger and map probabilities to local classes.

    ``variable_codes`` defines the local class order.  Probability mass assigned
    to the participant's fixed finger is removed, then the four retained classes
    are renormalized event-wise.
    """

    tensor = np.asarray(probabilities, dtype=float)
    if tensor.ndim != 3 or min(tensor.shape) < 1:
        raise ValueError(
            "probabilities must have shape (trials, events, physical_classes)."
        )
    if not np.all(np.isfinite(tensor)) or np.any(tensor < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    classes = np.asarray(model_classes)
    if classes.ndim != 1 or classes.shape[0] != tensor.shape[2]:
        raise ValueError("model_classes must match the probability class axis.")
    local_codes = tuple(variable_codes)
    if len(local_codes) < 2 or len(set(local_codes)) != len(local_codes):
        raise ValueError("variable_codes must contain unique class codes.")

    class_indices: list[int] = []
    for code in local_codes:
        matches = np.flatnonzero(classes == code)
        if matches.size != 1:
            raise ValueError(
                f"Physical code {code!r} must occur exactly once in model_classes."
            )
        class_indices.append(int(matches[0]))
    local = tensor[:, :, class_indices]
    mass = local.sum(axis=2, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("Masking leaves an event with no probability mass.")
    return local / mass
