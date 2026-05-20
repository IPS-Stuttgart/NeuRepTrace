"""Generic FieldTrip-style MATLAB ``.mat`` loader.

This module is intentionally dataset-agnostic. It provides the reusable pieces
that PyMEGDec previously had to own for a specific MEG study: participant file
discovery, MATLAB struct access, FieldTrip trial/time/label extraction, safe
channel-label trimming, and trialinfo-to-metadata conversion.

Dataset-specific naming conventions such as ``Part{participant}Data.mat`` or
paper-specific metadata meanings should be supplied by config files or by thin
project wrappers, not hard-coded in NeuRepTrace.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetadataColumnSpec:
    """Describe one metadata column derived from FieldTrip ``trialinfo``.

    Parameters
    ----------
    name:
        Output column name in the returned metadata table.
    index:
        Zero-based column index in ``trialinfo``.
    optional:
        If true, missing columns are filled with ``NA`` instead of raising.
    """

    name: str
    index: int
    optional: bool = False


@dataclass(frozen=True)
class FieldTripMatSpec:
    """Schema for converting one FieldTrip-style MATLAB struct to MNE epochs."""

    variable: str | None = None
    trial_field: str = "trial"
    time_field: str = "time"
    label_field: str = "label"
    trialinfo_field: str | None = "trialinfo"
    sensor_geometry_field: str | None = "grad"
    metadata_columns: tuple[MetadataColumnSpec, ...] = field(default_factory=tuple)
    ch_types: str | Sequence[str] = "mag"
    trim_channel_labels_to_data: bool = True
    require_equal_trial_time_lengths: bool = True
    require_trialinfo_rows_equal_trials: bool = True
    participant: str | int | None = None
    condition: str | None = None


@dataclass(frozen=True)
class ParticipantMatFiles:
    """Resolved MATLAB files for one participant."""

    participant: str
    main_file: Path
    cue_file: Path | None = None


def _loadmat(path: Path) -> dict[str, Any]:
    """Load a MATLAB file while keeping structs easy to introspect."""

    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError("Loading MATLAB .mat files requires scipy. Install scipy or mne with its standard dependencies.") from exc

    return loadmat(path, squeeze_me=True, struct_as_record=False)


def _public_mat_variables(mat: Mapping[str, Any]) -> list[str]:
    return [name for name in mat if not name.startswith("__")]


def _select_mat_variable(mat: Mapping[str, Any], variable: str | None) -> Any:
    """Return the requested MATLAB variable or infer the only public variable."""

    if variable is not None:
        try:
            return mat[variable]
        except KeyError as exc:
            public = ", ".join(_public_mat_variables(mat)) or "<none>"
            raise KeyError(f"MATLAB variable '{variable}' not found. Available public variables: {public}.") from exc

    public = _public_mat_variables(mat)
    if len(public) != 1:
        raise ValueError(
            "Could not infer MATLAB data variable because the file contains "
            f"{len(public)} public variables ({', '.join(public) or '<none>'}). "
            "Set FieldTripMatSpec.variable explicitly."
        )
    return mat[public[0]]


def _get_field(obj: Any, name: str, *, required: bool = True) -> Any:
    """Access ``name`` on MATLAB structs, numpy records, dicts, or Python objects."""

    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        obj = obj.item()

    if isinstance(obj, Mapping) and name in obj:
        return obj[name]
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if required:
        raise KeyError(f"Required FieldTrip field '{name}' not found.")
    return None


def _as_sequence(value: Any) -> list[Any]:
    """Normalize MATLAB cell arrays or scalar values to a Python list."""

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return [item for item in value.ravel()]
        if value.ndim == 0:
            return [value.item()]
        return [item for item in value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            if value.ndim == 0:
                return str(value.item())
            return "".join(str(part) for part in value.tolist())
        if value.ndim == 0:
            return _normalize_string(value.item())
    return str(value)


def _normalize_labels(labels: Any) -> list[str]:
    """Convert MATLAB label arrays/cells into a plain Python string list."""

    if isinstance(labels, np.ndarray) and labels.dtype.kind in {"U", "S"} and labels.ndim == 1:
        return [_normalize_string(label) for label in labels]
    return [_normalize_string(label) for label in _as_sequence(labels)]


def _normalize_trials(trials: Any) -> list[np.ndarray]:
    """Return trials as ``n_channels x n_times`` float arrays."""

    if isinstance(trials, np.ndarray) and trials.ndim == 3:
        # FieldTrip exports can occasionally be stacked as either
        # n_trials x n_channels x n_times or n_channels x n_times x n_trials.
        # scipy may preserve such numeric stacks as dtype=object after savemat;
        # convert to float before deciding the trial axis.
        stacked = np.asarray(trials, dtype=float)
        if stacked.shape[0] <= stacked.shape[-1]:
            return [np.asarray(trial, dtype=float) for trial in stacked]
        return [np.asarray(stacked[:, :, idx], dtype=float) for idx in range(stacked.shape[2])]

    normalized = [np.asarray(trial, dtype=float) for trial in _as_sequence(trials)]
    for idx, trial in enumerate(normalized):
        if trial.ndim != 2:
            raise ValueError(f"Trial {idx} has shape {trial.shape}; expected a 2-D channels x time matrix.")
    return normalized


def _normalize_times(times: Any, n_trials: int) -> list[np.ndarray]:
    """Return one time vector per trial."""

    if isinstance(times, np.ndarray) and times.ndim == 1:
        return [np.asarray(times, dtype=float) for _ in range(n_trials)]
    if isinstance(times, np.ndarray) and times.ndim == 2:
        return [np.asarray(row, dtype=float).ravel() for row in times]

    normalized = [np.asarray(time, dtype=float).ravel() for time in _as_sequence(times)]
    if len(normalized) == 1 and n_trials > 1:
        normalized = normalized * n_trials
    if len(normalized) != n_trials:
        raise ValueError(f"Found {len(normalized)} time vectors for {n_trials} trials.")
    return normalized


def _validate_trials_and_times(trials: Sequence[np.ndarray], times: Sequence[np.ndarray], *, require_equal_lengths: bool) -> None:
    if len(trials) == 0:
        raise ValueError("No trials found in FieldTrip data.")
    if len(trials) != len(times):
        raise ValueError(f"Found {len(trials)} trials but {len(times)} time vectors.")

    first_shape = trials[0].shape
    first_time = times[0]
    for idx, (trial, time) in enumerate(zip(trials, times, strict=True)):
        if trial.shape[1] != len(time):
            raise ValueError(f"Trial {idx} has {trial.shape[1]} samples but its time vector has {len(time)} entries.")
        if require_equal_lengths and trial.shape != first_shape:
            raise ValueError(f"Trial {idx} has shape {trial.shape}; expected {first_shape}.")
        if require_equal_lengths and not np.allclose(time, first_time):
            raise ValueError(f"Trial {idx} has a different time vector. Variable-length trials are not supported by MNE EpochsArray.")


def _labels_for_data(labels: Sequence[str], n_channels: int, *, trim: bool) -> list[str]:
    if len(labels) == n_channels:
        return list(labels)
    if len(labels) > n_channels and trim:
        warnings.warn(
            f"FieldTrip label list contains {len(labels)} channels but trial data has {n_channels}; trimming labels to the data shape.",
            RuntimeWarning,
            stacklevel=2,
        )
        return list(labels[:n_channels])
    raise ValueError(f"FieldTrip label count ({len(labels)}) does not match trial channel count ({n_channels}).")


def _sampling_frequency(time: np.ndarray) -> float:
    if len(time) < 2:
        raise ValueError("At least two time samples are required to infer the sampling frequency.")
    diffs = np.diff(time)
    median_diff = float(np.median(diffs))
    if not np.allclose(diffs, median_diff, rtol=1e-5, atol=1e-8):
        raise ValueError("Time vector is not uniformly sampled; cannot construct MNE EpochsArray.")
    if median_diff <= 0:
        raise ValueError("Time vector must be strictly increasing.")
    return 1.0 / median_diff


def _trialinfo_to_frame(
    trialinfo: Any,
    *,
    n_trials: int,
    columns: Sequence[MetadataColumnSpec],
    require_rows_equal_trials: bool,
) -> pd.DataFrame:
    if trialinfo is None:
        metadata = pd.DataFrame({"trial": np.arange(n_trials, dtype=int)})
    else:
        values = np.asarray(trialinfo)
        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values.reshape(-1, 1)

        if require_rows_equal_trials and values.shape[0] != n_trials:
            raise ValueError(f"trialinfo has {values.shape[0]} rows but FieldTrip data contains {n_trials} trials.")
        if values.shape[0] != n_trials:
            values = values[:n_trials]

        if columns:
            metadata = pd.DataFrame(index=np.arange(n_trials))
            for column in columns:
                if column.index >= values.shape[1]:
                    if column.optional:
                        metadata[column.name] = pd.NA
                        continue
                    raise ValueError(f"trialinfo column {column.index} requested for '{column.name}', but trialinfo has only {values.shape[1]} columns.")
                metadata[column.name] = values[:, column.index]
        else:
            metadata = pd.DataFrame(values, columns=[f"trialinfo_{idx}" for idx in range(values.shape[1])])
            metadata.insert(0, "trial", np.arange(n_trials, dtype=int))
    return metadata.reset_index(drop=True)


def _metadata_with_constants(metadata: pd.DataFrame, *, participant: str | int | None, condition: str | None) -> pd.DataFrame:
    metadata = metadata.copy()
    if participant is not None and "participant" not in metadata:
        metadata.insert(0, "participant", str(participant))
    if condition is not None and "condition" not in metadata:
        metadata["condition"] = condition
    return metadata


def load_fieldtrip_mat(path: str | Path, spec: FieldTripMatSpec | None = None) -> tuple[mne.EpochsArray, pd.DataFrame]:
    """Load a FieldTrip-style MATLAB file as MNE epochs and metadata.

    Parameters
    ----------
    path:
        MATLAB ``.mat`` file containing a FieldTrip-like data struct.
    spec:
        Field and validation schema. Defaults match common FieldTrip names
        (``trial``, ``time``, ``label``, ``trialinfo``).

    Returns
    -------
    epochs, metadata:
        An MNE ``EpochsArray`` and the corresponding trial metadata table.
    """

    spec = FieldTripMatSpec() if spec is None else spec
    mat_path = Path(path)
    data_struct = _select_mat_variable(_loadmat(mat_path), spec.variable)

    trials = _normalize_trials(_get_field(data_struct, spec.trial_field))
    times = _normalize_times(_get_field(data_struct, spec.time_field), len(trials))
    _validate_trials_and_times(trials, times, require_equal_lengths=spec.require_equal_trial_time_lengths)

    n_channels = int(trials[0].shape[0])
    labels = _labels_for_data(
        _normalize_labels(_get_field(data_struct, spec.label_field)),
        n_channels,
        trim=spec.trim_channel_labels_to_data,
    )
    sfreq = _sampling_frequency(times[0])
    data = np.stack(trials, axis=0)

    info = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=spec.ch_types)
    if spec.sensor_geometry_field is not None and _get_field(data_struct, spec.sensor_geometry_field, required=False) is not None:
        info["description"] = f"Loaded from {mat_path.name}; FieldTrip sensor geometry field '{spec.sensor_geometry_field}' was present but not converted."
    epochs = mne.EpochsArray(data, info, tmin=float(times[0][0]), verbose="error")

    trialinfo = _get_field(data_struct, spec.trialinfo_field, required=False) if spec.trialinfo_field is not None else None
    metadata = _trialinfo_to_frame(
        trialinfo,
        n_trials=len(trials),
        columns=spec.metadata_columns,
        require_rows_equal_trials=spec.require_trialinfo_rows_equal_trials,
    )
    metadata = _metadata_with_constants(metadata, participant=spec.participant, condition=spec.condition)
    epochs.metadata = metadata
    return epochs, metadata


def _expand_participants(participants: Iterable[str | int] | None) -> list[str] | None:
    if participants is None:
        return None
    return [str(participant) for participant in participants]


def discover_participant_mat_files(
    root: str | Path,
    *,
    participants: Iterable[str | int] | None = None,
    main_template: str = "Part{participant}Data.mat",
    cue_template: str | None = "Part{participant}CueData.mat",
    require_main: bool = True,
    require_cue: bool = False,
) -> list[ParticipantMatFiles]:
    """Resolve participant MATLAB files from configurable filename templates.

    If ``participants`` is omitted, participant identifiers are inferred from
    the first ``{participant}`` placeholder in ``main_template``. The default
    templates intentionally mirror the common PyMEGDec dataset convention while
    keeping it outside the decoding logic.
    """

    root_path = Path(root)
    participant_ids = _expand_participants(participants)
    if participant_ids is None:
        if "{participant}" not in main_template:
            raise ValueError("participants must be provided when main_template has no {participant} placeholder.")
        pattern = re.escape(main_template).replace(re.escape("{participant}"), r"(?P<participant>.+?)")
        regex = re.compile(f"^{pattern}$")
        cue_regex = None
        if cue_template is not None and "{participant}" in cue_template:
            cue_pattern = re.escape(cue_template).replace(re.escape("{participant}"), r"(?P<participant>.+?)")
            cue_regex = re.compile(f"^{cue_pattern}$")
        participant_ids = sorted(
            match.group("participant")
            for file in root_path.iterdir()
            if file.is_file() and not (cue_regex is not None and cue_regex.match(file.name))
            for match in [regex.match(file.name)]
            if match
        )

    resolved: list[ParticipantMatFiles] = []
    for participant in participant_ids:
        main_file = root_path / main_template.format(participant=participant)
        cue_file = root_path / cue_template.format(participant=participant) if cue_template is not None else None
        if require_main and not main_file.exists():
            raise FileNotFoundError(f"Missing main MATLAB file for participant {participant}: {main_file}")
        if require_cue and cue_file is not None and not cue_file.exists():
            raise FileNotFoundError(f"Missing cue MATLAB file for participant {participant}: {cue_file}")
        resolved.append(ParticipantMatFiles(participant=participant, main_file=main_file, cue_file=cue_file if cue_file and cue_file.exists() else None))
    return resolved
