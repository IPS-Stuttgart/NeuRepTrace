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

from neureptrace.io.dataset import EpochDataset


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
    variable_candidates: tuple[str, ...] = field(default_factory=tuple)
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


def _select_mat_variable(mat: Mapping[str, Any], variable: str | None, variable_candidates: Sequence[str] = ()) -> Any:
    """Return the requested MATLAB variable or infer the only public variable."""

    if variable is not None:
        try:
            return mat[variable]
        except KeyError as exc:
            public = ", ".join(_public_mat_variables(mat)) or "<none>"
            raise KeyError(f"MATLAB variable '{variable}' not found. Available public variables: {public}.") from exc

    for candidate in variable_candidates:
        if candidate in mat:
            return mat[candidate]

    public = _public_mat_variables(mat)
    if len(public) != 1:
        raise ValueError(
            "Could not infer MATLAB data variable because the file contains "
            f"{len(public)} public variables ({', '.join(public) or '<none>'}). "
            "Set FieldTripMatSpec.variable explicitly or provide variable_candidates."
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
            return "".join(str(part) for part in value.ravel(order="C").tolist()).rstrip()
        if value.dtype == object:
            if value.size == 1:
                return _normalize_string(value.reshape(-1)[0])
            return "".join(_normalize_string(item) for item in value.ravel(order="C")).rstrip()
        if value.ndim == 0:
            return _normalize_string(value.item())
    return str(value)


def _normalize_labels(labels: Any) -> list[str]:
    """Convert MATLAB label arrays/cells into a plain Python string list."""

    if isinstance(labels, np.ndarray) and labels.dtype.kind in {"U", "S"} and labels.ndim == 1:
        return [_normalize_string(label) for label in labels]
    if isinstance(labels, np.ndarray) and labels.ndim == 2 and labels.dtype.kind in {"O", "U", "S"}:
        return ["".join(_normalize_string(part) for part in row).rstrip() for row in labels]
    return [_normalize_string(label) for label in _as_sequence(labels)]


def _numeric_matrix_or_none(value: np.ndarray) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return matrix if matrix.ndim == 2 else None


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

    if isinstance(trials, np.ndarray) and trials.ndim == 2:
        matrix = _numeric_matrix_or_none(trials)
        if matrix is not None:
            return [matrix]

    normalized = [np.asarray(trial, dtype=float) for trial in _as_sequence(trials)]
    for idx, trial in enumerate(normalized):
        if trial.ndim != 2:
            raise ValueError(f"Trial {idx} has shape {trial.shape}; expected a 2-D channels x time matrix.")
    return normalized


def _normalize_times(times: Any, n_trials: int) -> list[np.ndarray]:
    """Return one time vector per trial."""

    if isinstance(times, np.ndarray):
        numeric_times: np.ndarray | None
        try:
            numeric_times = np.asarray(times, dtype=float)
        except (TypeError, ValueError):
            numeric_times = None
        if numeric_times is not None:
            if numeric_times.ndim == 1:
                return [numeric_times for _ in range(n_trials)]
            if numeric_times.ndim == 2:
                return [np.asarray(row, dtype=float).ravel() for row in numeric_times]

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
            values = values.reshape(1, -1) if n_trials == 1 else values.reshape(-1, 1)

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
    data_struct = _select_mat_variable(_loadmat(mat_path), spec.variable, spec.variable_candidates)

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


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    return tuple(str(item) for item in value)


def _metadata_columns_from_config(config: Mapping[str, Any]) -> tuple[MetadataColumnSpec, ...]:
    metadata_config = config.get("metadata", {}) or {}
    columns = metadata_config.get("columns", []) if isinstance(metadata_config, Mapping) else []
    specs: list[MetadataColumnSpec] = []
    for column in columns:
        if not isinstance(column, Mapping):
            raise ValueError("metadata.columns entries must be mappings")
        specs.append(
            MetadataColumnSpec(
                name=str(column["name"]),
                index=int(column["index"]),
                optional=bool(column.get("optional", False)),
            )
        )
    return tuple(specs)


def _metadata_maps(config: Mapping[str, Any]) -> dict[str, Any]:
    metadata = config.get("metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return {}
    return dict(metadata.get("maps", {}) or {})


def _metadata_filters(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = config.get("metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return []
    return list(metadata.get("filters", []) or [])


def _lookup_mapping(mapping: Mapping[Any, Any], value: Any) -> Any:
    if value in mapping:
        return mapping[value]
    text = str(value)
    if text in mapping:
        return mapping[text]
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and numeric in mapping:
        return mapping[numeric]
    return value


def _apply_metadata_maps(metadata: pd.DataFrame, maps: Mapping[str, Any]) -> pd.DataFrame:
    if not maps:
        return metadata
    mapped = metadata.copy()
    for column_name, mapping in maps.items():
        if column_name not in mapped.columns:
            raise ValueError(f"metadata.maps configured unknown column '{column_name}'.")
        if not isinstance(mapping, Mapping):
            raise ValueError(f"metadata.maps.{column_name} must be a mapping.")
        mapped[column_name] = mapped[column_name].map(
            lambda value, mapping=mapping: _lookup_mapping(mapping, value)
        )
    return mapped


def _filter_mask(metadata: pd.DataFrame, filters: list[dict[str, Any]]) -> np.ndarray:
    mask = np.ones(len(metadata), dtype=bool)
    for filter_spec in filters:
        column_name = str(filter_spec["column"])
        if column_name not in metadata.columns:
            raise ValueError(f"metadata.filters configured unknown column '{column_name}'.")
        values = metadata[column_name]
        if "include" in filter_spec:
            include = filter_spec["include"]
            if not isinstance(include, list):
                include = [include]
            mask &= values.isin(include).to_numpy()
        if "exclude" in filter_spec:
            exclude = filter_spec["exclude"]
            if not isinstance(exclude, list):
                exclude = [exclude]
            mask &= ~values.isin(exclude).to_numpy()
    return mask


def _apply_metadata_transforms(
    metadata: pd.DataFrame,
    data: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    transformed = _apply_metadata_maps(metadata, _metadata_maps(config))
    filters = _metadata_filters(config)
    if not filters:
        return transformed, data
    mask = _filter_mask(transformed, filters)
    if not np.any(mask):
        raise ValueError("metadata.filters removed all trials.")
    return transformed.loc[mask].reset_index(drop=True), data[mask]


def load_fieldtrip_mat_epochs(
    path: str | Path,
    config: Mapping[str, Any] | None = None,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
) -> EpochDataset:
    """Load a FieldTrip MATLAB file into the neutral ``EpochDataset`` form.

    ``metadata.maps`` and ``metadata.filters`` can recode and select trials
    without Python code in project-specific call sites.
    """

    config = dict(config or {})
    fields = config.get("fields", {}) or {}
    if not isinstance(fields, Mapping):
        raise ValueError("fields must be a mapping when provided")
    validation = config.get("validation", {}) or {}
    if not isinstance(validation, Mapping):
        raise ValueError("validation must be a mapping when provided")
    spec = FieldTripMatSpec(
        variable=config.get("variable"),
        variable_candidates=_string_tuple(config.get("variable_candidates", config.get("struct_candidates"))),
        trial_field=str(fields.get("trial", config.get("trial_field", "trial"))),
        time_field=str(fields.get("time", config.get("time_field", "time"))),
        label_field=str(fields.get("label", config.get("label_field", "label"))),
        trialinfo_field=fields.get(
            "trialinfo", config.get("trialinfo_field", "trialinfo")
        ),
        sensor_geometry_field=fields.get(
            "sensor_geometry", config.get("sensor_geometry_field", "grad")
        ),
        metadata_columns=_metadata_columns_from_config(config),
        ch_types=config.get("channel_type", config.get("ch_types", "mag")),
        trim_channel_labels_to_data=bool(
            validation.get(
                "trim_channel_labels_to_data",
                config.get("trim_channel_labels_to_data", True),
            )
        ),
        require_equal_trial_time_lengths=bool(
            validation.get("require_equal_trial_time_lengths", True)
        ),
        require_trialinfo_rows_equal_trials=bool(
            validation.get("require_trialinfo_rows_equal_trials", True)
        ),
        condition=config.get("condition"),
    )
    epochs, metadata = load_fieldtrip_mat(path, spec)
    metadata = metadata.reset_index(drop=True).copy()
    data = epochs.get_data(copy=True)
    metadata, data = _apply_metadata_transforms(metadata, data, config)
    for key, value in dict(extra_metadata or {}).items():
        if key not in metadata:
            metadata[key] = value
    return EpochDataset(
        data=data,
        times=epochs.times.copy(),
        channel_names=list(epochs.ch_names),
        metadata=metadata,
        name=str(config.get("name") or Path(path).stem),
        provenance={"path": str(path), "loader": "fieldtrip_mat"},
    )


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
