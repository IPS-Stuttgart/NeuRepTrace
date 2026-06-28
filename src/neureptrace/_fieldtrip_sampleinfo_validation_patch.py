"""Runtime patches for stricter FieldTrip loader validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_SAMPLEINFO_ERROR = "sampleinfo must contain finite integer sample bounds."
_PATCH_MARKER = "_neureptrace_fieldtrip_sampleinfo_validation_patched"
_IO_BOOL_PATCH_MARKER = "_neureptrace_io_fieldtrip_bool_config_patched"
_IO_TRIAL_STACK_PATCH_MARKER = "_neureptrace_io_fieldtrip_trial_stack_patched"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _contains_boolean(value: np.ndarray) -> bool:
    """Return whether an array contains Python or NumPy boolean values."""

    if np.issubdtype(value.dtype, np.bool_):
        return True
    if value.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in value.ravel(order="C"))
    return False


def _validate_sample_bound_order(bounds: np.ndarray) -> None:
    """Reject FieldTrip sample intervals whose end precedes their start."""

    if bounds.ndim == 2 and bounds.shape[1] == 2 and np.any(bounds[:, 1] < bounds[:, 0]):
        raise ValueError(_SAMPLEINFO_ERROR)


def _as_integer_sample_bounds(array: np.ndarray) -> np.ndarray:
    """Validate and return integer FieldTrip sample-bound pairs."""

    if _contains_boolean(array):
        raise ValueError(_SAMPLEINFO_ERROR)

    if np.issubdtype(array.dtype, np.integer):
        bounds = array.astype(int, copy=False)
        _validate_sample_bound_order(bounds)
        return bounds

    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_SAMPLEINFO_ERROR) from exc

    if not np.all(np.isfinite(numeric)):
        raise ValueError(_SAMPLEINFO_ERROR)

    rounded = np.round(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError(_SAMPLEINFO_ERROR)
    bounds = rounded.astype(int, copy=False)
    _validate_sample_bound_order(bounds)
    return bounds


def parse_bool_config(value: Any, *, name: str) -> bool:
    """Parse booleans from Python/YAML values without treating ``"false"`` as true."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if int(value) in {0, 1}:
            return bool(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def _metadata_columns_from_config(io_fieldtrip_mat: Any, config: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata_config = config.get("metadata", {}) or {}
    columns = metadata_config.get("columns", []) if isinstance(metadata_config, Mapping) else []
    specs: list[Any] = []
    for column_index, column in enumerate(columns):
        if not isinstance(column, Mapping):
            raise ValueError("metadata.columns entries must be mappings")
        specs.append(
            io_fieldtrip_mat.MetadataColumnSpec(
                name=str(column["name"]),
                index=io_fieldtrip_mat._metadata_column_index(column["index"]),
                optional=parse_bool_config(column.get("optional", False), name=f"metadata.columns[{column_index}].optional"),
            )
        )
    return tuple(specs)


def _install_sampleinfo_patch() -> None:
    import neureptrace.fieldtrip_mat as fieldtrip_mat

    if getattr(fieldtrip_mat._sampleinfo_array, _PATCH_MARKER, False):
        return

    def _sampleinfo_array(value: Any | None, *, n_trials: int) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(fieldtrip_mat._unwrap_scalar_object(value))
        if array.size == 0:
            return None
        if array.ndim == 1:
            if array.size != n_trials * 2:
                raise ValueError(f"sampleinfo must have shape {(n_trials, 2)}, got {array.shape}.")
            array = array.reshape(n_trials, 2)
        if array.ndim > 1 and array.shape[0] != n_trials and array.shape[-1] == n_trials:
            array = array.T
        if array.shape != (n_trials, 2):
            raise ValueError(f"sampleinfo must have shape {(n_trials, 2)}, got {array.shape}.")
        return _as_integer_sample_bounds(array)

    setattr(_sampleinfo_array, _PATCH_MARKER, True)
    fieldtrip_mat._sampleinfo_array = _sampleinfo_array


def _candidate_time_lengths(io_fieldtrip_mat: Any, value: Any) -> set[int]:
    """Return plausible samples-per-trial counts from a FieldTrip time field."""

    if isinstance(value, np.ndarray):
        try:
            numeric = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None:
            if numeric.ndim == 1:
                return {int(numeric.shape[0])}
            if numeric.ndim == 2:
                if 1 in numeric.shape:
                    return {int(max(numeric.shape))}
                return {int(numeric.shape[0]), int(numeric.shape[1])}

    lengths: set[int] = set()
    for item in io_fieldtrip_mat._as_sequence(value):
        try:
            vector = np.asarray(item, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if vector.size:
            lengths.add(int(vector.size))
    return lengths


def _normalize_trials_with_axis_hints(
    io_fieldtrip_mat: Any,
    trials: Any,
    *,
    n_channels: int | None,
    time_lengths: set[int],
) -> list[np.ndarray]:
    """Normalize trials while disambiguating 3-D FieldTrip stacks using labels/time."""

    if isinstance(trials, np.ndarray) and trials.ndim == 3:
        stacked = np.asarray(trials, dtype=float)
        candidates: list[tuple[str, list[np.ndarray]]] = []

        if (n_channels is None or stacked.shape[1] == n_channels) and (not time_lengths or stacked.shape[2] in time_lengths):
            candidates.append(("trial_channel_time", [np.asarray(trial, dtype=float) for trial in stacked]))
        if (n_channels is None or stacked.shape[0] == n_channels) and (not time_lengths or stacked.shape[1] in time_lengths):
            candidates.append(("channel_time_trial", [np.asarray(stacked[:, :, idx], dtype=float) for idx in range(stacked.shape[2])]))

        if len(candidates) == 1:
            return candidates[0][1]
        if len(candidates) > 1:
            # Preserve the historical interpretation when labels and time do not
            # uniquely identify the trial axis.
            return candidates[0][1]
        if n_channels is not None or time_lengths:
            raise ValueError(
                "Could not infer the trial axis for a 3-D FieldTrip trial array "
                f"with shape {stacked.shape}; labels imply {n_channels} channels "
                f"and time vectors imply sample counts {sorted(time_lengths) or '<unknown>'}."
            )

    return io_fieldtrip_mat._normalize_trials(trials)


def _install_io_trial_stack_patch() -> None:
    import neureptrace.io.fieldtrip_mat as io_fieldtrip_mat

    if getattr(io_fieldtrip_mat.load_fieldtrip_mat, _IO_TRIAL_STACK_PATCH_MARKER, False):
        return

    def load_fieldtrip_mat(path: str | Path, spec: Any | None = None) -> tuple[Any, Any]:
        """Load FieldTrip MATLAB data while inferring 3-D trial-stack axes from metadata."""

        spec = io_fieldtrip_mat.FieldTripMatSpec() if spec is None else spec
        mat_path = Path(path)
        data_struct = io_fieldtrip_mat._select_mat_variable(io_fieldtrip_mat._loadmat(mat_path), spec.variable, spec.variable_candidates)

        trial_field = io_fieldtrip_mat._get_field(data_struct, spec.trial_field)
        time_field = io_fieldtrip_mat._get_field(data_struct, spec.time_field)
        raw_labels = io_fieldtrip_mat._normalize_labels(io_fieldtrip_mat._get_field(data_struct, spec.label_field))
        trials = _normalize_trials_with_axis_hints(
            io_fieldtrip_mat,
            trial_field,
            n_channels=len(raw_labels),
            time_lengths=_candidate_time_lengths(io_fieldtrip_mat, time_field),
        )
        times = io_fieldtrip_mat._normalize_times(time_field, len(trials))
        io_fieldtrip_mat._validate_trials_and_times(trials, times, require_equal_lengths=spec.require_equal_trial_time_lengths)

        n_channels = int(trials[0].shape[0])
        labels = io_fieldtrip_mat._labels_for_data(raw_labels, n_channels, trim=spec.trim_channel_labels_to_data)
        sfreq = io_fieldtrip_mat._sampling_frequency(times[0])
        data = np.stack(trials, axis=0)

        info = io_fieldtrip_mat.mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=spec.ch_types)
        if spec.sensor_geometry_field is not None and io_fieldtrip_mat._get_field(data_struct, spec.sensor_geometry_field, required=False) is not None:
            info["description"] = f"Loaded from {mat_path.name}; FieldTrip sensor geometry field '{spec.sensor_geometry_field}' was present but not converted."
        epochs = io_fieldtrip_mat.mne.EpochsArray(data, info, tmin=float(times[0][0]), verbose="error")

        trialinfo = io_fieldtrip_mat._get_field(data_struct, spec.trialinfo_field, required=False) if spec.trialinfo_field is not None else None
        metadata = io_fieldtrip_mat._trialinfo_to_frame(
            trialinfo,
            n_trials=len(trials),
            columns=spec.metadata_columns,
            require_rows_equal_trials=spec.require_trialinfo_rows_equal_trials,
        )
        metadata = io_fieldtrip_mat._metadata_with_constants(metadata, participant=spec.participant, condition=spec.condition)
        epochs.metadata = metadata
        return epochs, metadata

    setattr(load_fieldtrip_mat, _IO_TRIAL_STACK_PATCH_MARKER, True)
    io_fieldtrip_mat.load_fieldtrip_mat = load_fieldtrip_mat


def _install_io_bool_config_patch() -> None:
    import neureptrace.io.fieldtrip_mat as io_fieldtrip_mat

    if getattr(io_fieldtrip_mat.load_fieldtrip_mat_epochs, _IO_BOOL_PATCH_MARKER, False):
        return

    def _patched_metadata_columns_from_config(config: Mapping[str, Any]) -> tuple[Any, ...]:
        return _metadata_columns_from_config(io_fieldtrip_mat, config)

    def load_fieldtrip_mat_epochs(
        path: str | Path,
        config: Mapping[str, Any] | None = None,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Load FieldTrip MATLAB epochs using strict boolean config parsing."""

        config = dict(config or {})
        fields = config.get("fields", {}) or {}
        if not isinstance(fields, Mapping):
            raise ValueError("fields must be a mapping when provided")
        validation = config.get("validation", {}) or {}
        if not isinstance(validation, Mapping):
            raise ValueError("validation must be a mapping when provided")
        spec = io_fieldtrip_mat.FieldTripMatSpec(
            variable=config.get("variable"),
            variable_candidates=io_fieldtrip_mat._string_tuple(config.get("variable_candidates", config.get("struct_candidates"))),
            trial_field=str(fields.get("trial", config.get("trial_field", "trial"))),
            time_field=str(fields.get("time", config.get("time_field", "time"))),
            label_field=str(fields.get("label", config.get("label_field", "label"))),
            trialinfo_field=fields.get("trialinfo", config.get("trialinfo_field", "trialinfo")),
            sensor_geometry_field=fields.get("sensor_geometry", config.get("sensor_geometry_field", "grad")),
            metadata_columns=_patched_metadata_columns_from_config(config),
            ch_types=config.get("channel_type", config.get("ch_types", "mag")),
            trim_channel_labels_to_data=parse_bool_config(
                validation.get("trim_channel_labels_to_data", config.get("trim_channel_labels_to_data", True)),
                name="trim_channel_labels_to_data",
            ),
            require_equal_trial_time_lengths=parse_bool_config(
                validation.get("require_equal_trial_time_lengths", True),
                name="require_equal_trial_time_lengths",
            ),
            require_trialinfo_rows_equal_trials=parse_bool_config(
                validation.get("require_trialinfo_rows_equal_trials", True),
                name="require_trialinfo_rows_equal_trials",
            ),
            condition=config.get("condition"),
        )
        epochs, metadata = io_fieldtrip_mat.load_fieldtrip_mat(path, spec)
        metadata = metadata.reset_index(drop=True).copy()
        data = epochs.get_data(copy=True)
        metadata, data = io_fieldtrip_mat._apply_metadata_transforms(metadata, data, config)
        for key, value in dict(extra_metadata or {}).items():
            if key not in metadata:
                metadata[key] = value
        return io_fieldtrip_mat.EpochDataset(
            data=data,
            times=epochs.times.copy(),
            channel_names=list(epochs.ch_names),
            metadata=metadata,
            name=str(config.get("name") or Path(path).stem),
            provenance={"path": str(path), "loader": "fieldtrip_mat"},
        )

    setattr(load_fieldtrip_mat_epochs, _IO_BOOL_PATCH_MARKER, True)
    io_fieldtrip_mat._metadata_columns_from_config = _patched_metadata_columns_from_config
    io_fieldtrip_mat.load_fieldtrip_mat_epochs = load_fieldtrip_mat_epochs


def install() -> None:
    """Install strict FieldTrip validation patches."""

    _install_sampleinfo_patch()
    _install_io_trial_stack_patch()
    _install_io_bool_config_patch()


__all__ = ["install", "parse_bool_config"]
