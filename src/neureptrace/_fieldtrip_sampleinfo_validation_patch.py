"""Runtime patches for stricter FieldTrip loader validation."""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_SAMPLEINFO_ERROR = "sampleinfo must contain finite integer sample bounds."
_PATCH_MARKER = "_neureptrace_fieldtrip_sampleinfo_validation_patched"
_LABEL_CONFIG_PATCH_MARKER = "_neureptrace_fieldtrip_label_config_validation_patched"
_TOPLEVEL_SHARED_TIME_VECTOR_PATCH_MARKER = "_neureptrace_fieldtrip_shared_time_vector_patched"
_IO_BOOL_PATCH_MARKER = "_neureptrace_io_fieldtrip_bool_config_patched"
_IO_SPEC_BOOL_PATCH_MARKER = "_neureptrace_io_fieldtrip_spec_bool_config_patched"
_IO_TRIAL_STACK_PATCH_MARKER = "_neureptrace_io_fieldtrip_trial_stack_patched"
_IO_SHARED_TIME_VECTOR_PATCH_MARKER = "_neureptrace_io_fieldtrip_shared_time_vector_patched"
_LABEL_BASE_ERROR = "label_base must be a finite numeric scalar or None, not a boolean value."
_LABEL_BASE_PARSE_ERROR = "label-base must be finite numeric or 'none'."
_TRIALINFO_COLUMN_ERROR = "trialinfo_column must be an integer column index, not a boolean value."
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_IO_SPEC_BOOL_FIELDS = (
    "trim_channel_labels_to_data",
    "require_equal_trial_time_lengths",
    "require_trialinfo_rows_equal_trials",
)


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

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a boolean value.")
        return parse_bool_config(value.item(), name=name)
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


def _scalar_value_for_numeric_config(value: Any, *, message: str) -> Any:
    """Return a scalar config value while rejecting booleans and arrays."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    return value


def _coerce_label_base(value: Any) -> float | None:
    """Normalize FieldTrip label-base controls without bool-to-float coercion."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return None
        value = text
    value = _scalar_value_for_numeric_config(value, message=_LABEL_BASE_ERROR)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_LABEL_BASE_ERROR) from exc
    if not np.isfinite(parsed):
        raise ValueError(_LABEL_BASE_ERROR)
    return parsed


def _coerce_trialinfo_column(value: Any) -> int:
    """Normalize a FieldTrip trialinfo column index without bool-as-int leakage."""

    value = _scalar_value_for_numeric_config(value, message=_TRIALINFO_COLUMN_ERROR)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_TRIALINFO_COLUMN_ERROR) from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(_TRIALINFO_COLUMN_ERROR)
    return int(parsed)


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


def _install_label_config_patch() -> None:
    import neureptrace.fieldtrip_mat as fieldtrip_mat

    if getattr(fieldtrip_mat._metadata_from_trialinfo, _LABEL_CONFIG_PATCH_MARKER, False):
        return

    original_parse_label_base = fieldtrip_mat._parse_label_base
    original_metadata_from_trialinfo = fieldtrip_mat._metadata_from_trialinfo

    def _parse_label_base(value: Any) -> float | None:
        try:
            return _coerce_label_base(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(_LABEL_BASE_PARSE_ERROR) from exc

    def _metadata_from_trialinfo(
        *,
        n_trials: int,
        trialinfo: np.ndarray | None,
        sampleinfo: np.ndarray | None,
        label_column: str,
        label_base: Any,
        trialinfo_column: Any,
    ) -> Any:
        return original_metadata_from_trialinfo(
            n_trials=n_trials,
            trialinfo=trialinfo,
            sampleinfo=sampleinfo,
            label_column=label_column,
            label_base=_coerce_label_base(label_base),
            trialinfo_column=_coerce_trialinfo_column(trialinfo_column),
        )

    setattr(_parse_label_base, _LABEL_CONFIG_PATCH_MARKER, True)
    setattr(_metadata_from_trialinfo, _LABEL_CONFIG_PATCH_MARKER, True)
    _parse_label_base.__wrapped__ = original_parse_label_base
    _metadata_from_trialinfo.__wrapped__ = original_metadata_from_trialinfo
    fieldtrip_mat._parse_label_base = _parse_label_base
    fieldtrip_mat._metadata_from_trialinfo = _metadata_from_trialinfo


def _install_top_level_shared_time_vector_patch() -> None:
    import neureptrace.fieldtrip_mat as fieldtrip_mat

    if getattr(fieldtrip_mat._times_to_array, _TOPLEVEL_SHARED_TIME_VECTOR_PATCH_MARKER, False):
        return

    original_times_to_array = fieldtrip_mat._times_to_array

    def _times_to_array(cells: list[Any], *, n_trials: int, n_times: int) -> np.ndarray:
        if len(cells) == 1 and n_trials > 1:
            vector = np.asarray(fieldtrip_mat._unwrap_scalar_object(cells[0]), dtype=float).ravel()
            if vector.size == n_times:
                cells = [vector.copy() for _ in range(n_trials)]
        return original_times_to_array(cells, n_trials=n_trials, n_times=n_times)

    setattr(_times_to_array, _TOPLEVEL_SHARED_TIME_VECTOR_PATCH_MARKER, True)
    fieldtrip_mat._times_to_array = _times_to_array


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


def _install_io_shared_time_vector_patch() -> None:
    import neureptrace.io.fieldtrip_mat as io_fieldtrip_mat

    if getattr(io_fieldtrip_mat._normalize_times, _IO_SHARED_TIME_VECTOR_PATCH_MARKER, False):
        return

    original_normalize_times = io_fieldtrip_mat._normalize_times

    def _normalize_times(times: Any, n_trials: int) -> list[np.ndarray]:
        if isinstance(times, np.ndarray):
            try:
                numeric_times = np.asarray(times, dtype=float)
            except (TypeError, ValueError):
                numeric_times = None
            if numeric_times is not None and numeric_times.ndim == 2 and 1 in numeric_times.shape:
                vector = numeric_times.ravel().astype(float, copy=False)
                return [vector.copy() for _ in range(n_trials)]
        return original_normalize_times(times, n_trials)

    setattr(_normalize_times, _IO_SHARED_TIME_VECTOR_PATCH_MARKER, True)
    io_fieldtrip_mat._normalize_times = _normalize_times


def _normalized_spec_bool_args(original_spec: type[Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    field_names = tuple(field.name for field in dataclasses.fields(original_spec))
    args_list = list(args)
    normalized_kwargs = dict(kwargs)
    for field_name in _IO_SPEC_BOOL_FIELDS:
        position = field_names.index(field_name)
        if position < len(args_list):
            args_list[position] = parse_bool_config(args_list[position], name=f"FieldTripMatSpec.{field_name}")
        if field_name in normalized_kwargs:
            normalized_kwargs[field_name] = parse_bool_config(normalized_kwargs[field_name], name=f"FieldTripMatSpec.{field_name}")
    return tuple(args_list), normalized_kwargs


def _install_io_spec_bool_config_patch() -> None:
    import neureptrace.io as io_package
    import neureptrace.io.fieldtrip_mat as io_fieldtrip_mat

    current_spec = io_fieldtrip_mat.FieldTripMatSpec
    if getattr(current_spec, _IO_SPEC_BOOL_PATCH_MARKER, False):
        return
    original_spec = current_spec

    class FieldTripMatSpec(original_spec):  # type: ignore[misc, valid-type]
        """FieldTrip spec constructor that preserves quoted boolean semantics."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            normalized_args, normalized_kwargs = _normalized_spec_bool_args(original_spec, args, kwargs)
            super().__init__(*normalized_args, **normalized_kwargs)

    FieldTripMatSpec.__name__ = original_spec.__name__
    FieldTripMatSpec.__qualname__ = original_spec.__qualname__
    FieldTripMatSpec.__module__ = original_spec.__module__
    FieldTripMatSpec.__wrapped__ = original_spec  # type: ignore[attr-defined]
    setattr(FieldTripMatSpec, _IO_SPEC_BOOL_PATCH_MARKER, True)
    io_fieldtrip_mat.FieldTripMatSpec = FieldTripMatSpec
    io_package.FieldTripMatSpec = FieldTripMatSpec


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
    _install_label_config_patch()
    _install_top_level_shared_time_vector_patch()
    _install_io_shared_time_vector_patch()
    _install_io_spec_bool_config_patch()
    _install_io_trial_stack_patch()
    _install_io_bool_config_patch()


__all__ = ["install", "parse_bool_config"]
