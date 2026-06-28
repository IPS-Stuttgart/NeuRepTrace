from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import scipy.io as sio

PathToken = str | int
DEFAULT_ROOT_PATH: tuple[PathToken, ...] = ("data", 0)
DEFAULT_TRIAL_PATH: tuple[PathToken, ...] = ("trial",)
DEFAULT_TIME_PATH: tuple[PathToken, ...] = ("time",)
DEFAULT_LABEL_PATH: tuple[PathToken, ...] = ("label",)
DEFAULT_TRIALINFO_PATH: tuple[PathToken, ...] = ("trialinfo",)
DEFAULT_SAMPLEINFO_PATH: tuple[PathToken, ...] = ("sampleinfo",)
INPUT_FORMAT_CHOICES = ("mne-epochs", "fieldtrip-mat")


@dataclass(frozen=True)
class FieldTripMatData:
    """Normalized FieldTrip raw/trial data loaded from a MATLAB ``.mat`` file."""

    trials: np.ndarray
    times: np.ndarray
    labels: tuple[str, ...]
    metadata: pd.DataFrame
    sfreq: float
    tmin: float


def parse_path_tokens(value: str | Sequence[str | int] | None, default: Sequence[PathToken]) -> tuple[PathToken, ...]:
    """Parse comma-separated or sequence path tokens into field/index tokens."""

    if value is None:
        return tuple(default)
    if isinstance(value, str):
        raw_tokens: Sequence[str | int] = [token.strip() for token in value.split(",") if token.strip()]
    else:
        raw_tokens = value
    tokens: list[PathToken] = []
    for token in raw_tokens:
        if isinstance(token, int):
            tokens.append(token)
            continue
        text = str(token).strip()
        tokens.append(int(text) if text.lstrip("+-").isdigit() else text)
    return tuple(tokens)


def load_fieldtrip_raw_mat_epochs(
    mat_path: Path | str,
    *,
    root_path: Sequence[PathToken] = DEFAULT_ROOT_PATH,
    trial_path: Sequence[PathToken] = DEFAULT_TRIAL_PATH,
    time_path: Sequence[PathToken] = DEFAULT_TIME_PATH,
    label_path: Sequence[PathToken] = DEFAULT_LABEL_PATH,
    trialinfo_path: Sequence[PathToken] | None = DEFAULT_TRIALINFO_PATH,
    sampleinfo_path: Sequence[PathToken] | None = DEFAULT_SAMPLEINFO_PATH,
    label_column: str = "condition",
    label_base: int | float | None = 1,
    trialinfo_column: int = 0,
    ch_type: str = "grad",
    trial_axis_order: str = "channel_time",
    trim_overlong_labels: bool = True,
) -> tuple[mne.EpochsArray, pd.DataFrame]:
    """Load a FieldTrip-like MATLAB raw struct as MNE epochs and metadata.

    The defaults match the Bush/PyMEGDec files: a top-level MATLAB variable
    named ``data`` containing FieldTrip-style ``trial``, ``time``, ``label``,
    ``trialinfo``, ``grad`` and ``sampleinfo`` fields. Overlong channel-level
    metadata is trimmed to the row count of the trial matrices and reported via
    ``RuntimeWarning``.
    """

    data = load_fieldtrip_raw_mat(
        mat_path,
        root_path=root_path,
        trial_path=trial_path,
        time_path=time_path,
        label_path=label_path,
        trialinfo_path=trialinfo_path,
        sampleinfo_path=sampleinfo_path,
        label_column=label_column,
        label_base=label_base,
        trialinfo_column=trialinfo_column,
        trial_axis_order=trial_axis_order,
        trim_overlong_labels=trim_overlong_labels,
    )
    info = mne.create_info(ch_names=list(data.labels), sfreq=data.sfreq, ch_types=[ch_type] * len(data.labels))
    event_labels = data.metadata[label_column].astype(str).to_numpy()
    class_names = sorted(pd.unique(event_labels), key=str)
    event_id = {class_name: class_index + 1 for class_index, class_name in enumerate(class_names)}
    events = np.column_stack(
        [
            np.arange(len(event_labels), dtype=int),
            np.zeros(len(event_labels), dtype=int),
            np.array([event_id[label] for label in event_labels], dtype=int),
        ]
    )
    epochs = mne.EpochsArray(data.trials, info, events=events, event_id=event_id, tmin=data.tmin, metadata=data.metadata.copy(), verbose="error")
    return epochs, data.metadata.copy()


def load_fieldtrip_raw_mat(
    mat_path: Path | str,
    *,
    root_path: Sequence[PathToken] = DEFAULT_ROOT_PATH,
    trial_path: Sequence[PathToken] = DEFAULT_TRIAL_PATH,
    time_path: Sequence[PathToken] = DEFAULT_TIME_PATH,
    label_path: Sequence[PathToken] = DEFAULT_LABEL_PATH,
    trialinfo_path: Sequence[PathToken] | None = DEFAULT_TRIALINFO_PATH,
    sampleinfo_path: Sequence[PathToken] | None = DEFAULT_SAMPLEINFO_PATH,
    label_column: str = "condition",
    label_base: int | float | None = 1,
    trialinfo_column: int = 0,
    trial_axis_order: str = "channel_time",
    trim_overlong_labels: bool = True,
) -> FieldTripMatData:
    mat = sio.loadmat(Path(mat_path), squeeze_me=False, struct_as_record=True)
    root = _follow_path(mat, root_path)
    trial_cells = _cell_vector(_follow_path(root, trial_path))
    time_cells = _cell_vector(_follow_path(root, time_path))
    trials = _trials_to_array(trial_cells, trial_axis_order=trial_axis_order)
    times = _times_to_array(time_cells, n_trials=trials.shape[0], n_times=trials.shape[2])
    sfreq, tmin = _sampling_properties(times)
    labels = _channel_labels(
        _follow_path(root, label_path),
        n_channels=trials.shape[1],
        trim_overlong_labels=trim_overlong_labels,
    )
    grad = _field_or_none(root, "grad")
    _warn_overlong_grad_fields(grad, n_channels=trials.shape[1], trim_overlong_labels=trim_overlong_labels)
    trialinfo = _trialinfo_array(_optional_follow_path(root, trialinfo_path), n_trials=trials.shape[0])
    sampleinfo = _sampleinfo_array(_optional_follow_path(root, sampleinfo_path), n_trials=trials.shape[0])
    metadata = _metadata_from_trialinfo(
        n_trials=trials.shape[0],
        trialinfo=trialinfo,
        sampleinfo=sampleinfo,
        label_column=label_column,
        label_base=label_base,
        trialinfo_column=trialinfo_column,
    )
    return FieldTripMatData(trials=trials, times=times, labels=tuple(labels), metadata=metadata, sfreq=sfreq, tmin=tmin)


def _follow_path(value: Any, path: Sequence[PathToken]) -> Any:
    current = value
    for token in path:
        if isinstance(token, str):
            current = _field(current, token)
        else:
            current = _index(current, token)
    return current


def _optional_follow_path(value: Any, path: Sequence[PathToken] | None) -> Any | None:
    if path is None:
        return None
    try:
        return _follow_path(value, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _field(value: Any, name: str) -> Any:
    value = _unwrap_scalar_object(value)
    if isinstance(value, dict):
        return value[name]
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return value[name]
    if isinstance(value, np.ndarray) and value.dtype.names and name in value.dtype.names:
        return value[name]
    raise KeyError(f"Field {name!r} not found in {type(value).__name__}.")


def _field_or_none(value: Any, name: str) -> Any | None:
    try:
        return _field(value, name)
    except KeyError:
        return None


def _index(value: Any, index: int) -> Any:
    if isinstance(value, np.ndarray):
        return value.ravel(order="C")[index]
    value = _unwrap_scalar_object(value)
    return value[index]


def _unwrap_scalar_object(value: Any) -> Any:
    current = value
    while isinstance(current, np.ndarray) and current.size == 1 and (current.dtype == object or current.dtype.names is not None):
        current = current.reshape(-1)[0]
    return current


def _cell_vector(value: Any) -> list[Any]:
    value = _unwrap_scalar_object(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_unwrap_scalar_object(item) for item in value.ravel(order="C")]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return [_unwrap_scalar_object(value.item())]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _trials_to_array(cells: list[Any], *, trial_axis_order: str) -> np.ndarray:
    trials = []
    for trial_index, cell in enumerate(cells):
        trial = np.asarray(_unwrap_scalar_object(cell), dtype=float)
        if trial.ndim != 2:
            raise ValueError(f"FieldTrip trial {trial_index} must be 2D, got shape {trial.shape}.")
        if trial_axis_order == "channel_time":
            trials.append(trial)
        elif trial_axis_order == "time_channel":
            trials.append(trial.T)
        else:
            raise ValueError("trial_axis_order must be 'channel_time' or 'time_channel'.")
    if not trials:
        raise ValueError("FieldTrip trial field is empty.")
    expected_shape = trials[0].shape
    for trial_index, trial in enumerate(trials):
        if trial.shape != expected_shape:
            raise ValueError(f"FieldTrip trial {trial_index} has shape {trial.shape}; expected {expected_shape}.")
    return np.stack(trials, axis=0)


def _times_to_array(cells: list[Any], *, n_trials: int, n_times: int) -> np.ndarray:
    vectors = []
    for trial_index, cell in enumerate(cells):
        vector = np.asarray(_unwrap_scalar_object(cell), dtype=float).ravel()
        if vector.size != n_times:
            raise ValueError(f"FieldTrip time vector {trial_index} has {vector.size} samples; expected {n_times}.")
        vectors.append(vector)
    if len(vectors) != n_trials:
        raise ValueError(f"FieldTrip time field has {len(vectors)} vectors; expected {n_trials}.")
    times = np.stack(vectors, axis=0)
    if not np.allclose(times, times[0][None, :], rtol=1e-7, atol=1e-12):
        raise ValueError("All FieldTrip time vectors must be identical for MNE EpochsArray conversion.")
    return times


def _sampling_properties(times: np.ndarray) -> tuple[float, float]:
    if times.shape[1] < 2:
        raise ValueError("At least two time samples are required to infer sampling frequency.")
    diffs = np.diff(times, axis=1)
    sample_interval = float(np.median(diffs[0]))
    if sample_interval <= 0 or not np.allclose(diffs, sample_interval, rtol=1e-6, atol=1e-12):
        raise ValueError("FieldTrip time vectors must be uniformly sampled with a common positive interval.")
    return 1.0 / sample_interval, float(times[0, 0])


def _matlab_string(value: Any) -> str:
    value = _unwrap_scalar_object(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            text = value.astype(str).ravel(order="C")
            return str(text[0]) if len(text) == 1 else "".join(text)
        if value.dtype == object:
            return "".join(_matlab_string(item) for item in value.ravel(order="C"))
        if value.size == 1:
            return str(value.item())
    return str(value)


def _string_vector(value: Any) -> list[str]:
    return [_matlab_string(item) for item in _cell_vector(value)]


def _channel_labels(value: Any, *, n_channels: int, trim_overlong_labels: bool) -> list[str]:
    labels = _string_vector(value)
    if len(labels) < n_channels:
        raise ValueError(f"FieldTrip data.label has {len(labels)} labels but trials have {n_channels} channels.")
    if len(labels) > n_channels:
        message = f"Trimming FieldTrip data.label from {len(labels)} to {n_channels} entries to match trial matrices."
        if not trim_overlong_labels:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        labels = labels[:n_channels]
    return _make_unique(labels)


def _make_unique(labels: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique = []
    for label in labels:
        count = counts.get(label, 0)
        counts[label] = count + 1
        unique.append(label if count == 0 else f"{label}-{count}")
    return unique


def _warn_overlong_grad_fields(grad: Any | None, *, n_channels: int, trim_overlong_labels: bool) -> None:
    if grad is None:
        return
    for field_name in ("label", "chantype", "chanunit", "chanpos", "chanori"):
        value = _field_or_none(grad, field_name)
        if value is None:
            continue
        array = _unwrap_scalar_object(value)
        if field_name in {"label", "chantype", "chanunit"}:
            n_entries = len(_string_vector(array))
        else:
            n_entries = int(np.asarray(array).shape[0]) if np.asarray(array).ndim > 0 else 1
        if n_entries > n_channels:
            message = f"Trimming/ignoring FieldTrip grad.{field_name} from {n_entries} to {n_channels} channel entries to match trial matrices."
            if not trim_overlong_labels:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        elif n_entries < n_channels:
            warnings.warn(
                f"FieldTrip grad.{field_name} has {n_entries} entries but trials have {n_channels} channels.",
                RuntimeWarning,
                stacklevel=2,
            )


def _trialinfo_array(value: Any | None, *, n_trials: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(_unwrap_scalar_object(value))
    if array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape[0] != n_trials and array.shape[-1] == n_trials:
        array = array.T
    if array.shape[0] != n_trials:
        raise ValueError(f"trialinfo has {array.shape[0]} rows; expected {n_trials}.")
    return array


def _sampleinfo_array(value: Any | None, *, n_trials: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(_unwrap_scalar_object(value))
    if array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    if array.shape[0] != n_trials and array.shape[-1] == n_trials:
        array = array.T
    if array.shape != (n_trials, 2):
        raise ValueError(f"sampleinfo must have shape {(n_trials, 2)}, got {array.shape}.")
    return array.astype(int, copy=False)


def _metadata_from_trialinfo(
    *,
    n_trials: int,
    trialinfo: np.ndarray | None,
    sampleinfo: np.ndarray | None,
    label_column: str,
    label_base: int | float | None,
    trialinfo_column: int,
) -> pd.DataFrame:
    metadata = pd.DataFrame({"trial": np.arange(n_trials, dtype=int)})
    if trialinfo is None:
        metadata[label_column] = np.arange(n_trials, dtype=int)
    else:
        if not 0 <= trialinfo_column < trialinfo.shape[1]:
            raise ValueError(f"trialinfo_column={trialinfo_column} outside trialinfo with {trialinfo.shape[1]} columns.")
        for column in range(trialinfo.shape[1]):
            metadata[f"trialinfo_{column}"] = trialinfo[:, column]
        labels = trialinfo[:, trialinfo_column]
        if label_base is not None:
            try:
                labels = labels.astype(float) - float(label_base)
                if np.allclose(labels, np.round(labels)):
                    labels = np.round(labels).astype(int)
            except (TypeError, ValueError) as exc:
                raise ValueError("label_base can only be applied to numeric trialinfo labels.") from exc
        metadata[label_column] = labels
        if trialinfo.shape[1] == 1:
            metadata["trialinfo"] = trialinfo[:, 0]
    if sampleinfo is not None:
        metadata["sample_start"] = sampleinfo[:, 0]
        metadata["sample_stop"] = sampleinfo[:, 1]
    return metadata


def write_fieldtrip_raw_mat_epochs(
    mat_path: Path | str,
    *,
    epochs_out: Path,
    metadata_out: Path | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> tuple[Path, Path]:
    epochs, metadata = load_fieldtrip_raw_mat_epochs(mat_path, **kwargs)
    metadata_path = metadata_out or epochs_out.with_name(f"{epochs_out.stem}_metadata.csv")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Metadata output already exists: {metadata_path}. Pass overwrite=True to replace it.")
    epochs_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(epochs_out, overwrite=overwrite)
    metadata.to_csv(metadata_path, index=False)
    return epochs_out, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FieldTrip-like MATLAB raw/trial data to MNE Epochs FIF plus metadata CSV.")
    parser.add_argument("mat", type=Path)
    parser.add_argument("--epochs-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--root-path", default="data,0")
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--label-base", type=float, default=1.0)
    parser.add_argument("--trialinfo-column", type=int, default=0)
    parser.add_argument("--ch-type", default="grad")
    parser.add_argument("--trial-axis-order", choices=("channel_time", "time_channel"), default="channel_time")
    parser.add_argument("--no-trim-overlong-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    epochs_out, metadata_out = write_fieldtrip_raw_mat_epochs(
        args.mat,
        epochs_out=args.epochs_out,
        metadata_out=args.metadata_out,
        root_path=parse_path_tokens(args.root_path, DEFAULT_ROOT_PATH),
        label_column=args.label_column,
        label_base=args.label_base,
        trialinfo_column=args.trialinfo_column,
        ch_type=args.ch_type,
        trial_axis_order=args.trial_axis_order,
        trim_overlong_labels=not args.no_trim_overlong_labels,
        overwrite=args.overwrite,
    )
    print(f"Wrote epochs: {epochs_out}")
    print(f"Wrote metadata: {metadata_out}")


if __name__ == "__main__":
    main()
