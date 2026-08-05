"""Build event-level Katja Button Press MEG features from SPM files.

The builder streams one subject and one trial at a time from the SPM ``.dat``
files. It never materializes the full raw dataset. The resulting NPZ cache is
consumed by :mod:`neureptrace.katja_finger_sequence_benchmark`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
from scipy.io import loadmat

from neureptrace._katja_finger_sequence_support import DEFAULT_PARTICIPANTS

DEFAULT_WINDOW_CENTERS_SECONDS = (-0.15, -0.05, 0.05, 0.15)
DEFAULT_WINDOW_WIDTH_SECONDS = 0.10
DEFAULT_BASELINE_SECONDS = (-0.35, -0.05)
DEFAULT_BUTTON_LAG_SECONDS = 0.03


@dataclass(frozen=True, slots=True)
class SPMContinuousHeader:
    """Minimal SPM M/EEG header needed for memory-mapped trial access."""

    header_path: Path
    data_path: Path
    shape: tuple[int, int, int]
    dtype: np.dtype
    offset: int
    sampling_frequency: float
    time_onset: float
    channel_labels: tuple[str, ...]
    channel_types: tuple[str, ...]
    channel_bad: np.ndarray


@dataclass(frozen=True, slots=True)
class SubjectFeatureRows:
    """Feature rows and event metadata for one participant."""

    features: np.ndarray
    subjects: np.ndarray
    trial_ids: np.ndarray
    press_positions: np.ndarray
    sequence_ids: np.ndarray
    finger_codes: np.ndarray
    event_times_seconds: np.ndarray
    n_trials_considered: int
    n_trials_correct_order: int
    n_trials_retained: int
    n_trials_dropped_out_of_bounds: int
    n_trials_dropped_nonfinite: int


def _as_text(value: Any) -> str:
    """Convert MATLAB/scipy scalar text representations to a clean string."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8", errors="replace")
        return str(scalar)
    if array.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in array.reshape(-1).tolist())
    return str(value)


def _as_bool(value: Any) -> bool:
    array = np.asarray(value)
    if array.size == 0:
        return False
    return bool(array.reshape(-1)[0])


def _spm_dtype(dtype_code: int, *, big_endian: bool) -> np.dtype:
    """Translate common SPM file-array datatype codes to NumPy dtypes."""

    dtypes = {
        2: np.dtype("u1"),
        4: np.dtype("i2"),
        8: np.dtype("i4"),
        16: np.dtype("f4"),
        64: np.dtype("f8"),
        256: np.dtype("i1"),
        512: np.dtype("u2"),
        768: np.dtype("u4"),
    }
    try:
        dtype = dtypes[int(dtype_code)]
    except KeyError as exc:
        raise ValueError(f"Unsupported SPM data type code {dtype_code!r}.") from exc
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder(">" if big_endian else "<")
    return dtype


def _find_spm_header(subject_directory: Path) -> Path:
    candidates = sorted(
        path for path in subject_directory.glob("fc*.mat") if not path.name.startswith("Bfc")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one sensor-level fc*.mat file in {subject_directory}; "
            f"found {len(candidates)}."
        )
    return candidates[0]


def load_spm_continuous_header(subject_directory: str | Path) -> SPMContinuousHeader:
    """Load the lightweight SPM header while leaving the multi-GB data file mapped."""

    directory = Path(subject_directory)
    header_path = _find_spm_header(directory)
    loaded = loadmat(header_path, squeeze_me=True, struct_as_record=False)
    if "D" not in loaded:
        raise ValueError(f"SPM header {header_path} does not contain structure D.")
    header = loaded["D"]
    data = header.data
    dimensions = tuple(int(value) for value in np.asarray(data.dim).reshape(-1))
    if len(dimensions) != 3 or min(dimensions) < 1:
        raise ValueError(f"Unexpected SPM data dimensions {dimensions!r} in {header_path}.")
    data_path = header_path.with_suffix(".dat")
    if not data_path.exists():
        stored_name = Path(_as_text(getattr(data, "fname", ""))).name
        alternative = header_path.parent / stored_name
        if alternative.exists():
            data_path = alternative
        else:
            raise FileNotFoundError(data_path)

    channels = np.atleast_1d(header.channels).reshape(-1)
    if channels.shape[0] != dimensions[0]:
        raise ValueError(
            f"SPM channel count {channels.shape[0]} does not match data shape "
            f"{dimensions[0]} in {header_path}."
        )
    labels = tuple(_as_text(getattr(channel, "label", "")) for channel in channels)
    types = tuple(_as_text(getattr(channel, "type", "")) for channel in channels)
    bad = np.asarray(
        [_as_bool(getattr(channel, "bad", False)) for channel in channels],
        dtype=bool,
    )

    dtype = _spm_dtype(
        int(np.asarray(data.dtype).reshape(-1)[0]),
        big_endian=_as_bool(getattr(data, "be", False)),
    )
    offset = int(np.asarray(getattr(data, "offset", 0)).reshape(-1)[0])
    expected_bytes = int(np.prod(dimensions, dtype=np.int64)) * dtype.itemsize + offset
    actual_bytes = data_path.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"SPM data file {data_path} is too small: {actual_bytes} bytes, "
            f"expected at least {expected_bytes}."
        )

    return SPMContinuousHeader(
        header_path=header_path,
        data_path=data_path,
        shape=dimensions,
        dtype=dtype,
        offset=offset,
        sampling_frequency=float(header.Fsample),
        time_onset=float(header.timeOnset),
        channel_labels=labels,
        channel_types=types,
        channel_bad=bad,
    )


def good_meggrad_labels(header: SPMContinuousHeader) -> tuple[str, ...]:
    """Return unique non-bad MEGGRAD labels in acquisition order."""

    selected: list[str] = []
    seen: set[str] = set()
    for label, channel_type, bad in zip(
        header.channel_labels,
        header.channel_types,
        header.channel_bad.tolist(),
        strict=True,
    ):
        if channel_type.strip().upper() != "MEGGRAD" or bad:
            continue
        if not label or label in seen:
            raise ValueError(
                f"Invalid or duplicate good MEGGRAD label {label!r} in {header.header_path}."
            )
        seen.add(label)
        selected.append(label)
    if not selected:
        raise ValueError(f"No good MEGGRAD channels found in {header.header_path}.")
    return tuple(selected)


def common_good_meggrad_labels(
    headers: Sequence[SPMContinuousHeader],
) -> tuple[str, ...]:
    """Find a stable cross-participant intersection of good MEGGRAD labels."""

    if not headers:
        raise ValueError("At least one SPM header is required.")
    per_header = [set(good_meggrad_labels(header)) for header in headers]
    common = set.intersection(*per_header)
    ordered = tuple(
        label for label in good_meggrad_labels(headers[0]) if label in common
    )
    if not ordered:
        raise ValueError("Participants have no common good MEGGRAD channels.")
    return ordered


def _hdf5_group_field(group: h5py.Group, *aliases: str) -> h5py.Dataset:
    by_lower = {name.lower(): name for name in group.keys()}
    for alias in aliases:
        actual = by_lower.get(alias.lower())
        if actual is not None:
            item = group[actual]
            if not isinstance(item, h5py.Dataset):
                raise ValueError(f"MATLAB field {actual!r} is not a numeric dataset.")
            return item
    raise ValueError(f"MATLAB structure is missing field; tried {', '.join(aliases)}.")


def _read_numeric_field(group: h5py.Group, *aliases: str) -> np.ndarray:
    dataset = _hdf5_group_field(group, *aliases)
    if h5py.check_dtype(ref=dataset.dtype) is not None:
        raise ValueError(
            f"MATLAB field {dataset.name!r} unexpectedly contains object references."
        )
    return np.asarray(dataset)


def _orient_trial_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    array = np.squeeze(array)
    if array.ndim != 1:
        raise ValueError(
            f"Behavior field {name!r} must be a trial vector; got shape {array.shape}."
        )
    return array


def _orient_trial_matrix(
    values: np.ndarray,
    *,
    n_trials: int,
    name: str,
    n_events: int = 5,
) -> np.ndarray:
    array = np.asarray(values)
    array = np.squeeze(array)
    if array.shape == (n_events, n_trials):
        array = array.T
    elif array.shape != (n_trials, n_events):
        if array.size != n_trials * n_events:
            raise ValueError(
                f"Behavior field {name!r} must have {n_trials} x {n_events} values; "
                f"got shape {array.shape}."
            )
        array = np.asarray(array).reshape((n_events, n_trials), order="F").T
    return array


def load_subject_behavior(behavior_directory: str | Path) -> dict[str, np.ndarray]:
    """Load the behavioral fields used for correct-order response locking."""

    directory = Path(behavior_directory)
    behavior_path = directory / "behDataMEG.mat"
    target_path = directory / "targetMEG.mat"
    with h5py.File(behavior_path, "r") as archive:
        if "A" not in archive or not isinstance(archive["A"], h5py.Group):
            raise ValueError(f"{behavior_path} does not contain MATLAB structure A.")
        group = archive["A"]
        points = _orient_trial_vector(
            _read_numeric_field(group, "points"), name="points"
        )
        n_trials = int(points.shape[0])
        sequence_ids = _orient_trial_vector(
            _read_numeric_field(group, "seqID", "sequenceID"),
            name="seqID",
        )
        npress = _orient_trial_vector(
            _read_numeric_field(group, "npress"), name="npress"
        )
        press = _orient_trial_matrix(
            _read_numeric_field(group, "press"),
            n_trials=n_trials,
            name="press",
        )
        cue_finger = _orient_trial_matrix(
            _read_numeric_field(group, "cueFinger", "cuefinger"),
            n_trials=n_trials,
            name="cueFinger",
        )
        timing_ms = _orient_trial_matrix(
            _read_numeric_field(group, "timing"),
            n_trials=n_trials,
            name="timing",
        )

    with h5py.File(target_path, "r") as archive:
        if "T" not in archive or not isinstance(archive["T"], h5py.Group):
            raise ValueError(f"{target_path} does not contain MATLAB structure T.")
        cue_duration_ms = _orient_trial_vector(
            _read_numeric_field(archive["T"], "cueDur", "cueDuration"),
            name="cueDur",
        )

    vectors = {
        "points": points,
        "sequence_ids": sequence_ids,
        "npress": npress,
        "cue_duration_ms": cue_duration_ms,
    }
    for name, vector in vectors.items():
        if vector.shape[0] != n_trials:
            raise ValueError(
                f"Behavior field {name!r} has {vector.shape[0]} trials; expected {n_trials}."
            )
    return {
        **vectors,
        "press": press,
        "cue_finger": cue_finger,
        "timing_ms": timing_ms,
    }


def _channel_indices(
    header: SPMContinuousHeader, labels: Sequence[str]
) -> np.ndarray:
    by_label = {label: index for index, label in enumerate(header.channel_labels)}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise ValueError(
            f"SPM header {header.header_path} is missing {len(missing)} selected channels; "
            f"first missing label: {missing[0]!r}."
        )
    return np.asarray([by_label[label] for label in labels], dtype=int)


def _sample_bounds(
    *,
    start_seconds: float,
    stop_seconds: float,
    time_onset: float,
    sampling_frequency: float,
    n_samples: int,
) -> tuple[int, int] | None:
    start = int(round((start_seconds - time_onset) * sampling_frequency))
    stop = int(round((stop_seconds - time_onset) * sampling_frequency))
    if start < 0 or stop > n_samples or stop <= start:
        return None
    return start, stop


def _baseline_scale(baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(baseline, axis=1)
    scale = np.nanstd(baseline, axis=1)
    positive = scale[np.isfinite(scale) & (scale > 0.0)]
    if positive.size == 0:
        raise ValueError(
            "Trial baseline has no positive finite channel standard deviations."
        )
    floor = max(
        float(np.median(positive)) * 1e-6,
        np.finfo(np.float64).tiny,
    )
    scale = np.maximum(scale, floor)
    return mean, scale


def extract_subject_feature_rows(
    *,
    subject: str,
    header: SPMContinuousHeader,
    behavior: dict[str, np.ndarray],
    selected_channel_labels: Sequence[str],
    window_centers_seconds: Sequence[float] = DEFAULT_WINDOW_CENTERS_SECONDS,
    window_width_seconds: float = DEFAULT_WINDOW_WIDTH_SECONDS,
    baseline_seconds: tuple[float, float] = DEFAULT_BASELINE_SECONDS,
    button_lag_seconds: float = DEFAULT_BUTTON_LAG_SECONDS,
) -> SubjectFeatureRows:
    """Extract four response-locked window means for presses 2--5."""

    if window_width_seconds <= 0.0 or not np.isfinite(window_width_seconds):
        raise ValueError("window_width_seconds must be positive and finite.")
    centers = tuple(float(value) for value in window_centers_seconds)
    if not centers or not np.all(np.isfinite(centers)):
        raise ValueError(
            "window_centers_seconds must be a non-empty finite sequence."
        )
    if not (
        np.isfinite(baseline_seconds[0])
        and np.isfinite(baseline_seconds[1])
        and baseline_seconds[0] < baseline_seconds[1]
    ):
        raise ValueError("baseline_seconds must be a finite increasing interval.")
    if not np.isfinite(button_lag_seconds):
        raise ValueError("button_lag_seconds must be finite.")

    n_trials = min(
        int(header.shape[2]),
        *(int(np.asarray(value).shape[0]) for value in behavior.values()),
    )
    channel_indices = _channel_indices(header, selected_channel_labels)
    data = np.memmap(
        header.data_path,
        dtype=header.dtype,
        mode="r",
        offset=header.offset,
        shape=header.shape,
        order="F",
    )
    baseline_bounds = _sample_bounds(
        start_seconds=baseline_seconds[0],
        stop_seconds=baseline_seconds[1],
        time_onset=header.time_onset,
        sampling_frequency=header.sampling_frequency,
        n_samples=header.shape[1],
    )
    if baseline_bounds is None:
        raise ValueError(
            f"Baseline interval {baseline_seconds!r} is outside the epoch in "
            f"{header.header_path}."
        )

    feature_rows: list[np.ndarray] = []
    trial_ids: list[int] = []
    press_positions: list[int] = []
    sequence_ids: list[Any] = []
    finger_codes: list[Any] = []
    event_times: list[float] = []
    correct_order_count = 0
    retained_trials = 0
    dropped_out_of_bounds = 0
    dropped_nonfinite = 0

    press = np.asarray(behavior["press"])
    cue_finger = np.asarray(behavior["cue_finger"])
    timing_ms = np.asarray(behavior["timing_ms"], dtype=float)
    cue_duration_ms = np.asarray(behavior["cue_duration_ms"], dtype=float)
    points = np.asarray(behavior["points"], dtype=float)
    npress = np.asarray(behavior["npress"], dtype=float)
    sequence = np.asarray(behavior["sequence_ids"])

    baseline_start, baseline_stop = baseline_bounds
    half_width = float(window_width_seconds) / 2.0
    for trial_index in range(n_trials):
        recorded = press[trial_index, :5]
        instructed = cue_finger[trial_index, :5]
        correct_order = bool(
            points[trial_index] >= 1
            and npress[trial_index] >= 5
            and np.all(np.isfinite(np.asarray(recorded, dtype=float)))
            and np.all(np.isfinite(timing_ms[trial_index, :5]))
            and np.array_equal(recorded, instructed)
        )
        if not correct_order:
            continue
        correct_order_count += 1

        variable_codes = recorded[1:5]
        if np.unique(variable_codes).shape[0] != 4:
            raise ValueError(
                f"Participant {subject}, trial {trial_index + 1} does not contain four "
                "unique variable finger codes."
            )
        baseline = np.asarray(
            data[channel_indices, baseline_start:baseline_stop, trial_index],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(baseline)):
            dropped_nonfinite += 1
            continue
        try:
            baseline_mean, baseline_scale = _baseline_scale(baseline)
        except ValueError:
            dropped_nonfinite += 1
            continue

        trial_features: list[np.ndarray] = []
        trial_event_times: list[float] = []
        trial_valid = True
        trial_nonfinite = False
        for press_index in range(1, 5):
            event_seconds = (
                cue_duration_ms[trial_index]
                + timing_ms[trial_index, press_index]
            ) / 1000.0 + float(button_lag_seconds)
            vectors: list[np.ndarray] = []
            for center in centers:
                bounds = _sample_bounds(
                    start_seconds=event_seconds + center - half_width,
                    stop_seconds=event_seconds + center + half_width,
                    time_onset=header.time_onset,
                    sampling_frequency=header.sampling_frequency,
                    n_samples=header.shape[1],
                )
                if bounds is None:
                    trial_valid = False
                    break
                start, stop = bounds
                window = np.asarray(
                    data[channel_indices, start:stop, trial_index],
                    dtype=np.float64,
                )
                window_mean = np.nanmean(window, axis=1)
                normalized = (window_mean - baseline_mean) / baseline_scale
                if not np.all(np.isfinite(normalized)):
                    trial_valid = False
                    trial_nonfinite = True
                    break
                vectors.append(normalized.astype(np.float32, copy=False))
            if not trial_valid:
                break
            trial_features.append(
                np.concatenate(vectors).astype(np.float32, copy=False)
            )
            trial_event_times.append(event_seconds)

        if not trial_valid:
            if trial_nonfinite:
                dropped_nonfinite += 1
            else:
                dropped_out_of_bounds += 1
            continue

        retained_trials += 1
        for local_index, feature_vector in enumerate(trial_features):
            press_index = local_index + 1
            feature_rows.append(feature_vector)
            trial_ids.append(trial_index + 1)
            press_positions.append(press_index + 1)
            sequence_ids.append(sequence[trial_index])
            finger_codes.append(recorded[press_index])
            event_times.append(trial_event_times[local_index])

    if not feature_rows:
        raise ValueError(
            f"No valid correct-order feature rows were extracted for {subject}."
        )
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    return SubjectFeatureRows(
        features=features,
        subjects=np.asarray([subject] * features.shape[0]),
        trial_ids=np.asarray(trial_ids, dtype=int),
        press_positions=np.asarray(press_positions, dtype=int),
        sequence_ids=np.asarray(sequence_ids),
        finger_codes=np.asarray(finger_codes),
        event_times_seconds=np.asarray(event_times, dtype=np.float64),
        n_trials_considered=n_trials,
        n_trials_correct_order=correct_order_count,
        n_trials_retained=retained_trials,
        n_trials_dropped_out_of_bounds=dropped_out_of_bounds,
        n_trials_dropped_nonfinite=dropped_nonfinite,
    )


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("Comma-separated participant list must not be empty.")
    return result


def _concatenate(rows: Iterable[SubjectFeatureRows], field: str) -> np.ndarray:
    values = [np.asarray(getattr(row, field)) for row in rows]
    return np.concatenate(values, axis=0)


def build_katja_spm_feature_cache(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    participants: Sequence[str] = DEFAULT_PARTICIPANTS,
    window_centers_seconds: Sequence[float] = DEFAULT_WINDOW_CENTERS_SECONDS,
    window_width_seconds: float = DEFAULT_WINDOW_WIDTH_SECONDS,
    baseline_seconds: tuple[float, float] = DEFAULT_BASELINE_SECONDS,
    button_lag_seconds: float = DEFAULT_BUTTON_LAG_SECONDS,
) -> dict[str, Any]:
    """Build and save the complete cross-participant event-row feature cache."""

    root = Path(dataset_root)
    participant_tuple = tuple(str(value) for value in participants)
    if not participant_tuple or len(set(participant_tuple)) != len(participant_tuple):
        raise ValueError("participants must contain unique identifiers.")
    headers = [
        load_spm_continuous_header(root / subject) for subject in participant_tuple
    ]
    selected_labels = common_good_meggrad_labels(headers)

    subject_rows: list[SubjectFeatureRows] = []
    subject_summary: dict[str, Any] = {}
    for subject, header in zip(participant_tuple, headers, strict=True):
        behavior = load_subject_behavior(root / "beh_data" / subject)
        rows = extract_subject_feature_rows(
            subject=subject,
            header=header,
            behavior=behavior,
            selected_channel_labels=selected_labels,
            window_centers_seconds=window_centers_seconds,
            window_width_seconds=window_width_seconds,
            baseline_seconds=baseline_seconds,
            button_lag_seconds=button_lag_seconds,
        )
        subject_rows.append(rows)
        subject_summary[subject] = {
            "spm_header": str(header.header_path),
            "spm_data": str(header.data_path),
            "n_channels_total": int(header.shape[0]),
            "n_good_meggrad": len(good_meggrad_labels(header)),
            "n_trials_considered": rows.n_trials_considered,
            "n_trials_correct_order": rows.n_trials_correct_order,
            "n_trials_retained": rows.n_trials_retained,
            "n_trials_dropped_out_of_bounds": rows.n_trials_dropped_out_of_bounds,
            "n_trials_dropped_nonfinite": rows.n_trials_dropped_nonfinite,
        }

    feature_widths = {row.features.shape[1] for row in subject_rows}
    if len(feature_widths) != 1:
        raise ValueError(
            f"Subject feature widths differ unexpectedly: {sorted(feature_widths)}."
        )
    metadata = {
        "format": "neureptrace_katja_button_press_event_features_v1",
        "participants": list(participant_tuple),
        "selected_channel_policy": (
            "intersection_of_good_MEGGRAD_across_participants"
        ),
        "selected_channel_count": len(selected_labels),
        "selected_channel_labels": list(selected_labels),
        "window_centers_seconds": [
            float(value) for value in window_centers_seconds
        ],
        "window_width_seconds": float(window_width_seconds),
        "baseline_seconds": [float(value) for value in baseline_seconds],
        "baseline_normalization": "per_trial_per_channel_z_score",
        "button_lag_seconds": float(button_lag_seconds),
        "event_positions": [2, 3, 4, 5],
        "correct_order_rule": (
            "points>=1, npress>=5, and recorded press equals cueFinger"
        ),
        "subject_summary": subject_summary,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=_concatenate(subject_rows, "features"),
        subjects=_concatenate(subject_rows, "subjects"),
        trial_ids=_concatenate(subject_rows, "trial_ids"),
        press_positions=_concatenate(subject_rows, "press_positions"),
        sequence_ids=_concatenate(subject_rows, "sequence_ids"),
        finger_codes=_concatenate(subject_rows, "finger_codes"),
        event_times_seconds=_concatenate(
            subject_rows, "event_times_seconds"
        ),
        correct_order=np.ones(
            sum(row.features.shape[0] for row in subject_rows), dtype=bool
        ),
        channel_labels=np.asarray(selected_labels),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    metadata["output_path"] = str(output)
    metadata["n_event_rows"] = int(
        sum(row.features.shape[0] for row in subject_rows)
    )
    metadata["feature_width"] = int(next(iter(feature_widths)))
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--button-lag-ms", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Reusing existing feature cache: {output}")
        with np.load(output, allow_pickle=False) as cache:
            print(
                json.dumps(
                    {
                        "features_shape": list(cache["features"].shape),
                        "participants": sorted(
                            set(cache["subjects"].astype(str).tolist())
                        ),
                    },
                    indent=2,
                )
            )
        return 0
    metadata = build_katja_spm_feature_cache(
        args.dataset_root,
        output,
        participants=_parse_csv(args.participants),
        button_lag_seconds=float(args.button_lag_ms) / 1000.0,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
