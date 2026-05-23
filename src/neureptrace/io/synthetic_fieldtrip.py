"""Private-data-free FieldTrip-style MATLAB fixtures for tests and demos.

The generator intentionally mirrors the small subset of the FieldTrip raw/trial
schema consumed by :mod:`neureptrace.fieldtrip_mat`: ``trial``, ``time``,
``trialinfo``, ``label``, optional ``sampleinfo``, and ``grad`` channel
metadata.  Defaults keep the historical ``Part*Data.mat`` / ``Part*CueData.mat``
file convention useful for PyMEGDec migration tests, while all naming knobs are
configurable so the helper remains dataset-agnostic.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyntheticFieldTripConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for a synthetic FieldTrip-like decoding dataset.

    Labels are stored in ``trialinfo`` as contiguous integer class ids beginning
    at ``label_base``.  This matches many MATLAB-exported stimulus-decoding
    datasets and lets NeuRepTrace's FieldTrip loader create zero-based
    ``condition`` metadata with its default ``label_base=1`` setting.
    """

    participant_id: int | str = 2
    n_classes: int = 16
    main_repeats_per_class: int = 10
    cue_repeats_per_class: int = 5
    n_channels: int = 8
    n_times: int = 261
    tmin: float = -0.5
    tmax: float = 0.8
    stimulus_window: tuple[float, float] = (0.15, 0.25)
    signal_scale: float = 6.0
    noise_scale: float = 0.05
    alpha_scale: float = 0.02
    cue_shift_scale: float = 0.15
    random_seed: int = 13
    label_base: int = 1
    variable_name: str = "data"
    main_file_template: str = "Part{participant}Data.mat"
    cue_file_template: str | None = "Part{participant}CueData.mat"
    manifest_name: str | None = "synthetic_fieldtrip_manifest.json"
    include_sampleinfo: bool = True


@dataclass(frozen=True)
class SyntheticFieldTripOutput:
    """Paths and dimensions written by :func:`write_synthetic_fieldtrip_dataset`."""

    data_dir: Path
    participant_id: str
    main_path: Path
    cue_path: Path | None
    manifest_path: Path | None
    main_trials: int
    cue_trials: int
    n_classes: int
    n_channels: int
    n_times: int
    variable_name: str


# Compatibility aliases for migrated PyMEGDec code/tests.
SyntheticDataConfig = SyntheticFieldTripConfig
SyntheticDataOutput = SyntheticFieldTripOutput


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_config(config: SyntheticFieldTripConfig) -> None:
    _require(str(config.participant_id).strip() != "", "participant_id must not be empty.")
    _require(config.n_classes >= 2, "n_classes must be at least 2 for decoding demos.")
    _require(config.main_repeats_per_class >= 1, "main_repeats_per_class must be at least 1.")
    _require(config.cue_repeats_per_class >= 0, "cue_repeats_per_class must be non-negative.")
    _require(config.n_channels >= 1, "n_channels must be at least 1.")
    _require(config.n_times >= 3, "n_times must be at least 3.")
    _require(config.label_base >= 0, "label_base must be non-negative.")
    _require(config.variable_name.strip() != "", "variable_name must not be empty.")
    _require("{participant}" in config.main_file_template, "main_file_template must contain '{participant}'.")
    if config.cue_file_template is not None:
        _require("{participant}" in config.cue_file_template, "cue_file_template must contain '{participant}'.")

    window_start, window_stop = config.stimulus_window
    _require(config.tmin < config.tmax, "tmin must be smaller than tmax.")
    _require(window_start < window_stop, "stimulus_window start must be smaller than stop.")
    _require(
        window_stop >= config.tmin and window_start <= config.tmax,
        "stimulus_window must overlap the generated time vector.",
    )
    for field_name, minimum, message in (
        ("signal_scale", np.finfo(float).tiny, "signal_scale must be positive."),
        ("noise_scale", 0.0, "noise_scale must be non-negative."),
        ("alpha_scale", 0.0, "alpha_scale must be non-negative."),
        ("cue_shift_scale", 0.0, "cue_shift_scale must be non-negative."),
    ):
        _require(getattr(config, field_name) >= minimum, message)


def _balanced_labels(config: SyntheticFieldTripConfig, repeats_per_class: int) -> np.ndarray:
    """Return cyclic class labels with contiguous-fold friendly ordering."""

    class_ids = np.arange(config.label_base, config.label_base + config.n_classes, dtype=int)
    return np.tile(class_ids, repeats_per_class)


def _class_prototypes(rng: np.random.Generator, n_classes: int, n_channels: int) -> np.ndarray:
    prototypes = rng.normal(size=(n_classes, n_channels))
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes / np.maximum(norms, np.finfo(float).eps)


def _cell_column(values: Sequence[object]) -> np.ndarray:
    cell = np.empty((len(values), 1), dtype=object)
    for index, value in enumerate(values):
        cell[index, 0] = value
    return cell


def _cell_row(values: Sequence[object]) -> np.ndarray:
    cell = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        cell[0, index] = value
    return cell


def _channel_names(n_channels: int) -> list[str]:
    return [f"MEG{index + 1:03d}" for index in range(n_channels)]


def _channel_positions(n_channels: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_channels, endpoint=False)
    radius_mm = 80.0
    return np.column_stack(
        [
            radius_mm * np.cos(angles),
            radius_mm * np.sin(angles),
            20.0 * np.sin(2.0 * angles),
        ]
    )


def _sampleinfo(n_trials: int, n_times: int) -> np.ndarray:
    starts = np.arange(n_trials, dtype=int) * n_times + 1
    stops = starts + n_times - 1
    return np.column_stack([starts, stops])


def _synthetic_fieldtrip_data(
    labels: np.ndarray,
    *,
    time_vector: np.ndarray,
    prototypes: np.ndarray,
    rng: np.random.Generator,
    config: SyntheticFieldTripConfig,
    cue: bool = False,
) -> dict[str, object]:
    stimulus_mask = (time_vector >= config.stimulus_window[0]) & (time_vector <= config.stimulus_window[1])
    channel_phase = np.linspace(0.0, np.pi, config.n_channels, endpoint=False)[:, None]
    alpha_carrier = np.sin(2.0 * np.pi * 10.0 * time_vector[None, :] + channel_phase)
    cue_shift = rng.normal(scale=config.cue_shift_scale, size=(config.n_channels, 1)) if cue else 0.0

    trials: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for trial_index, label in enumerate(labels):
        trial = rng.normal(scale=config.noise_scale, size=(config.n_channels, config.n_times))
        if config.alpha_scale:
            trial += config.alpha_scale * alpha_carrier
        class_index = int(label) - config.label_base
        trial[:, stimulus_mask] += config.signal_scale * prototypes[class_index][:, None] + cue_shift
        # Tiny deterministic offset makes accidental trial reordering visible
        # without overwhelming the class-informative pattern.
        trial += 1e-4 * (trial_index + 1)
        trials.append(trial.astype(float, copy=False))
        times.append(time_vector[None, :])

    channel_names = _channel_names(config.n_channels)
    data: dict[str, object] = {
        "trial": _cell_row(trials),
        "time": _cell_row(times),
        "trialinfo": labels.reshape(-1, 1),
        "label": _cell_column(channel_names),
        "grad": {
            "label": _cell_column(channel_names),
            "chantype": _cell_column(["meggrad"] * config.n_channels),
            "chanunit": _cell_column(["T/m"] * config.n_channels),
            "chanpos": _channel_positions(config.n_channels),
            "coordsys": "synthetic",
        },
    }
    if config.include_sampleinfo:
        data["sampleinfo"] = _sampleinfo(labels.size, config.n_times)
    return data


def _format_participant_file(template: str, participant_id: int | str) -> str:
    return template.format(participant=participant_id)


def _manifest(output: SyntheticFieldTripOutput, config: SyntheticFieldTripConfig) -> dict[str, object]:
    return {
        "participant_id": output.participant_id,
        "variable_name": output.variable_name,
        "main_file": output.main_path.name,
        "cue_file": output.cue_path.name if output.cue_path is not None else None,
        "main_trials": output.main_trials,
        "cue_trials": output.cue_trials,
        "n_classes": output.n_classes,
        "n_channels": output.n_channels,
        "n_times": output.n_times,
        "config": asdict(config),
    }


def write_synthetic_fieldtrip_dataset(
    data_dir: str | Path,
    config: SyntheticFieldTripConfig | None = None,
    *,
    overwrite: bool = False,
    write_manifest: bool = True,
) -> SyntheticFieldTripOutput:
    """Write synthetic FieldTrip-like ``.mat`` files.

    Parameters
    ----------
    data_dir:
        Directory that receives generated MATLAB files.
    config:
        Dataset parameters.  Defaults create a private-data-free fixture with
        the old PyMEGDec/Bush participant filename convention.
    overwrite:
        Replace existing output files when true.
    write_manifest:
        Write a JSON manifest when ``config.manifest_name`` is not ``None``.
    """

    try:
        import scipy.io as sio
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError("Writing synthetic FieldTrip MATLAB files requires scipy.") from exc

    config = config or SyntheticFieldTripConfig()
    _validate_config(config)

    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    participant_id = str(config.participant_id)
    main_path = output_dir / _format_participant_file(config.main_file_template, participant_id)
    cue_path = (
        output_dir / _format_participant_file(config.cue_file_template, participant_id)
        if config.cue_file_template is not None and config.cue_repeats_per_class > 0
        else None
    )
    manifest_path = output_dir / config.manifest_name if write_manifest and config.manifest_name else None

    existing = [path for path in (main_path, cue_path, manifest_path) if path is not None and path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Synthetic FieldTrip output already exists: {names}. Pass overwrite=True to replace it.")

    rng = np.random.default_rng(config.random_seed)
    prototypes = _class_prototypes(rng, config.n_classes, config.n_channels)
    time_vector = np.linspace(config.tmin, config.tmax, config.n_times)
    main_labels = _balanced_labels(config, config.main_repeats_per_class)
    cue_labels = _balanced_labels(config, config.cue_repeats_per_class) if cue_path is not None else np.asarray([], dtype=int)

    sio.savemat(
        main_path,
        {config.variable_name: _synthetic_fieldtrip_data(main_labels, time_vector=time_vector, prototypes=prototypes, rng=rng, config=config)},
    )
    if cue_path is not None:
        sio.savemat(
            cue_path,
            {
                config.variable_name: _synthetic_fieldtrip_data(
                    cue_labels,
                    time_vector=time_vector,
                    prototypes=prototypes,
                    rng=rng,
                    config=config,
                    cue=True,
                )
            },
        )

    output = SyntheticFieldTripOutput(
        data_dir=output_dir,
        participant_id=participant_id,
        main_path=main_path,
        cue_path=cue_path,
        manifest_path=manifest_path,
        main_trials=int(main_labels.size),
        cue_trials=int(cue_labels.size),
        n_classes=config.n_classes,
        n_channels=config.n_channels,
        n_times=config.n_times,
        variable_name=config.variable_name,
    )
    if manifest_path is not None:
        manifest_path.write_text(json.dumps(_manifest(output, config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


write_synthetic_dataset = write_synthetic_fieldtrip_dataset


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_float_range(value: str) -> tuple[float, float]:
    values = _parse_float_list(value)
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated values: start,stop.")
    return values


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Build the synthetic FieldTrip generator parser."""

    defaults = SyntheticFieldTripConfig()
    parser = argparse.ArgumentParser(prog=prog, description="Create private-data-free FieldTrip-style MATLAB demo files.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", required=True, help="Output directory for generated MAT files.")
    parser.add_argument("--participant", default=defaults.participant_id, help="Participant id used in output file names.")
    parser.add_argument("--classes", type=int, default=defaults.n_classes, help="Number of stimulus classes.")
    parser.add_argument("--main-repeats", type=int, default=defaults.main_repeats_per_class, help="Main-experiment repeats per class.")
    parser.add_argument("--cue-repeats", type=int, default=defaults.cue_repeats_per_class, help="Cue/control repeats per class.")
    parser.add_argument("--no-cue", action="store_true", help="Do not write a cue/control MAT file.")
    parser.add_argument("--channels", type=int, default=defaults.n_channels, help="Number of channels.")
    parser.add_argument("--times", type=int, default=defaults.n_times, help="Number of time samples per trial.")
    parser.add_argument("--tmin", type=float, default=defaults.tmin, help="First sample time in seconds.")
    parser.add_argument("--tmax", type=float, default=defaults.tmax, help="Last sample time in seconds.")
    parser.add_argument(
        "--stimulus-window",
        type=_parse_float_range,
        default=defaults.stimulus_window,
        help="Class-informative window as start,stop in seconds.",
    )
    parser.add_argument("--signal-scale", type=float, default=defaults.signal_scale, help="Class-pattern amplitude in the stimulus window.")
    parser.add_argument("--noise-scale", type=float, default=defaults.noise_scale, help="Gaussian observation noise scale.")
    parser.add_argument("--alpha-scale", type=float, default=defaults.alpha_scale, help="Background 10 Hz carrier amplitude.")
    parser.add_argument("--cue-shift-scale", type=float, default=defaults.cue_shift_scale, help="Small cue-domain shift scale.")
    parser.add_argument("--seed", type=int, default=defaults.random_seed, help="Random seed for reproducible data.")
    parser.add_argument("--label-base", type=int, default=defaults.label_base, help="First integer class label stored in trialinfo.")
    parser.add_argument("--variable", default=defaults.variable_name, help="Top-level MATLAB variable name.")
    parser.add_argument("--main-template", default=defaults.main_file_template, help="Main MAT filename template containing {participant}.")
    parser.add_argument("--cue-template", default=defaults.cue_file_template, help="Cue MAT filename template containing {participant}.")
    parser.add_argument("--manifest-name", default=defaults.manifest_name, help="Manifest JSON filename.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated files.")
    parser.add_argument("--no-manifest", action="store_true", help="Do not write a JSON manifest.")
    return parser


def make_synthetic_fieldtrip_data(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Generate private-data-free FieldTrip-style MATLAB demo files."""

    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    config = SyntheticFieldTripConfig(
        participant_id=args.participant,
        n_classes=args.classes,
        main_repeats_per_class=args.main_repeats,
        cue_repeats_per_class=args.cue_repeats,
        n_channels=args.channels,
        n_times=args.times,
        tmin=args.tmin,
        tmax=args.tmax,
        stimulus_window=args.stimulus_window,
        signal_scale=args.signal_scale,
        noise_scale=args.noise_scale,
        alpha_scale=args.alpha_scale,
        cue_shift_scale=args.cue_shift_scale,
        random_seed=args.seed,
        label_base=args.label_base,
        variable_name=args.variable,
        main_file_template=args.main_template,
        cue_file_template=None if args.no_cue else args.cue_template,
        manifest_name=args.manifest_name,
    )
    output = write_synthetic_fieldtrip_dataset(
        args.out_dir,
        config,
        overwrite=args.overwrite,
        write_manifest=not args.no_manifest,
    )
    print(f"Wrote main data: {output.main_path}")
    if output.cue_path is not None:
        print(f"Wrote cue data: {output.cue_path}")
    if output.manifest_path is not None:
        print(f"Wrote manifest: {output.manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``neureptrace synthetic-fieldtrip``."""

    return make_synthetic_fieldtrip_data(argv)


if __name__ == "__main__":
    raise SystemExit(main())
