"""Config-driven dataset workflow entry points.

This module is intentionally small: it makes NeuRepTrace callable from a
declarative JSON/YAML file without moving dataset-specific conventions into the
core decoding code.  It is the bridge needed to retire thin dataset adapter
packages such as PyMEGDec while keeping the numerical workflows in NeuRepTrace.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd

from neureptrace.mne_time_decode import run_time_resolved_decode

ConfigDict = dict[str, Any]


class DatasetConfigError(ValueError):
    """Raised when a dataset configuration is malformed."""


def _read_yaml(path: Path) -> ConfigDict:
    """Read YAML using PyYAML only when the user selected a YAML file."""

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise DatasetConfigError(
            "YAML configuration files require PyYAML. Install it or use JSON."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise DatasetConfigError("The top-level YAML document must be a mapping.")
    return dict(payload)


def load_dataset_config(path: Path) -> ConfigDict:
    """Load a NeuRepTrace dataset workflow config from JSON or YAML."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif suffix in {".yml", ".yaml"}:
        payload = _read_yaml(path)
    else:
        raise DatasetConfigError(
            f"Unsupported config extension '{path.suffix}'. Use .json, .yml, or .yaml."
        )

    if not isinstance(payload, dict):
        raise DatasetConfigError("The top-level config document must be a mapping.")
    return dict(payload)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DatasetConfigError(f"Config section '{name}' must be a mapping.")
    return value


def _first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _resolve_path(value: str | Path | None, *, config_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    return path if path.is_absolute() else config_dir / path


def _require_path(value: str | Path | None, *, config_dir: Path, name: str) -> Path:
    path = _resolve_path(value, config_dir=config_dir)
    if path is None:
        raise DatasetConfigError(f"Missing required path '{name}'.")
    return path


def _as_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise DatasetConfigError(f"'{name}' must contain finite numeric values.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetConfigError(f"'{name}' must contain finite numeric values.") from exc
    if not math.isfinite(parsed):
        raise DatasetConfigError(f"'{name}' must contain finite numeric values.")
    return parsed


def _as_float_pair(value: Any, *, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise DatasetConfigError(f"'{name}' must contain exactly two numeric values.")
    start, stop = (_as_finite_float(item, name=name) for item in value)
    return start, stop


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        if int(value) in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise DatasetConfigError(f"Cannot interpret {value!r} as a boolean.")


def _decode_kwargs(config: Mapping[str, Any], *, config_path: Path) -> dict[str, Any]:
    """Translate config sections to ``run_time_resolved_decode`` keyword args."""

    config_dir = config_path.parent
    dataset = _section(config, "dataset")
    decoding = _section(config, "decoding")
    preprocessing = _section(config, "preprocessing")
    outputs = _section(config, "outputs")
    tuning = _section(config, "tuning")
    calibration = _section(config, "calibration")
    observations = _section(config, "observations")

    epochs_path = _require_path(
        _first(config.get("epochs"), dataset.get("epochs"), dataset.get("epochs_path")),
        config_dir=config_dir,
        name="dataset.epochs",
    )
    out_path = _require_path(
        _first(config.get("out"), outputs.get("metrics_csv"), outputs.get("out"), outputs.get("results_csv")),
        config_dir=config_dir,
        name="outputs.metrics_csv",
    )

    label_column = _first(config.get("label_column"), decoding.get("label_column"))
    if not label_column:
        raise DatasetConfigError("Missing required value 'decoding.label_column'.")

    metadata_csv = _resolve_path(
        _first(config.get("metadata_csv"), dataset.get("metadata_csv"), dataset.get("metadata")),
        config_dir=config_dir,
    )
    calibration_out_path = _resolve_path(
        _first(config.get("calibration_out"), outputs.get("calibration_csv"), calibration.get("out")),
        config_dir=config_dir,
    )
    observation_out_path = _resolve_path(
        _first(config.get("observations_out"), outputs.get("observations_csv"), observations.get("out")),
        config_dir=config_dir,
    )

    temporal_train_window = _as_float_pair(
        _first(decoding.get("temporal_train_window"), preprocessing.get("temporal_train_window")),
        name="temporal_train_window",
    )
    baseline_window = _as_float_pair(
        _first(preprocessing.get("baseline_window"), decoding.get("baseline_window")),
        name="baseline_window",
    )

    return {
        "epochs_path": epochs_path,
        "metadata_csv": metadata_csv,
        "label_column": str(label_column),
        "group_column": _first(config.get("group_column"), decoding.get("group_column")),
        "dataset_name": _first(dataset.get("name"), config.get("dataset"), default=""),
        "out_path": out_path,
        "picks": _first(preprocessing.get("picks"), decoding.get("picks"), default="data"),
        "tmin": _first(preprocessing.get("tmin"), decoding.get("tmin")),
        "tmax": _first(preprocessing.get("tmax"), decoding.get("tmax")),
        "window_ms": _first(preprocessing.get("window_ms"), decoding.get("window_ms"), default=20.0),
        "step_ms": _first(preprocessing.get("step_ms"), decoding.get("step_ms"), default=10.0),
        "n_splits": _first(decoding.get("n_splits"), config.get("n_splits"), default=5),
        "max_iter": _first(decoding.get("max_iter"), config.get("max_iter"), default=1000),
        "decoder": _first(decoding.get("decoder"), config.get("decoder"), default="logistic"),
        "emission_mode": _first(decoding.get("emission_mode"), config.get("emission_mode"), default="calibrated"),
        "feature_preprocessor": _first(
            preprocessing.get("feature_preprocessor"),
            decoding.get("feature_preprocessor"),
            default="none",
        ),
        "pca_components": _first(preprocessing.get("pca_components"), decoding.get("pca_components")),
        "normalization": _first(preprocessing.get("normalization"), decoding.get("normalization"), default="none"),
        "baseline_window": baseline_window,
        "tune_hyperparameters": _as_bool(
            _first(tuning.get("enabled"), decoding.get("tune_hyperparameters")),
            default=False,
        ),
        "tuning_cv_splits": _first(tuning.get("cv_splits"), decoding.get("tuning_cv_splits"), default=3),
        "tuning_scoring": _first(tuning.get("scoring"), decoding.get("tuning_scoring"), default="accuracy"),
        "tuning_c_grid": _first(tuning.get("c_grid"), decoding.get("tuning_c_grid")),
        "calibration_out_path": calibration_out_path,
        "calibration_bins": _first(calibration.get("bins"), decoding.get("calibration_bins"), default=10),
        "observation_out_path": observation_out_path,
        "subject": _first(dataset.get("subject"), config.get("subject")),
        "temporal_train_window": temporal_train_window,
        "time_decode_backend": _first(decoding.get("time_decode_backend"), config.get("time_decode_backend"), default="sklearn"),
        "alignment_method": _first(decoding.get("alignment_method"), config.get("alignment_method"), default="none"),
        "alignment_anchor_mode": _first(decoding.get("alignment_anchor_mode"), config.get("alignment_anchor_mode"), default="class_mean"),
        "alignment_anchor_column": _first(decoding.get("alignment_anchor_column"), config.get("alignment_anchor_column")),
        "alignment_repetition_cap": _first(decoding.get("alignment_repetition_cap"), config.get("alignment_repetition_cap"), default=16),
        "alignment_components": _first(decoding.get("alignment_components"), config.get("alignment_components"), default=64),
        "alignment_times": _first(decoding.get("alignment_times"), config.get("alignment_times")),
        "alignment_target_projection": _first(
            decoding.get("alignment_target_projection"),
            config.get("alignment_target_projection"),
            default="group_projection",
        ),
        "alignment_target_calibration_per_anchor": _first(
            decoding.get("alignment_target_calibration_per_anchor"),
            config.get("alignment_target_calibration_per_anchor"),
            default=1,
        ),
        "alignment_target_calibration_seed": _first(
            decoding.get("alignment_target_calibration_seed"),
            config.get("alignment_target_calibration_seed"),
            default=13,
        ),
        "dann_hidden_units": _first(decoding.get("dann_hidden_units"), config.get("dann_hidden_units"), default=64),
        "dann_embedding_dim": _first(decoding.get("dann_embedding_dim"), config.get("dann_embedding_dim"), default=32),
        "dann_max_epochs": _first(decoding.get("dann_max_epochs"), config.get("dann_max_epochs"), default=80),
        "dann_batch_size": _first(decoding.get("dann_batch_size"), config.get("dann_batch_size"), default=128),
        "dann_learning_rate": _first(decoding.get("dann_learning_rate"), config.get("dann_learning_rate"), default=1e-3),
        "dann_weight_decay": _first(decoding.get("dann_weight_decay"), config.get("dann_weight_decay"), default=1e-4),
        "dann_domain_loss_weight": _first(decoding.get("dann_domain_loss_weight"), config.get("dann_domain_loss_weight"), default=0.1),
        "dann_validation_fraction": _first(decoding.get("dann_validation_fraction"), config.get("dann_validation_fraction"), default=0.1),
        "dann_patience": _first(decoding.get("dann_patience"), config.get("dann_patience"), default=10),
        "dann_dropout": _first(decoding.get("dann_dropout"), config.get("dann_dropout"), default=0.1),
        "dann_random_state": _first(decoding.get("dann_random_state"), config.get("dann_random_state"), default=13),
        "dann_device": _first(decoding.get("dann_device"), config.get("dann_device"), default="auto"),
        "label_shuffle_control": _as_bool(
            _first(decoding.get("label_shuffle_control"), config.get("label_shuffle_control")),
            default=False,
        ),
        "label_shuffle_seed": _first(decoding.get("label_shuffle_seed"), config.get("label_shuffle_seed"), default=13),
    }


def _validate_static_config(config: Mapping[str, Any], *, config_path: Path, check_files: bool = True) -> list[str]:
    problems: list[str] = []
    try:
        kwargs = _decode_kwargs(config, config_path=config_path)
    except DatasetConfigError as exc:
        return [str(exc)]

    if check_files:
        if kwargs["metadata_csv"] is not None and not kwargs["metadata_csv"].exists():
            problems.append(f"metadata_csv does not exist: {kwargs['metadata_csv']}")
        if not kwargs["epochs_path"].exists():
            problems.append(f"epochs file does not exist: {kwargs['epochs_path']}")

    output_paths = [
        kwargs["out_path"],
        kwargs["calibration_out_path"],
        kwargs["observation_out_path"],
    ]
    for path in output_paths:
        if path is not None and path.exists() and path.is_dir():
            problems.append(f"Output path points to a directory, not a file: {path}")
    return problems


def validate_dataset_config(
    config_path: Path,
    *,
    check_files: bool = True,
    check_metadata_columns: bool = True,
) -> list[str]:
    """Return validation problems for a config. An empty list means valid."""

    config = load_dataset_config(config_path)
    problems = _validate_static_config(config, config_path=config_path, check_files=check_files)
    if problems or not check_files:
        return problems

    kwargs = _decode_kwargs(config, config_path=config_path)
    epochs_path: Path = kwargs["epochs_path"]
    if not epochs_path.exists():
        return problems

    if not check_metadata_columns:
        return problems

    metadata_csv: Path | None = kwargs["metadata_csv"]
    metadata: pd.DataFrame | None = None
    if metadata_csv is not None and metadata_csv.exists():
        metadata = pd.read_csv(metadata_csv)
    else:
        try:
            import mne

            epochs = mne.read_epochs(epochs_path, preload=False, verbose="error")
            metadata = epochs.metadata
        except Exception as exc:  # pragma: no cover - depends on user data files
            problems.append(f"Could not inspect metadata columns from {epochs_path}: {exc}")
            return problems

    if metadata is None:
        problems.append("No metadata is available. Provide dataset.metadata_csv or store metadata in the epochs file.")
        return problems

    label_column = kwargs["label_column"]
    group_column = kwargs["group_column"]
    if label_column not in metadata.columns:
        problems.append(f"label_column '{label_column}' is missing from metadata columns: {', '.join(map(str, metadata.columns))}")
    if group_column is not None and group_column not in metadata.columns:
        problems.append(f"group_column '{group_column}' is missing from metadata columns: {', '.join(map(str, metadata.columns))}")
    return problems


def run_decode_from_config(config_path: Path):
    """Run the configured MNE time-resolved decoding workflow."""

    config = load_dataset_config(config_path)
    kwargs = _decode_kwargs(config, config_path=config_path)
    return run_time_resolved_decode(**kwargs)


def decode_from_config_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a NeuRepTrace decoding workflow from a JSON/YAML config.")
    parser.add_argument("config", type=Path, help="Path to a .json, .yml, or .yaml workflow config.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the config and input metadata without running the decoder.",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Only validate config structure; do not require input files to exist.",
    )
    args = parser.parse_args(argv)

    problems = validate_dataset_config(args.config, check_files=not args.no_check_files)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 2
    if args.validate_only:
        print(f"Config is valid: {args.config}")
        return 0

    results = run_decode_from_config(args.config)
    print(f"Decoded {len(results)} result rows from {args.config}")
    return 0


def validate_dataset_config_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a NeuRepTrace JSON/YAML dataset workflow config.")
    parser.add_argument("config", type=Path, help="Path to a .json, .yml, or .yaml workflow config.")
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Only validate config structure; do not require input files to exist.",
    )
    parser.add_argument(
        "--no-check-metadata-columns",
        action="store_true",
        help="Do not inspect metadata columns in the epochs file or metadata CSV.",
    )
    args = parser.parse_args(argv)

    problems = validate_dataset_config(
        args.config,
        check_files=not args.no_check_files,
        check_metadata_columns=not args.no_check_metadata_columns,
    )
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 2
    print(f"Config is valid: {args.config}")
    return 0
