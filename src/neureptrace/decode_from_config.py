"""Run time-resolved decoding from a dataset/workflow config."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neureptrace.dataset_config import (
    apply_overrides,
    effective_config,
    expand_path,
    load_config,
    load_epoch_dataset_from_config,
    provenance_payload,
)
from neureptrace import mne_time_decode
from neureptrace.mne_time_decode_ensemble import (
    ENSEMBLE_DECODER,
    normalize_time_decode_decoder_name,
    run_time_resolved_decode as run_ensemble_time_resolved_decode,
)
from neureptrace.mne_time_decode import DEFAULT_BASELINE_WINDOW


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return dict(value)


def _window_ms(
    preprocessing: Mapping[str, Any],
    *,
    key_ms: str,
    key_seconds: str,
    default: float,
) -> float:
    if key_ms in preprocessing:
        return float(preprocessing[key_ms])
    if key_seconds in preprocessing:
        return float(preprocessing[key_seconds]) * 1000.0
    return default


def _list_value(value: Any, *, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        if text.startswith("["):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError:
                loaded = None
            if loaded is None:
                if not text.endswith("]"):
                    raise ValueError(f"{name} must be a list, comma-separated string, or whitespace-separated string.")
                text = text[1:-1].strip()
                return [part.strip() for chunk in text.split(",") for part in chunk.split() if part.strip()]
            if not isinstance(loaded, list):
                raise ValueError(f"{name} must be a list, comma-separated string, or whitespace-separated string.")
            return loaded
        return [part.strip() for chunk in text.split(",") for part in chunk.split() if part.strip()]
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return list(value)
    return [value]


def _float_tuple(value: Any, *, name: str, length: int | None = None, allow_none: bool = False) -> tuple[float, ...] | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be null.")
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "off", "false"}:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be disabled.")
    values = tuple(float(item) for item in _list_value(value, name=name))
    if length is not None and len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} value(s).")
    return values


def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    values = tuple(str(item).strip() for item in _list_value(value, name=name) if str(item).strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def _find_project_root(start: Path) -> Path:
    """Find a repository-like project root, falling back to the current directory."""

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return Path.cwd()


def _base_for_policy(config: Mapping[str, Any], *, config_dir: Path) -> Path:
    paths = _section(config, "paths")
    policy = str(
        paths.get("base", paths.get("relative_to", "cwd"))
    ).strip().lower().replace("-", "_")
    if policy in {"cwd", "current_working_directory"}:
        return Path.cwd()
    if policy == "config_dir":
        return config_dir
    if policy == "project_root":
        return _find_project_root(config_dir)
    raise ValueError("paths.base must be one of: cwd, config_dir, project_root.")


def _output_base_dir(config: Mapping[str, Any], *, config_dir: Path) -> Path:
    outputs = _section(config, "outputs")
    policy_base = _base_for_policy(config, config_dir=config_dir)
    base_dir = outputs.get("base_dir") or outputs.get("dir")
    if base_dir in {None, ""}:
        return policy_base
    dataset_name = str(_section(config, "dataset").get("name", "dataset"))
    return expand_path(str(base_dir).format(dataset=dataset_name), base_dir=policy_base)


def _resolve_output(
    config: Mapping[str, Any],
    *,
    config_dir: Path,
    key: str,
    default: str | None = None,
) -> Path | None:
    outputs = _section(config, "outputs")
    value = outputs.get(key, default)
    if value is None or value == "":
        return None
    dataset_name = str(_section(config, "dataset").get("name", "dataset"))
    formatted = str(value).format(dataset=dataset_name)
    path = Path(formatted)
    if path.is_absolute():
        return path
    return _output_base_dir(config, config_dir=config_dir) / path


def _decode_kwargs(config: Mapping[str, Any], *, config_dir: Path) -> dict[str, Any]:
    preprocessing = _section(config, "preprocessing")
    decoding = _section(config, "decoding") or _section(config, "workflow")
    if "label_column" not in decoding:
        raise ValueError("decode-from-config requires decoding.label_column.")

    baseline_window = preprocessing.get("baseline_window", DEFAULT_BASELINE_WINDOW)
    temporal_train_window = decoding.get(
        "temporal_train_window",
        preprocessing.get("temporal_train_window"),
    )
    decode_window = decoding.get(
        "decode_window",
        preprocessing.get("decode_window"),
    )
    temporal_train_mode = decoding.get(
        "temporal_train_mode",
        preprocessing.get("temporal_train_mode"),
    )

    kwargs = {
        "label_column": decoding["label_column"],
        "group_column": decoding.get("group_column"),
        "out_path": _resolve_output(
            config,
            config_dir=config_dir,
            key="summary_csv",
            default="results/{dataset}_summary.csv",
        ),
        "picks": preprocessing.get("picks", "data"),
        "tmin": preprocessing.get("tmin"),
        "tmax": preprocessing.get("tmax"),
        "window_ms": _window_ms(
            preprocessing,
            key_ms="window_ms",
            key_seconds="window_size",
            default=20.0,
        ),
        "step_ms": _window_ms(
            preprocessing,
            key_ms="step_ms",
            key_seconds="window_step",
            default=10.0,
        ),
        "n_splits": int(decoding.get("n_splits", 5)),
        "max_iter": int(decoding.get("max_iter", 1000)),
        "decoder": decoding.get("decoder", decoding.get("classifier", "logistic")),
        "emission_mode": decoding.get("emission_mode", "calibrated"),
        "feature_preprocessor": decoding.get(
            "feature_preprocessor",
            preprocessing.get("feature_preprocessor", "none"),
        ),
        "pca_components": decoding.get(
            "pca_components",
            preprocessing.get("pca_components"),
        ),
        "normalization": preprocessing.get("normalization", "none"),
        "baseline_window": tuple(baseline_window) if baseline_window is not None else None,
        "tune_hyperparameters": bool(decoding.get("tune_hyperparameters", False)),
        "tuning_cv_splits": int(decoding.get("tuning_cv_splits", 3)),
        "tuning_scoring": decoding.get("tuning_scoring", "accuracy"),
        "tuning_c_grid": decoding.get("tuning_c_grid"),
        "calibration_out_path": _resolve_output(
            config,
            config_dir=config_dir,
            key="calibration_csv",
        ),
        "calibration_bins": int(decoding.get("calibration_bins", 10)),
        "observation_out_path": _resolve_output(
            config,
            config_dir=config_dir,
            key="observations_csv",
        ),
        "subject": decoding.get("subject"),
        "decode_window": tuple(decode_window) if decode_window is not None else None,
        "temporal_train_window": (
            tuple(temporal_train_window) if temporal_train_window is not None else None
        ),
        "time_decode_backend": decoding.get("time_decode_backend", "sklearn"),
        "class_prior_correction": decoding.get(
            "class_prior_correction",
            decoding.get("prior_correction", "none"),
        ),
        "label_shuffle_control": bool(decoding.get("label_shuffle_control", False)),
        "label_shuffle_seed": int(decoding.get("label_shuffle_seed", 13)),
    }
    if temporal_train_mode is not None:
        kwargs["temporal_train_mode"] = temporal_train_mode
    if "outer_test_groups" in decoding or "outer_test_group" in decoding:
        outer_test_groups = decoding.get("outer_test_groups", decoding.get("outer_test_group"))
        if outer_test_groups is not None and outer_test_groups != "":
            kwargs["outer_test_groups"] = _string_tuple(
                outer_test_groups,
                name="decoding.outer_test_groups",
            )
    if normalize_time_decode_decoder_name(str(kwargs["decoder"])) == ENSEMBLE_DECODER:
        if "ensemble_weights" in decoding or "ensemble_weight" in decoding:
            ensemble_weights = decoding.get("ensemble_weights", decoding.get("ensemble_weight"))
            if ensemble_weights is not None and ensemble_weights != "":
                kwargs["ensemble_weights"] = _float_tuple(ensemble_weights, name="decoding.ensemble_weights", length=2)
        if "ensemble_source_decoders" in decoding or "ensemble_source_decoder" in decoding:
            ensemble_source_decoders = decoding.get("ensemble_source_decoders", decoding.get("ensemble_source_decoder"))
            if ensemble_source_decoders is not None and ensemble_source_decoders != "":
                kwargs["ensemble_source_decoders"] = _string_tuple(
                    ensemble_source_decoders,
                    name="decoding.ensemble_source_decoders",
                )
        if "ensemble_baseline_window" in decoding:
            kwargs["ensemble_baseline_window"] = _float_tuple(
                decoding["ensemble_baseline_window"],
                name="decoding.ensemble_baseline_window",
                length=2,
                allow_none=True,
            )
        if "ensemble_baseline_group_columns" in decoding:
            kwargs["ensemble_baseline_group_columns"] = _string_tuple(
                decoding["ensemble_baseline_group_columns"],
                name="decoding.ensemble_baseline_group_columns",
            )
        if "ensemble_min_probability" in decoding:
            kwargs["ensemble_min_probability"] = float(decoding["ensemble_min_probability"])
    return kwargs


def _output_paths_from_kwargs(kwargs: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ("out_path", "calibration_out_path", "observation_out_path"):
        value = kwargs.get(key)
        if value is not None:
            paths.append(Path(value))
    return paths


def _write_provenance_sidecars(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    config_dir: Path,
    output_paths: Sequence[Path],
) -> None:
    outputs = _section(config, "outputs")
    if not bool(outputs.get("provenance", True)):
        return
    payload = provenance_payload(
        config,
        config_path=config_path,
        base_dir=config_dir,
        include_file_hashes=bool(outputs.get("hash_input_files", True)),
    )
    for output_path in output_paths:
        sidecar = Path(str(output_path) + ".provenance.json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def _run_time_decode_in_memory(dataset, *, channel_type: str, kwargs: dict[str, Any]):
    """Run the existing MNE decoder without writing a temporary FIF file."""

    epochs = dataset.to_mne_epochs(channel_type=channel_type)
    metadata = dataset.metadata.reset_index(drop=True).copy()
    original_loader = mne_time_decode._load_epochs_and_metadata
    decoder_name = normalize_time_decode_decoder_name(str(kwargs.get("decoder", "")))
    runner = run_ensemble_time_resolved_decode if decoder_name == ENSEMBLE_DECODER else mne_time_decode.run_time_resolved_decode

    def _configured_loader(_epochs_path, _metadata_csv, **_kwargs):
        return epochs, metadata

    mne_time_decode._load_epochs_and_metadata = _configured_loader
    try:
        return runner(
            epochs_path=Path("<configured-in-memory-epochs>"),
            metadata_csv=None,
            **kwargs,
        )
    finally:
        mne_time_decode._load_epochs_and_metadata = original_loader


def run_decode_from_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    write_provenance: bool | None = None,
):
    """Load a dataset config and run the configured time-resolved decoder."""

    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    if write_provenance is not None:
        config.setdefault("outputs", {})["provenance"] = bool(write_provenance)
    dataset = load_epoch_dataset_from_config(
        config,
        base_dir=config_path.parent,
        check_files=True,
    )
    channel_type = str(_section(config, "dataset").get("channel_type", "mag"))
    kwargs = _decode_kwargs(config, config_dir=config_path.parent)
    results = _run_time_decode_in_memory(dataset, channel_type=channel_type, kwargs=kwargs)
    _write_provenance_sidecars(
        config,
        config_path=config_path,
        config_dir=config_path.parent,
        output_paths=_output_paths_from_kwargs(kwargs),
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run NeuRepTrace time-resolved decoding from a JSON/YAML config."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override a dotted config key, e.g. --set decoding.classifier=lda.",
    )
    parser.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print the effective config and exit without decoding.",
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="Do not write .provenance.json sidecars next to output CSVs.",
    )
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.overrides)
    if args.print_effective_config:
        print(
            json.dumps(
                effective_config(config, base_dir=args.config.parent),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0

    results = run_decode_from_config(
        args.config,
        overrides=args.overrides,
        write_provenance=not args.no_provenance,
    )
    print(f"Wrote {len(results)} decoding rows from {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
