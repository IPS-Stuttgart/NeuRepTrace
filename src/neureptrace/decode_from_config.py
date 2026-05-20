"""Run time-resolved decoding from a dataset/workflow config."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from neureptrace.dataset_config import (
    apply_overrides,
    expand_path,
    load_config,
    load_epoch_dataset_from_config,
)
from neureptrace.mne_time_decode import DEFAULT_BASELINE_WINDOW, run_time_resolved_decode


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


def _resolve_output(
    config: Mapping[str, Any],
    *,
    base_dir: Path,
    key: str,
    default: str | None = None,
) -> Path | None:
    outputs = _section(config, "outputs")
    value = outputs.get(key, default)
    if value is None or value == "":
        return None
    dataset_name = str(_section(config, "dataset").get("name", "dataset"))
    formatted = str(value).format(dataset=dataset_name)
    return expand_path(formatted, base_dir=base_dir)


def _decode_kwargs(config: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    preprocessing = _section(config, "preprocessing")
    decoding = _section(config, "decoding") or _section(config, "workflow")
    if "label_column" not in decoding:
        raise ValueError("decode-from-config requires decoding.label_column.")

    baseline_window = preprocessing.get("baseline_window", DEFAULT_BASELINE_WINDOW)
    temporal_train_window = decoding.get(
        "temporal_train_window",
        preprocessing.get("temporal_train_window"),
    )

    return {
        "label_column": decoding["label_column"],
        "group_column": decoding.get("group_column"),
        "out_path": _resolve_output(
            config,
            base_dir=base_dir,
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
            base_dir=base_dir,
            key="calibration_csv",
        ),
        "calibration_bins": int(decoding.get("calibration_bins", 10)),
        "observation_out_path": _resolve_output(
            config,
            base_dir=base_dir,
            key="observations_csv",
        ),
        "subject": decoding.get("subject"),
        "temporal_train_window": (
            tuple(temporal_train_window) if temporal_train_window is not None else None
        ),
    }


def run_decode_from_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
):
    """Load a dataset config and run the configured time-resolved decoder."""

    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    dataset = load_epoch_dataset_from_config(
        config,
        base_dir=config_path.parent,
        check_files=True,
    )
    channel_type = str(_section(config, "dataset").get("channel_type", "mag"))
    epochs = dataset.to_mne_epochs(channel_type=channel_type)
    kwargs = _decode_kwargs(config, base_dir=config_path.parent)

    with TemporaryDirectory(prefix="neureptrace-config-") as tmpdir:
        epochs_path = Path(tmpdir) / "configured-epo.fif"
        epochs.save(epochs_path, overwrite=True, verbose="error")
        return run_time_resolved_decode(
            epochs_path=epochs_path,
            metadata_csv=None,
            **kwargs,
        )


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
    args = parser.parse_args(argv)

    results = run_decode_from_config(args.config, overrides=args.overrides)
    print(f"Wrote {len(results)} decoding rows from {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
