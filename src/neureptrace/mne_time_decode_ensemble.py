from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neureptrace.decoding import DECODER_CLI_CHOICES, FEATURE_PREPROCESSOR_CHOICES, TUNING_SCORING_CHOICES, normalize_feature_preprocessor
from neureptrace.mne_time_decode import (
    EMISSION_RUN_CHOICES,
    FEATURE_PREPROCESSOR_RUN_CHOICES,
    RESULT_SELECTION_METRIC_CHOICES,
    RESULT_SELECTION_MINIMIZE_METRICS,
    _best_time_by_metric,
    run_time_resolved_decode as _run_time_resolved_decode,
)
from neureptrace.observation_ensemble import (
    DEFAULT_BASELINE_GROUP_COLUMNS,
    DEFAULT_BASELINE_WINDOW,
    DEFAULT_MIN_PROBABILITY,
    DEFAULT_WEIGHTS,
    ensemble_probability_observations,
    summarize_ensemble_metrics,
)
from neureptrace.observation_schema import read_validated_probability_observations
from neureptrace.observations import ProbabilityObservationTable

ENSEMBLE_DECODER = "logistic_svm_ensemble"
ENSEMBLE_DECODER_ALIASES = (
    ENSEMBLE_DECODER,
    "logistic-svm-ensemble",
    "logistic_linear_svm_ensemble",
    "logistic-linear-svm-ensemble",
    "calibrated_logistic_svm_ensemble",
    "calibrated-logistic-svm-ensemble",
    "calibrated_logistic_linear_svm_ensemble",
    "calibrated-logistic-linear-svm-ensemble",
)
ENSEMBLE_DECODER_CLI_CHOICES = tuple(dict.fromkeys((*DECODER_CLI_CHOICES, *ENSEMBLE_DECODER_ALIASES)))
ENSEMBLE_OUTPUT_EMISSION_MODE = "baseline_debiased_calibrated_ensemble"
_SOURCE_DECODERS = ("logistic", "linear_svm")


def normalize_time_decode_decoder_name(decoder: str) -> str:
    """Normalize the extended time-decoding decoder namespace."""

    normalized = str(decoder).strip().lower().replace("-", "_")
    if normalized in {alias.replace("-", "_") for alias in ENSEMBLE_DECODER_ALIASES}:
        return ENSEMBLE_DECODER
    return normalized


def _is_ensemble_decoder(decoder: str) -> bool:
    return normalize_time_decode_decoder_name(decoder) == ENSEMBLE_DECODER


def _parse_weights(weights: Sequence[float] | None) -> tuple[float, float]:
    if weights is None:
        return tuple(DEFAULT_WEIGHTS)  # type: ignore[return-value]
    if len(weights) != 2:
        raise ValueError("logistic_svm_ensemble expects exactly two weights: logistic and linear_svm.")
    return float(weights[0]), float(weights[1])


def _read_optional_observations(observation_path: Path | None) -> pd.DataFrame | None:
    if observation_path is None:
        return None
    return pd.read_csv(observation_path)


def run_time_resolved_decode(
    epochs_path: Path,
    label_column: str,
    out_path: Path,
    *,
    metadata_csv: Path | None = None,
    group_column: str | None = None,
    picks: str = "data",
    tmin: float | None = None,
    tmax: float | None = None,
    window_ms: float = 20.0,
    step_ms: float = 10.0,
    n_splits: int = 5,
    max_iter: int = 1000,
    decoder: str = "logistic",
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
    temporal_train_window: tuple[float, float] | None = None,
    ensemble_weights: Sequence[float] | None = None,
    ensemble_baseline_window: tuple[float, float] | None = DEFAULT_BASELINE_WINDOW,
    ensemble_baseline_group_columns: Sequence[str] = DEFAULT_BASELINE_GROUP_COLUMNS,
    ensemble_min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> pd.DataFrame:
    """Run time-resolved decoding with optional logistic/SVM probability ensembling.

    The extended ``logistic_svm_ensemble`` decoder is intentionally implemented
    as a first-class wrapper around the existing held-out probability pipeline:
    logistic and linear-SVM source decoders are fit inside the same outer folds,
    their calibrated observation probabilities are aligned one-to-one, combined
    by baseline-debiased log-probability averaging, and then summarized through
    the standard NeuRepTrace observation/metric schema.
    """

    if not _is_ensemble_decoder(decoder):
        return _run_time_resolved_decode(
            epochs_path=epochs_path,
            metadata_csv=metadata_csv,
            label_column=label_column,
            group_column=group_column,
            out_path=out_path,
            picks=picks,
            tmin=tmin,
            tmax=tmax,
            window_ms=window_ms,
            step_ms=step_ms,
            n_splits=n_splits,
            max_iter=max_iter,
            decoder=decoder,
            emission_mode=emission_mode,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            tune_hyperparameters=tune_hyperparameters,
            tuning_cv_splits=tuning_cv_splits,
            tuning_scoring=tuning_scoring,
            tuning_c_grid=tuning_c_grid,
            calibration_out_path=calibration_out_path,
            calibration_bins=calibration_bins,
            observation_out_path=observation_out_path,
            subject=subject,
            temporal_train_window=temporal_train_window,
        )

    if emission_mode != "calibrated":
        raise ValueError("logistic_svm_ensemble is defined for --emission-mode calibrated only.")

    weights = _parse_weights(ensemble_weights)
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)

    with tempfile.TemporaryDirectory(prefix="neureptrace_logistic_svm_ensemble_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        source_observation_paths: list[Path] = []
        source_metric_frames: list[pd.DataFrame] = []
        for source_decoder in _SOURCE_DECODERS:
            source_out = tmp_dir / f"{source_decoder}_time_decode.csv"
            source_observations = tmp_dir / f"{source_decoder}_observations.csv"
            source_metric_frames.append(
                _run_time_resolved_decode(
                    epochs_path=epochs_path,
                    metadata_csv=metadata_csv,
                    label_column=label_column,
                    group_column=group_column,
                    out_path=source_out,
                    picks=picks,
                    tmin=tmin,
                    tmax=tmax,
                    window_ms=window_ms,
                    step_ms=step_ms,
                    n_splits=n_splits,
                    max_iter=max_iter,
                    decoder=source_decoder,
                    emission_mode="calibrated",
                    feature_preprocessor=feature_preprocessor,
                    pca_components=pca_components,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid,
                    calibration_out_path=None,
                    calibration_bins=calibration_bins,
                    observation_out_path=source_observations,
                    subject=subject,
                    temporal_train_window=temporal_train_window,
                )
            )
            source_observation_paths.append(source_observations)

        observations = read_validated_probability_observations(
            source_observation_paths,
            profile="generic",
            require_normalized=True,
        )
        ensemble = ensemble_probability_observations(
            observations,
            decoders=_SOURCE_DECODERS,
            weights=weights,
            source_emission_mode="calibrated",
            baseline_window=ensemble_baseline_window,
            baseline_group_columns=ensemble_baseline_group_columns,
            min_probability=ensemble_min_probability,
            output_decoder=ENSEMBLE_DECODER,
            output_emission_mode=ENSEMBLE_OUTPUT_EMISSION_MODE,
        )

    if observation_out_path is not None:
        ProbabilityObservationTable(ensemble).to_csv(observation_out_path)

    results = summarize_ensemble_metrics(ensemble, ece_bins=calibration_bins)
    results["feature_preprocessor"] = feature_preprocessor_name
    results["pca_components"] = "" if pca_components is None else pca_components
    results["source_decoders"] = "|".join(_SOURCE_DECODERS)
    results["ensemble_weights"] = "|".join(f"{weight:.12g}" for weight in weights)
    results["baseline_window_start"] = "" if ensemble_baseline_window is None else float(ensemble_baseline_window[0])
    results["baseline_window_stop"] = "" if ensemble_baseline_window is None else float(ensemble_baseline_window[1])

    # Preserve temporal metadata columns when they are constant across the source summaries.
    for column in (
        "temporal_mode",
        "temporal_train_window_start",
        "temporal_train_window_stop",
        "train_time",
        "train_window_start",
        "train_window_stop",
        "n_train_windows",
    ):
        values = [frame[column].iloc[0] for frame in source_metric_frames if column in frame.columns and not frame.empty]
        if values and all(value == values[0] for value in values):
            results[column] = values[0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    if calibration_out_path is not None:
        # Full calibration-bin output for ensembled observations is already
        # represented by ECE in the metrics table.  Write a compact placeholder
        # with provenance rather than silently ignoring the requested path.
        calibration_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "decoder": ENSEMBLE_DECODER,
                    "emission_mode": ENSEMBLE_OUTPUT_EMISSION_MODE,
                    "note": "Calibration bins for logistic_svm_ensemble are summarized by ECE in the metrics CSV.",
                }
            ]
        ).to_csv(calibration_out_path, index=False)

    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run calibrated time-resolved decoding on an MNE Epochs FIF file, including logistic/SVM probability ensembling."
    )
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--group-column")
    parser.add_argument("--picks", default="data")
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--decoder", choices=ENSEMBLE_DECODER_CLI_CHOICES, default="logistic")
    parser.add_argument("--emission-mode", choices=EMISSION_RUN_CHOICES, default="calibrated")
    parser.add_argument("--feature-preprocessor", choices=FEATURE_PREPROCESSOR_RUN_CHOICES, default="none")
    parser.add_argument(
        "--pca-components",
        help=(
            "PCA component count or explained-variance fraction. With "
            "--feature-preprocessor anova-select, this is the selected feature percentile."
        ),
    )
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Use nested inner-CV hyperparameter selection inside each outer train fold.")
    parser.add_argument("--tuning-cv-splits", type=int, default=3, help="Maximum number of inner CV folds for --tune-hyperparameters.")
    parser.add_argument("--tuning-scoring", choices=TUNING_SCORING_CHOICES, default="accuracy", help="Inner-CV objective for --tune-hyperparameters.")
    parser.add_argument("--selection-metric", choices=RESULT_SELECTION_METRIC_CHOICES, default="accuracy", help="Metric used only for the console 'best time' summary.")
    parser.add_argument("--tuning-c-grid", default="0.01,0.1,1.0,10.0,100.0", help="Comma-separated positive C values for tuned logistic regression and linear SVM.")
    parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--observations-out", type=Path, help="Optional held-out trial/time probability observation CSV.")
    parser.add_argument("--subject", help="Optional subject identifier to include in output CSVs.")
    parser.add_argument(
        "--temporal-train-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help=(
            "Train one model per time-window center in START..STOP seconds, "
            "evaluate each model at every test time, and average probabilities."
        ),
    )
    parser.add_argument(
        "--ensemble-weight",
        action="append",
        type=float,
        dest="ensemble_weights",
        help="Weight for logistic_svm_ensemble sources. Repeat twice: first logistic, then linear_svm. Defaults to 0.5/0.5.",
    )
    parser.add_argument("--ensemble-baseline-window", nargs=2, type=float, default=DEFAULT_BASELINE_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--no-ensemble-baseline-debiasing", action="store_true")
    parser.add_argument(
        "--ensemble-baseline-group-column",
        action="append",
        dest="ensemble_baseline_group_columns",
        help="Column for ensemble baseline-offset grouping. Repeat to override defaults subject and fold.",
    )
    parser.add_argument("--ensemble-min-probability", type=float, default=DEFAULT_MIN_PROBABILITY)
    args = parser.parse_args(argv)

    results = run_time_resolved_decode(
        epochs_path=args.epochs,
        metadata_csv=args.metadata_csv,
        label_column=args.label_column,
        group_column=args.group_column,
        out_path=args.out,
        picks=args.picks,
        tmin=args.tmin,
        tmax=args.tmax,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        n_splits=args.n_splits,
        max_iter=args.max_iter,
        decoder=args.decoder,
        emission_mode=args.emission_mode,
        feature_preprocessor=args.feature_preprocessor,
        pca_components=args.pca_components,
        tune_hyperparameters=args.tune_hyperparameters,
        tuning_cv_splits=args.tuning_cv_splits,
        tuning_scoring=args.tuning_scoring,
        tuning_c_grid=args.tuning_c_grid,
        calibration_out_path=args.calibration_out,
        calibration_bins=args.calibration_bins,
        observation_out_path=args.observations_out,
        subject=args.subject,
        temporal_train_window=tuple(args.temporal_train_window) if args.temporal_train_window is not None else None,
        ensemble_weights=tuple(args.ensemble_weights) if args.ensemble_weights is not None else None,
        ensemble_baseline_window=None if args.no_ensemble_baseline_debiasing else tuple(args.ensemble_baseline_window),
        ensemble_baseline_group_columns=tuple(args.ensemble_baseline_group_columns or DEFAULT_BASELINE_GROUP_COLUMNS),
        ensemble_min_probability=args.ensemble_min_probability,
    )

    print(f"Wrote {args.out}")
    if args.observations_out is not None:
        print(f"Wrote probability observations: {args.observations_out}")
    for emission_mode_name, summary in results.groupby("emission_mode", sort=True):
        time_summary = summary.groupby("time")[["accuracy", "log_loss", "brier", "ece"]].mean()
        best_time = _best_time_by_metric(time_summary, args.selection_metric)
        best_value = time_summary.loc[best_time, args.selection_metric]
        direction = "lowest" if args.selection_metric in RESULT_SELECTION_MINIMIZE_METRICS else "highest"
        print(
            f"Best {emission_mode_name} mean {args.selection_metric} "
            f"({direction}): {best_value:.3f} at {best_time:.3f}s"
        )


if __name__ == "__main__":
    main()
