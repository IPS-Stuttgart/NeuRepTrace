"""Run and combine the locked Katja ensemble population calibration curve.

This is a follow-up to the completed k20 source/model ensemble evaluation.  It
uses the same 16 targets, five calibration seeds, exact-ICA feature cache, model
members, and nested calibration split context, but evaluates all six registered
calibration sizes.  Each target is evaluated in a separate process so multiple
workers can share the two workstation GPUs without sharing Torch state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from katja_ensemble_population import PRIMARY_CONFIGURATION, _summarize_population

DEFAULT_TARGETS = (
    "s06",
    "s08",
    "s09",
    "s10",
    "s11",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s20",
    "s21",
    "s24",
    "s25",
    "s28",
)
DEFAULT_CALIBRATION_COUNTS = (1, 3, 5, 10, 15, 20)
DEFAULT_CALIBRATION_SEEDS = (13, 29, 47, 71, 101)
CONTROL_CONFIGURATION = "nine_single_protocol"
EXPECTED_CONFIGURATIONS = (
    "nine_single_protocol",
    "nine_model_ensemble3",
    "nine_source_ensemble3",
    "all16_single_protocol",
    "all16_model_ensemble3",
    "hybrid_source_model_ensemble6",
)


def _parse_text_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("Comma-separated text list must not be empty.")
    return result


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("Comma-separated integer list must contain positive values.")
    return result


def _run_target(
    *,
    target_index: int,
    target: str,
    feature_cache: Path,
    output_root: Path,
    calibration_counts: tuple[int, ...],
    calibration_seeds: tuple[int, ...],
    gpu_count: int,
) -> tuple[str, int, Path]:
    part = output_root / "parts" / target
    part.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / f"{target}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    population_script = Path(__file__).with_name("katja_ensemble_population.py")
    command = [
        sys.executable,
        "-u",
        str(population_script),
        "--feature-cache",
        str(feature_cache),
        "--targets",
        target,
        "--calibration-counts",
        ",".join(str(value) for value in calibration_counts),
        "--calibration-seeds",
        ",".join(str(value) for value in calibration_seeds),
        "--output-dir",
        str(part),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(target_index % gpu_count)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return target, int(completed.returncode), log_path


def _combine_results(
    *,
    output_root: Path,
    targets: tuple[str, ...],
    calibration_counts: tuple[int, ...],
    calibration_seeds: tuple[int, ...],
    reference_k20: float,
    reference_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = [output_root / "parts" / target for target in targets]
    per_seed = pd.concat(
        [
            pd.read_csv(part / "katja_ensemble_population_per_seed.csv")
            for part in parts
        ],
        ignore_index=True,
    ).sort_values(["configuration", "target", "k", "seed"])

    observed_targets = set(per_seed["target"].astype(str).unique())
    observed_counts = set(per_seed["k"].astype(int).unique())
    observed_seeds = set(per_seed["seed"].astype(int).unique())
    observed_configurations = set(per_seed["configuration"].astype(str).unique())
    if observed_targets != set(targets):
        raise RuntimeError(
            f"Unexpected target registry: {sorted(observed_targets)!r}."
        )
    if observed_counts != set(calibration_counts):
        raise RuntimeError(
            f"Unexpected calibration-count registry: {sorted(observed_counts)!r}."
        )
    if observed_seeds != set(calibration_seeds):
        raise RuntimeError(
            f"Unexpected calibration-seed registry: {sorted(observed_seeds)!r}."
        )
    if observed_configurations != set(EXPECTED_CONFIGURATIONS):
        raise RuntimeError(
            "Unexpected ensemble configuration registry: "
            f"{sorted(observed_configurations)!r}."
        )
    expected_rows = (
        len(targets)
        * len(calibration_counts)
        * len(calibration_seeds)
        * len(EXPECTED_CONFIGURATIONS)
    )
    if per_seed.shape[0] != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} per-seed rows, got {per_seed.shape[0]}."
        )
    duplicate_columns = ["configuration", "target", "k", "seed"]
    if per_seed.duplicated(duplicate_columns).any():
        raise RuntimeError("Combined result contains duplicate configuration folds.")

    per_target, summary = _summarize_population(per_seed)
    primary_curve = summary[
        summary["configuration"] == PRIMARY_CONFIGURATION
    ].sort_values("k")
    if primary_curve.shape[0] != len(calibration_counts):
        raise RuntimeError("Primary curve is incomplete.")
    reproduced_k20 = float(
        primary_curve.loc[
            primary_curve["k"] == 20,
            "independent_accuracy_mean",
        ].iloc[0]
    )
    if not np.isclose(
        reproduced_k20,
        reference_k20,
        atol=reference_tolerance,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Full-curve k20 result does not reproduce the locked follow-up: "
            f"observed={reproduced_k20:.8f}, expected={reference_k20:.8f}."
        )

    paired_rows: list[dict[str, float | int]] = []
    for calibration_count in calibration_counts:
        primary = (
            per_target[
                (per_target["configuration"] == PRIMARY_CONFIGURATION)
                & (per_target["k"] == calibration_count)
            ]
            .set_index("target")["independent_accuracy"]
            .sort_index()
        )
        control = (
            per_target[
                (per_target["configuration"] == CONTROL_CONFIGURATION)
                & (per_target["k"] == calibration_count)
            ]
            .set_index("target")["independent_accuracy"]
            .sort_index()
        )
        if not primary.index.equals(control.index):
            raise RuntimeError("Primary and control target registries differ.")
        difference = primary - control
        paired_test = stats.ttest_rel(primary.to_numpy(), control.to_numpy())
        paired_rows.append(
            {
                "k": int(calibration_count),
                "n_targets": int(difference.shape[0]),
                "mean_primary_minus_control": float(difference.mean()),
                "wins": int((difference > 0.0).sum()),
                "ties": int((difference == 0.0).sum()),
                "losses": int((difference < 0.0).sum()),
                "paired_t_statistic": float(paired_test.statistic),
                "paired_t_pvalue": float(paired_test.pvalue),
            }
        )
    paired = pd.DataFrame(paired_rows)
    return per_seed, per_target, summary, primary_curve, paired


def run_full_curve(
    *,
    feature_cache: Path,
    output_root: Path,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    calibration_counts: tuple[int, ...] = DEFAULT_CALIBRATION_COUNTS,
    calibration_seeds: tuple[int, ...] = DEFAULT_CALIBRATION_SEEDS,
    workers: int = 8,
    gpu_count: int = 2,
    reference_k20: float = 0.657863,
    reference_tolerance: float = 5e-5,
) -> None:
    """Run target workers, combine them, and write validated curve artifacts."""

    if not feature_cache.is_file():
        raise FileNotFoundError(feature_cache)
    if "s05" in targets:
        raise ValueError("Development participant s05 must remain excluded.")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique.")
    expected_targets = set(DEFAULT_TARGETS)
    observed_targets = set(targets)
    if observed_targets != expected_targets:
        missing = sorted(expected_targets - observed_targets)
        unexpected = sorted(observed_targets - expected_targets)
        raise ValueError(
            "Locked full curve requires the exact target cohort; "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    if tuple(sorted(set(calibration_counts))) != tuple(calibration_counts):
        raise ValueError("calibration_counts must be unique and increasing.")
    if tuple(calibration_counts) != DEFAULT_CALIBRATION_COUNTS:
        raise ValueError(
            f"Locked full curve requires {DEFAULT_CALIBRATION_COUNTS!r}."
        )
    if tuple(calibration_seeds) != DEFAULT_CALIBRATION_SEEDS:
        raise ValueError(
            f"Locked calibration seeds require {DEFAULT_CALIBRATION_SEEDS!r}."
        )
    if workers < 1 or gpu_count < 1:
        raise ValueError("workers and gpu_count must be positive.")

    output_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_target,
                target_index=index,
                target=target,
                feature_cache=feature_cache,
                output_root=output_root,
                calibration_counts=calibration_counts,
                calibration_seeds=calibration_seeds,
                gpu_count=gpu_count,
            )
            for index, target in enumerate(targets)
        ]
        for future in as_completed(futures):
            target, returncode, log_path = future.result()
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            print(f"\n===== {target} =====\n{log_text}", flush=True)
            if returncode != 0:
                failures.append(f"{target}: exit {returncode}; log={log_path}")
    if failures:
        raise RuntimeError("Target workers failed: " + "; ".join(failures))

    per_seed, per_target, summary, primary_curve, paired = _combine_results(
        output_root=output_root,
        targets=targets,
        calibration_counts=calibration_counts,
        calibration_seeds=calibration_seeds,
        reference_k20=reference_k20,
        reference_tolerance=reference_tolerance,
    )
    per_seed.to_csv(output_root / "katja_ensemble_full_curve_per_seed.csv", index=False)
    per_target.to_csv(
        output_root / "katja_ensemble_full_curve_per_target.csv",
        index=False,
    )
    summary.to_csv(output_root / "katja_ensemble_full_curve_summary.csv", index=False)
    primary_curve.to_csv(output_root / "katja_ensemble_primary_curve.csv", index=False)
    paired.to_csv(output_root / "katja_ensemble_primary_vs_control.csv", index=False)

    part_metadata = [
        json.loads(
            (
                output_root
                / "parts"
                / target
                / "katja_ensemble_population_metadata.json"
            ).read_text(encoding="utf-8")
        )
        for target in targets
    ]
    metadata = {
        "analysis_status": "full_curve_follow_up_after_k20_ensemble_result",
        "development_run_id": 31007742231,
        "development_artifact_id": 8931215635,
        "k20_population_run_id": 31009907653,
        "k20_population_artifact_id": 8932702885,
        "primary_configuration": PRIMARY_CONFIGURATION,
        "control_configuration": CONTROL_CONFIGURATION,
        "targets": list(targets),
        "calibration_counts": list(calibration_counts),
        "calibration_seeds": list(calibration_seeds),
        "k20_reproduction": float(
            primary_curve.loc[
                primary_curve["k"] == 20,
                "independent_accuracy_mean",
            ].iloc[0]
        ),
        "workers": int(workers),
        "gpu_count": int(gpu_count),
        "parts": part_metadata,
        "workflow_run_id": (
            int(os.environ["GITHUB_RUN_ID"])
            if os.environ.get("GITHUB_RUN_ID")
            else None
        ),
    }
    (output_root / "katja_ensemble_full_curve_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("\nPrimary hybrid curve:\n", flush=True)
    print(primary_curve.to_string(index=False), flush=True)
    print("\nPrimary versus exact single-model control:\n", flush=True)
    print(paired.to_string(index=False), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument(
        "--calibration-counts",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_COUNTS),
    )
    parser.add_argument(
        "--calibration-seeds",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_SEEDS),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument("--reference-k20", type=float, default=0.657863)
    parser.add_argument("--reference-tolerance", type=float, default=5e-5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_full_curve(
        feature_cache=Path(args.feature_cache),
        output_root=Path(args.output_dir),
        targets=_parse_text_csv(args.targets),
        calibration_counts=_parse_int_csv(args.calibration_counts),
        calibration_seeds=_parse_int_csv(args.calibration_seeds),
        workers=args.workers,
        gpu_count=args.gpu_count,
        reference_k20=args.reference_k20,
        reference_tolerance=args.reference_tolerance,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
