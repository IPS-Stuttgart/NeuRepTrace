"""Combine worker outputs from the Katja physical-finger pseudo-target run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from katja_physical_finger_benchmark import (
    JULIA_K20_REFERENCE,
    METHOD_BLEND,
    METHOD_LOCAL,
    METHOD_PHYSICAL,
    METHODS,
)
from neureptrace._katja_finger_sequence_support import _mean_sem


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("Comma-separated values must not be empty.")
    return result


def combine(
    root: Path,
    *,
    expected_targets: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    paths = sorted(root.glob("worker*/katja_physical_per_seed.csv"))
    if not paths:
        raise FileNotFoundError(f"No worker per-seed CSV files found under {root}.")
    per_seed = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    duplicate_columns = ["method", "target", "seed", "k"]
    if per_seed.duplicated(duplicate_columns).any():
        duplicates = per_seed.loc[
            per_seed.duplicated(duplicate_columns, keep=False),
            duplicate_columns,
        ]
        raise ValueError(f"Duplicate worker results found:\n{duplicates}")

    observed_targets = tuple(sorted(per_seed["target"].unique().tolist()))
    if observed_targets != tuple(sorted(expected_targets)):
        raise ValueError(
            f"Worker target coverage {observed_targets!r} does not match "
            f"{tuple(sorted(expected_targets))!r}."
        )
    expected_keys = {
        (method, target, int(seed))
        for method in METHODS
        for target in expected_targets
        for seed in expected_seeds
    }
    observed_keys = {
        (str(row.method), str(row.target), int(row.seed))
        for row in per_seed.itertuples(index=False)
    }
    if observed_keys != expected_keys:
        missing = sorted(expected_keys.difference(observed_keys))
        extra = sorted(observed_keys.difference(expected_keys))
        raise ValueError(
            f"Incomplete worker coverage; missing={missing[:10]!r}, extra={extra[:10]!r}."
        )

    per_seed = per_seed.sort_values(["method", "target", "seed"]).reset_index(
        drop=True
    )
    per_target = (
        per_seed.groupby(["method", "target", "k"], as_index=False)
        .agg(
            independent_accuracy=("independent_accuracy", "mean"),
            permutation_accuracy=("permutation_accuracy", "mean"),
            n_seeds=("seed", "nunique"),
            n_evaluation_trials=("n_evaluation_trials", "first"),
            n_evaluation_events=("n_evaluation_events", "first"),
            target_fixed_physical_code=("target_fixed_physical_code", "first"),
        )
        .sort_values(["method", "target"])
        .reset_index(drop=True)
    )

    summary_rows: list[dict[str, Any]] = []
    for method, frame in per_target.groupby("method", sort=True):
        independent_mean, independent_sem = _mean_sem(
            frame["independent_accuracy"].to_numpy()
        )
        permutation_mean, permutation_sem = _mean_sem(
            frame["permutation_accuracy"].to_numpy()
        )
        summary_rows.append(
            {
                "method": method,
                "k": int(frame["k"].iloc[0]),
                "n_targets": int(frame.shape[0]),
                "independent_accuracy_mean": independent_mean,
                "independent_accuracy_sem": independent_sem,
                "permutation_accuracy_mean": permutation_mean,
                "permutation_accuracy_sem": permutation_sem,
                "independent_delta_vs_julia": independent_mean
                - JULIA_K20_REFERENCE,
                "independent_outperforms_julia": bool(
                    independent_mean > JULIA_K20_REFERENCE
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)

    local = per_target[per_target["method"] == METHOD_LOCAL].set_index("target")
    paired_rows: list[dict[str, Any]] = []
    target_delta_rows: list[pd.DataFrame] = []
    for method in (METHOD_PHYSICAL, METHOD_BLEND):
        candidate = per_target[per_target["method"] == method].set_index("target")
        aligned = local.join(
            candidate,
            how="inner",
            lsuffix="_local",
            rsuffix="_candidate",
        )
        differences = (
            aligned["independent_accuracy_candidate"]
            - aligned["independent_accuracy_local"]
        ).to_numpy(dtype=float)
        mean_delta, sem_delta = _mean_sem(differences)
        critical = float(stats.t.ppf(0.975, differences.size - 1))
        lower = mean_delta - critical * sem_delta
        upper = mean_delta + critical * sem_delta
        p_value = float(stats.ttest_1samp(differences, 0.0).pvalue)
        tolerance = 1e-12
        paired_rows.append(
            {
                "candidate_method": method,
                "reference_method": METHOD_LOCAL,
                "n_targets": int(differences.size),
                "mean_independent_delta": mean_delta,
                "sem_independent_delta": sem_delta,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "p_value_two_sided": p_value,
                "wins": int(np.count_nonzero(differences > tolerance)),
                "ties": int(np.count_nonzero(np.abs(differences) <= tolerance)),
                "losses": int(np.count_nonzero(differences < -tolerance)),
                "predeclared_success": bool(mean_delta > 0.0),
            }
        )
        delta = aligned.reset_index()[
            ["target", "target_fixed_physical_code_local"]
        ].copy()
        delta["candidate_method"] = method
        delta["independent_delta"] = differences
        delta = delta.rename(
            columns={
                "target_fixed_physical_code_local": "target_fixed_physical_code"
            }
        )
        target_delta_rows.append(delta)
    paired = pd.DataFrame(paired_rows)
    target_deltas = pd.concat(target_delta_rows, ignore_index=True)

    subgroup_rows: list[dict[str, Any]] = []
    for (method, fixed_code), frame in target_deltas.groupby(
        ["candidate_method", "target_fixed_physical_code"],
        sort=True,
    ):
        mean_delta, sem_delta = _mean_sem(frame["independent_delta"].to_numpy())
        subgroup_rows.append(
            {
                "candidate_method": method,
                "target_fixed_physical_code": fixed_code,
                "n_targets": int(frame.shape[0]),
                "mean_independent_delta": mean_delta,
                "sem_independent_delta": sem_delta,
            }
        )
    subgroup = pd.DataFrame(subgroup_rows)

    worker_metadata: dict[str, Any] = {}
    for path in sorted(root.glob("worker*/katja_physical_metadata.json")):
        worker_metadata[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    metadata = {
        "analysis_status": "combined_frozen_structural_outer_pseudo_target_evaluation",
        "expected_targets": list(expected_targets),
        "expected_seeds": list(expected_seeds),
        "methods": list(METHODS),
        "primary_candidate": METHOD_BLEND,
        "reference_method": METHOD_LOCAL,
        "success_rule": (
            "mean paired independent k20 accuracy of local_physical_blend6 "
            "exceeds local_ensemble3 across all 17 outer targets"
        ),
        "worker_result_files": [str(path) for path in paths],
        "worker_metadata": worker_metadata,
    }
    return per_seed, per_target, summary, paired, subgroup, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = _parse_csv(args.targets)
    seeds = tuple(int(value) for value in _parse_csv(args.seeds))
    per_seed, per_target, summary, paired, subgroup, metadata = combine(
        root,
        expected_targets=targets,
        expected_seeds=seeds,
    )
    per_seed.to_csv(output / "katja_physical_per_seed.csv", index=False)
    per_target.to_csv(output / "katja_physical_per_target.csv", index=False)
    summary.to_csv(output / "katja_physical_summary.csv", index=False)
    paired.to_csv(output / "katja_physical_paired.csv", index=False)
    subgroup.to_csv(output / "katja_physical_fixed_code_subgroups.csv", index=False)
    (output / "katja_physical_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("\n=== Combined summary ===")
    print(summary.to_string(index=False))
    print("\n=== Paired comparisons ===")
    print(paired.to_string(index=False))
    print("\n=== Fixed-code subgroup deltas ===")
    print(subgroup.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
