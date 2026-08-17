"""Calibration-prior matching for saved Katja independent-window predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neureptrace.decoding.katja_window_structure import (
    compose_template_order_finger_probabilities,
    learn_finger_templates,
    match_probability_marginals,
)
from neureptrace.katja_julia_window_benchmark import (
    DEFAULT_SEEDS,
    JULIA_SUBJECTS,
    _append_row,
    _load_npz_metadata,
    _metric_row,
    _parse_csv,
    _utc_timestamp,
    _write_status,
    relabel_minimum_overlap,
    select_nested_trial_splits,
)
from neureptrace.katja_window_duration_screen import _load_ensemble, _parse_numbers
from neureptrace.katja_window_accuracy_push import _composite_trial_ids


BASELINE_METHODS = {
    "trial_transformer_offline": "trial_transformer_ensemble",
    "hierarchical_tcn": "hierarchical_tcn_ensemble",
    "hybrid": "hybrid_ensemble",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summaries(rows: pd.DataFrame, output_dir: Path, baseline_path: Path) -> None:
    metrics = [
        "accuracy_raw_labels",
        "balanced_accuracy_raw_labels",
        "press_only_finger_accuracy",
        "rest_recall",
        "press_detection_accuracy",
        "trial_macro_accuracy_raw_labels",
    ]
    identity = ["family", "k_trials_per_sequence", "target"]
    subject = rows.groupby(identity, as_index=False)[metrics].mean()
    subject.to_csv(output_dir / "subject_seed_averages.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for keys, frame in subject.groupby(identity[:-1], sort=True):
        family, k = keys
        row: dict[str, Any] = {
            "family": family,
            "k_trials_per_sequence": int(k),
            "n_subjects": int(frame["target"].nunique()),
        }
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"sem_{metric}"] = (
                float(np.std(values, ddof=1) / math.sqrt(values.size))
                if values.size > 1
                else float("nan")
            )
        summary_rows.append(row)
    pd.DataFrame(summary_rows).sort_values(
        ["k_trials_per_sequence", "mean_accuracy_raw_labels"],
        ascending=[True, False],
    ).to_csv(output_dir / "summary_subject_sem.csv", index=False)

    baseline = pd.read_csv(baseline_path)
    baseline["family"] = baseline["method"].map(
        {method: family for family, method in BASELINE_METHODS.items()}
    )
    baseline = baseline[baseline["family"].notna()][
        ["target", "family", "split_seed", "k_trials_per_sequence", "accuracy_raw_labels"]
    ].rename(columns={"accuracy_raw_labels": "baseline_accuracy_raw_labels"})
    paired = rows.merge(
        baseline,
        on=["target", "family", "split_seed", "k_trials_per_sequence"],
        how="left",
        validate="one_to_one",
    )
    paired["delta_vs_independent_ensemble"] = (
        paired["accuracy_raw_labels"] - paired["baseline_accuracy_raw_labels"]
    )
    paired.to_csv(output_dir / "paired_fold_deltas.csv", index=False)
    paired.groupby(identity, as_index=False)[
        ["accuracy_raw_labels", "baseline_accuracy_raw_labels", "delta_vs_independent_ensemble"]
    ].mean().to_csv(output_dir / "paired_subject_deltas.csv", index=False)


def run_prior_match(args: argparse.Namespace) -> Path:
    if not 0.0 <= float(args.template_order_blend) <= 1.0:
        raise ValueError("template_order_blend must be in [0, 1]")
    if not np.isfinite(args.prior_pseudocount) or float(args.prior_pseudocount) <= 0.0:
        raise ValueError("prior_pseudocount must be finite and positive")
    protocol_category = (
        "2+3 transductive" if args.apply_prior_match else "3 calibration-only"
    )
    cache = Path(args.cache).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    partial_path = output_dir / "fold_results.partial.csv"
    metadata = _load_npz_metadata(cache)
    raw_labels = metadata["finger_ids"].astype(np.int64)
    raw_order = metadata["press_order"].astype(np.int64)
    training_labels = relabel_minimum_overlap(
        raw_labels, metadata["press_overlap_fraction"], args.minimum_overlap
    )
    subject_ids = metadata["subject_indices"].astype(np.int64)
    sequence_ids = metadata["sequence_id"].astype(np.int64)
    trial_ids = metadata["trial_id"].astype(np.int64)
    composite_trials = _composite_trial_ids(subject_ids, trial_ids)
    targets = _parse_csv(args.targets)
    families = _parse_csv(args.families)
    split_seeds = tuple(int(value) for value in _parse_numbers(args.split_seeds, int))
    model_seeds = tuple(int(value) for value in _parse_numbers(args.model_seeds, int))
    requested_k = tuple(int(value) for value in _parse_numbers(args.k_values, int))
    feasibility = json.loads((results_root / "combined" / "feasibility.json").read_text())
    completed: set[tuple[str, str, int, int]] = set()
    if args.resume and partial_path.exists():
        existing = pd.read_csv(partial_path).drop_duplicates(
            ["target", "family", "split_seed", "k_trials_per_sequence"], keep="last"
        )
        existing.to_csv(partial_path, index=False)
        completed = set(
            zip(
                existing["target"].astype(str),
                existing["family"].astype(str),
                existing["split_seed"].astype(int),
                existing["k_trials_per_sequence"].astype(int),
                strict=True,
            )
        )
    _write_status(
        status_path,
        state="configured",
        protocol_category=protocol_category,
        uses_unlabeled_evaluation_features_for_adaptation=bool(args.apply_prior_match),
        evaluation_labels_used_for_adaptation=False,
    )
    expected = 0
    for target in targets:
        target_index = JULIA_SUBJECTS.index(target)
        target_global = np.flatnonzero(subject_ids == target_index)
        feasible = tuple(int(k) for k in feasibility[target]["feasible_k_values"])
        selected_k = tuple(k for k in requested_k if k in feasible)
        if not selected_k:
            continue
        target_sequence = sequence_ids[target_global]
        target_trials = trial_ids[target_global]
        target_raw = raw_labels[target_global]
        target_training = training_labels[target_global]
        for split_seed in split_seeds:
            splits = select_nested_trial_splits(
                target_sequence,
                target_trials,
                k_values=feasible,
                seed=split_seed,
                context=target,
                split_mode="fixed_max_complement",
            )
            for k in selected_k:
                split = splits[k]
                calibration_global = target_global[split.calibration_rows]
                evaluation_global = target_global[split.evaluation_rows]
                templates = (
                    learn_finger_templates(
                        raw_labels,
                        raw_order,
                        composite_trials,
                        calibration_indices=calibration_global,
                        evaluation_indices=evaluation_global,
                    )
                    if args.template_order_blend > 0.0
                    else None
                )
                counts = np.bincount(raw_labels[calibration_global], minlength=6).astype(float)
                target_prior = (counts + float(args.prior_pseudocount)) / (
                    counts.sum() + 6.0 * float(args.prior_pseudocount)
                )
                for family in families:
                    expected += 1
                    key = (target, family, split_seed, k)
                    if key in completed:
                        continue
                    _write_status(
                        status_path,
                        state="fold_start",
                        target=target,
                        family=family,
                        split_seed=int(split_seed),
                        k_trials_per_sequence=int(k),
                        n_completed=len(completed),
                    )
                    ensemble = _load_ensemble(
                        results_root,
                        family=family,
                        target=target,
                        split_seed=split_seed,
                        k=k,
                        model_seeds=model_seeds,
                    )
                    if not np.array_equal(ensemble["row_indices"], evaluation_global):
                        raise ValueError("Saved prediction rows do not match reconstructed evaluation rows")
                    base_probabilities = np.asarray(ensemble["probabilities"], dtype=np.float64)
                    if args.template_order_blend > 0.0:
                        required = {
                            "aux_press_probabilities",
                            "aux_order_probabilities",
                            "aux_template_probabilities",
                        }
                        missing = sorted(required - set(ensemble))
                        if missing or templates is None:
                            raise ValueError(
                                f"Family {family} cannot compose template/order probabilities: {missing}"
                            )
                        composed = compose_template_order_finger_probabilities(
                            ensemble["aux_press_probabilities"],
                            ensemble["aux_order_probabilities"],
                            ensemble["aux_template_probabilities"],
                            templates,
                        )
                        blend = float(args.template_order_blend)
                        base_probabilities = (1.0 - blend) * base_probabilities + blend * composed
                        base_probabilities /= np.maximum(
                            base_probabilities.sum(axis=1, keepdims=True), 1e-12
                        )
                    if args.apply_prior_match:
                        adjusted, biases = match_probability_marginals(
                            base_probabilities,
                            target_prior,
                            max_iterations=args.max_iterations,
                            tolerance=args.tolerance,
                            damping=args.damping,
                        )
                    else:
                        adjusted = base_probabilities
                        biases = np.ones(6, dtype=np.float64)
                    row = _metric_row(
                        method=f"{family}_calibration_prior_matched",
                        target=target,
                        target_index=target_index,
                        seed=split_seed,
                        split=split,
                        probabilities=adjusted,
                        raw_labels=target_raw,
                        training_labels=target_training,
                        target_trial_ids=target_trials,
                        n_source_windows=int(np.sum(subject_ids != target_index)),
                        n_source_subjects=len(JULIA_SUBJECTS) - 1,
                        adaptation_stages="saved_five_seed_ensemble,calibration_prior_matching",
                    )
                    row.update(
                        {
                            "family": family,
                            "split_seed": int(split_seed),
                            "protocol_category": protocol_category,
                            "uses_target_calibration_labels": True,
                            "uses_unlabeled_evaluation_features_for_adaptation": bool(
                                args.apply_prior_match
                            ),
                            "evaluation_labels_used_for_adaptation": False,
                            "calibration_evaluation_disjoint": True,
                            "prior_pseudocount": float(args.prior_pseudocount),
                            "template_order_blend": float(args.template_order_blend),
                            "candidate_role": str(args.candidate_role),
                            "target_prior": json.dumps(target_prior.tolist(), separators=(",", ":")),
                            "class_biases": json.dumps(biases.tolist(), separators=(",", ":")),
                        }
                    )
                    _append_row(partial_path, row)
                    completed.add(key)
                    print(
                        f"{target} split={split_seed} k={k} {family} "
                        f"prior_matched={row['accuracy_raw_labels']:.4f}",
                        flush=True,
                    )
    if not partial_path.exists():
        raise RuntimeError("No prior-matched rows were produced")
    rows = pd.read_csv(partial_path).drop_duplicates(
        ["target", "family", "split_seed", "k_trials_per_sequence"], keep="last"
    )
    rows.to_csv(output_dir / "fold_results.csv", index=False)
    _summaries(rows, output_dir, results_root / "combined" / "fold_results.csv")
    validation = {
        "all_required_checks_pass": bool(
            len(rows) == expected
            and rows["calibration_evaluation_disjoint"].astype(bool).all()
            and not rows["evaluation_labels_used_for_adaptation"].astype(bool).any()
        ),
        "n_expected_rows": int(expected),
        "n_observed_rows": int(len(rows)),
        "protocol_category": protocol_category,
        "strict_protocol3_calibration_only": not bool(args.apply_prior_match),
    }
    _atomic_json(output_dir / "validation.json", validation)
    _atomic_json(
        output_dir / "provenance.json",
        {
            "created_at": _utc_timestamp(),
            "cache": str(cache),
            "results_root": str(results_root),
            "targets": list(targets),
            "families": list(families),
            "k_values": list(requested_k),
            "split_seeds": list(split_seeds),
            "model_seeds": list(model_seeds),
            "adaptation": (
                "multiplicative class-bias fitting on unlabeled evaluation probabilities "
                "to match the raw-label prior estimated from disjoint calibration trials"
                if args.apply_prior_match
                else "calibration-template/order probability composition without batch adaptation"
            ),
            "template_order_blend": float(args.template_order_blend),
            "candidate_role": str(args.candidate_role),
        },
    )
    _write_status(
        status_path,
        state="done",
        n_rows=int(len(rows)),
        validation_passed=validation["all_required_checks_pass"],
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--targets", default=",".join(JULIA_SUBJECTS))
    parser.add_argument("--families", default="trial_transformer_offline,hierarchical_tcn,hybrid")
    parser.add_argument("--k-values", default="20")
    parser.add_argument("--split-seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--model-seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--minimum-overlap", type=float, default=0.2)
    parser.add_argument("--prior-pseudocount", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument(
        "--apply-prior-match", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--template-order-blend", type=float, default=0.0)
    parser.add_argument(
        "--candidate-role",
        choices=("predeclared_fixed", "exploratory"),
        default="predeclared_fixed",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_prior_match(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
