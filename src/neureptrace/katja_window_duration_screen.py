"""Leakage-guarded explicit-duration decoding of saved Katja window predictions."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from neureptrace.decoding.katja_window_structure import (
    ensemble_prediction_bundles,
    estimate_state_duration_priors,
    explicit_duration_trial_decode,
    learn_finger_templates,
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
from neureptrace.katja_window_accuracy_push import _composite_trial_ids


DEFAULT_FAMILIES = (
    "trial_transformer_offline",
    "hybrid",
    "duration_prior_only",
    "auxiliary_duration_prior",
)


def _parse_numbers(values: str, cast: type[int] | type[float]) -> tuple[int | float, ...]:
    parsed = tuple(cast(value.strip()) for value in values.split(",") if value.strip())
    if not parsed:
        raise ValueError("At least one numeric value is required")
    return parsed


def _candidate_id(candidate: dict[str, float]) -> str:
    canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _bundle_paths(
    results_root: Path,
    *,
    family: str,
    target: str,
    split_seed: int,
    k: int,
    model_seeds: Iterable[int],
) -> list[Path]:
    return [
        results_root
        / f"target={target}"
        / "predictions"
        / family
        / f"target={target}"
        / f"split_seed={split_seed}"
        / f"k={k}"
        / f"model_seed={model_seed}.npz"
        for model_seed in model_seeds
    ]


def _load_ensemble(
    results_root: Path,
    *,
    family: str,
    target: str,
    split_seed: int,
    k: int,
    model_seeds: tuple[int, ...],
) -> dict[str, Any]:
    if family != "hybrid":
        paths = _bundle_paths(
            results_root,
            family=family,
            target=target,
            split_seed=split_seed,
            k=k,
            model_seeds=model_seeds,
        )
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {family} prediction bundles: {missing[:3]}")
        return ensemble_prediction_bundles(paths)

    tcn = _load_ensemble(
        results_root,
        family="hierarchical_tcn",
        target=target,
        split_seed=split_seed,
        k=k,
        model_seeds=model_seeds,
    )
    context = _load_ensemble(
        results_root,
        family="trial_transformer_offline",
        target=target,
        split_seed=split_seed,
        k=k,
        model_seeds=model_seeds,
    )
    if not np.array_equal(tcn["row_indices"], context["row_indices"]):
        raise ValueError("Hybrid families do not contain identical evaluation rows")
    result: dict[str, Any] = {
        "row_indices": tcn["row_indices"].copy(),
        "probabilities": 0.5 * (tcn["probabilities"] + context["probabilities"]),
        "split_seed": int(split_seed),
        "model_seeds": model_seeds,
    }
    for name in ("aux_order_probabilities", "aux_overlap_prediction"):
        if name in tcn and name in context:
            result[name] = 0.5 * (tcn[name] + context[name])
    if "aux_template_probabilities" in context:
        result["aux_template_probabilities"] = context["aux_template_probabilities"]
    return result


def _decode_evaluation_trials(
    ensemble: dict[str, Any],
    *,
    evaluation_rows: np.ndarray,
    composite_trial_ids: np.ndarray,
    templates: tuple[tuple[int, ...], ...],
    duration_priors,
    candidate: dict[str, float],
) -> np.ndarray:
    probabilities = np.asarray(ensemble["probabilities"], dtype=np.float64)
    if not np.array_equal(np.asarray(ensemble["row_indices"]), evaluation_rows):
        raise ValueError("Saved prediction rows do not match the reconstructed evaluation split")
    predicted = np.empty(evaluation_rows.size, dtype=np.int64)
    evaluation_trials = composite_trial_ids[evaluation_rows]
    order_probabilities = ensemble.get("aux_order_probabilities")
    overlap_probabilities = ensemble.get("aux_overlap_prediction")
    template_probabilities = ensemble.get("aux_template_probabilities")
    for trial in np.unique(evaluation_trials):
        local = np.flatnonzero(evaluation_trials == trial)
        predicted[local] = explicit_duration_trial_decode(
            probabilities[local],
            templates,
            duration_priors=duration_priors,
            order_probabilities=None if order_probabilities is None else order_probabilities[local],
            overlap_probabilities=None if overlap_probabilities is None else overlap_probabilities[local],
            template_probabilities=(
                None if template_probabilities is None else template_probabilities[local]
            ),
            order_weight=candidate["order_weight"],
            overlap_weight=candidate["overlap_weight"],
            template_weight=candidate["template_weight"],
            duration_weight=candidate["duration_weight"],
            press_duration_scale=candidate["press_duration_scale"],
            rest_duration_scale=candidate["rest_duration_scale"],
            rest_log_bias=candidate["rest_log_bias"],
        ).labels
    return predicted


def _summarize(rows: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        "accuracy_raw_labels",
        "balanced_accuracy_raw_labels",
        "accuracy_tau_labels",
        "press_only_finger_accuracy",
        "rest_recall",
        "press_detection_accuracy",
        "trial_macro_accuracy_raw_labels",
    ]
    identity = ["candidate_id", "family", "k_trials_per_sequence", "target"]
    subject = rows.groupby(identity, as_index=False)[metrics].mean()
    subject.to_csv(output_dir / "subject_seed_averages.csv", index=False)
    summary_rows: list[dict[str, Any]] = []
    for keys, frame in subject.groupby(identity[:-1], sort=True):
        candidate_id, family, k = keys
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
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
    summary = pd.DataFrame(summary_rows)
    candidate_columns = [
        "candidate_id",
        "calibration_prior_strength",
        "duration_weight",
        "press_duration_scale",
        "rest_duration_scale",
        "rest_log_bias",
        "order_weight",
        "overlap_weight",
        "template_weight",
        "candidate_role",
    ]
    summary = summary.merge(rows[candidate_columns].drop_duplicates(), on="candidate_id", how="left")
    summary.sort_values(
        ["k_trials_per_sequence", "family", "mean_accuracy_raw_labels"],
        ascending=[True, True, False],
    ).to_csv(output_dir / "summary_subject_sem.csv", index=False)

    baseline_path = rows.attrs.get("baseline_path")
    if baseline_path and Path(baseline_path).exists():
        baseline = pd.read_csv(baseline_path)
        baseline = baseline[
            baseline["method"].eq("trial_transformer_ensemble_structured")
        ][["target", "split_seed", "k_trials_per_sequence", "accuracy_raw_labels"]].rename(
            columns={"accuracy_raw_labels": "baseline_accuracy_raw_labels"}
        )
        paired = rows.merge(
            baseline,
            on=["target", "split_seed", "k_trials_per_sequence"],
            how="left",
            validate="many_to_one",
        )
        paired["delta_vs_geometric_structured"] = (
            paired["accuracy_raw_labels"] - paired["baseline_accuracy_raw_labels"]
        )
        paired.to_csv(output_dir / "paired_fold_deltas.csv", index=False)
        paired_subject = paired.groupby(identity, as_index=False)[
            ["accuracy_raw_labels", "baseline_accuracy_raw_labels", "delta_vs_geometric_structured"]
        ].mean()
        paired_subject.to_csv(output_dir / "paired_subject_deltas.csv", index=False)


def run_duration_screen(args: argparse.Namespace) -> Path:
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
    candidates: list[dict[str, float]] = []
    for values in itertools.product(
        _parse_numbers(args.calibration_prior_strengths, float),
        _parse_numbers(args.duration_weights, float),
        _parse_numbers(args.press_duration_scales, float),
        _parse_numbers(args.rest_duration_scales, float),
        _parse_numbers(args.rest_log_biases, float),
        _parse_numbers(args.order_weights, float),
        _parse_numbers(args.overlap_weights, float),
        _parse_numbers(args.template_weights, float),
    ):
        candidate = dict(
            zip(
                (
                    "calibration_prior_strength",
                    "duration_weight",
                    "press_duration_scale",
                    "rest_duration_scale",
                    "rest_log_bias",
                    "order_weight",
                    "overlap_weight",
                    "template_weight",
                ),
                (float(value) for value in values),
                strict=True,
            )
        )
        candidate["candidate_id"] = _candidate_id(candidate)
        candidates.append(candidate)
    candidate_role = "predeclared_fixed" if len(candidates) == 1 else "exploratory_grid"
    completed: set[tuple[str, str, int, int, str]] = set()
    if args.resume and partial_path.exists():
        existing = pd.read_csv(partial_path).drop_duplicates(
            ["target", "family", "split_seed", "k_trials_per_sequence", "candidate_id"],
            keep="last",
        )
        existing.to_csv(partial_path, index=False)
        completed = set(
            zip(
                existing["target"].astype(str),
                existing["family"].astype(str),
                existing["split_seed"].astype(int),
                existing["k_trials_per_sequence"].astype(int),
                existing["candidate_id"].astype(str),
                strict=True,
            )
        )
    _write_status(
        status_path,
        state="configured",
        n_candidates=len(candidates),
        candidate_role=candidate_role,
        evaluation_labels_used_for_fitting=False,
        evaluation_labels_used_for_candidate_selection=False,
    )
    expected = 0
    for target in targets:
        target_index = JULIA_SUBJECTS.index(target)
        target_global = np.flatnonzero(subject_ids == target_index)
        source_global = np.flatnonzero(subject_ids != target_index)
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
                planned_keys = {
                    (target, family, split_seed, k, str(candidate["candidate_id"]))
                    for candidate in candidates
                    for family in families
                }
                if planned_keys.issubset(completed):
                    expected += len(planned_keys)
                    continue
                evaluation_global = target_global[split.evaluation_rows]
                calibration_global = target_global[split.calibration_rows]
                templates = learn_finger_templates(
                    raw_labels,
                    raw_order,
                    composite_trials,
                    calibration_indices=calibration_global,
                    evaluation_indices=evaluation_global,
                )
                ensembles: dict[str, dict[str, Any]] = {}
                for family in families:
                    if family == "duration_prior_only":
                        ensembles[family] = {
                            "row_indices": evaluation_global.copy(),
                            "probabilities": np.full(
                                (evaluation_global.size, 6), 1.0 / 6.0, dtype=np.float64
                            ),
                            "non_neural_task_prior_control": True,
                        }
                        continue
                    if family == "auxiliary_duration_prior":
                        auxiliary = _load_ensemble(
                            results_root,
                            family="trial_transformer_offline",
                            target=target,
                            split_seed=split_seed,
                            k=k,
                            model_seeds=model_seeds,
                        )
                        auxiliary["probabilities"] = np.full(
                            (evaluation_global.size, 6), 1.0 / 6.0, dtype=np.float64
                        )
                        auxiliary["non_neural_task_prior_control"] = False
                        auxiliary["finger_probabilities_replaced_with_uniform"] = True
                        ensembles[family] = auxiliary
                        continue
                    ensembles[family] = _load_ensemble(
                        results_root,
                        family=family,
                        target=target,
                        split_seed=split_seed,
                        k=k,
                        model_seeds=model_seeds,
                    )
                for candidate in candidates:
                    priors = estimate_state_duration_priors(
                        raw_labels,
                        raw_order,
                        composite_trials,
                        source_indices=source_global,
                        calibration_indices=calibration_global,
                        evaluation_indices=evaluation_global,
                        calibration_prior_strength=candidate["calibration_prior_strength"],
                    )
                    for family, ensemble in ensembles.items():
                        expected += 1
                        key = (target, family, split_seed, k, str(candidate["candidate_id"]))
                        if key in completed:
                            continue
                        _write_status(
                            status_path,
                            state="fold_start",
                            target=target,
                            family=family,
                            split_seed=int(split_seed),
                            k_trials_per_sequence=int(k),
                            candidate_id=candidate["candidate_id"],
                            n_completed=len(completed),
                        )
                        predicted = _decode_evaluation_trials(
                            ensemble,
                            evaluation_rows=evaluation_global,
                            composite_trial_ids=composite_trials,
                            templates=templates,
                            duration_priors=priors,
                            candidate=candidate,
                        )
                        row = _metric_row(
                            method=f"{family}_explicit_duration",
                            target=target,
                            target_index=target_index,
                            seed=split_seed,
                            split=split,
                            probabilities=ensemble["probabilities"],
                            raw_labels=target_raw,
                            training_labels=target_training,
                            target_trial_ids=target_trials,
                            n_source_windows=int(source_global.size),
                            n_source_subjects=len(JULIA_SUBJECTS) - 1,
                            adaptation_stages="saved_five_seed_ensemble,explicit_duration_hsmm",
                            predicted_labels=predicted,
                        )
                        row.update(
                            {
                                "family": family,
                                "split_seed": int(split_seed),
                                "candidate_id": candidate["candidate_id"],
                                **candidate,
                                "candidate_role": candidate_role,
                                "duration_endpoint": "raw_six_class_scoring_labels",
                                "duration_source_trial_count": priors.source_trial_count,
                                "duration_calibration_trial_count": priors.calibration_trial_count,
                                "duration_calibration_weight": priors.calibration_weight,
                                "evaluation_labels_used_for_fitting": False,
                                "evaluation_labels_used_for_candidate_selection": False,
                                "calibration_evaluation_disjoint": True,
                                "non_neural_task_prior_control": bool(
                                    ensemble.get("non_neural_task_prior_control", False)
                                ),
                                "finger_probabilities_replaced_with_uniform": bool(
                                    ensemble.get("finger_probabilities_replaced_with_uniform", False)
                                ),
                            }
                        )
                        _append_row(partial_path, row)
                        completed.add(key)
                        print(
                            f"{target} split={split_seed} k={k} {family} "
                            f"candidate={candidate['candidate_id']} accuracy={row['accuracy_raw_labels']:.4f}",
                            flush=True,
                        )
    if not partial_path.exists():
        raise RuntimeError("No explicit-duration rows were produced")
    rows = pd.read_csv(partial_path).drop_duplicates(
        ["target", "family", "split_seed", "k_trials_per_sequence", "candidate_id"],
        keep="last",
    )
    rows.to_csv(output_dir / "fold_results.csv", index=False)
    rows.attrs["baseline_path"] = str(results_root / "combined" / "fold_results.csv")
    _summarize(rows, output_dir)
    validation = {
        "all_required_checks_pass": bool(
            len(rows) == expected
            and rows["calibration_evaluation_disjoint"].astype(bool).all()
            and not rows["evaluation_labels_used_for_fitting"].astype(bool).any()
            and not rows["evaluation_labels_used_for_candidate_selection"].astype(bool).any()
        ),
        "n_expected_rows": int(expected),
        "n_observed_rows": int(len(rows)),
        "candidate_role": candidate_role,
        "posthoc_candidate_ranking_is_confirmatory": False if len(candidates) > 1 else True,
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
            "candidates": candidates,
            "candidate_role": candidate_role,
            "duration_fit": (
                "raw six-class source labels plus held-out-target calibration labels; "
                "held-out-target evaluation labels excluded"
            ),
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
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--k-values", default="20")
    parser.add_argument("--split-seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--model-seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--minimum-overlap", type=float, default=0.2)
    parser.add_argument("--calibration-prior-strengths", default="8")
    parser.add_argument("--duration-weights", default="1")
    parser.add_argument("--press-duration-scales", default="1")
    parser.add_argument("--rest-duration-scales", default="1")
    parser.add_argument("--rest-log-biases", default="0")
    parser.add_argument("--order-weights", default="0.35")
    parser.add_argument("--overlap-weights", default="0.15")
    parser.add_argument("--template-weights", default="0.15")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_duration_screen(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
