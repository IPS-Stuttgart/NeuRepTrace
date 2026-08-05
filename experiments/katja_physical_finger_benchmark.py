"""Evaluate a frozen global physical-finger head through outer pseudo-target folds.

This experiment compares three same-source-budget methods at k20:

* the established three-model participant-local ensemble;
* three models trained on the five global physical finger identities, with the
  target participant's fixed first finger masked at inference;
* an equal-probability blend of the three local and three physical models.

No hyperparameter grid is searched.  Every requested participant is treated as
an outer pseudo-target, and evaluation labels are read only after all member
probabilities have been produced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from katja_ensemble_source_sweep import _model_kwargs, _model_seed
from neureptrace._katja_finger_sequence_support import (
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_PARTICIPANTS,
    _composite_trial_ids,
    _constant_trial_values,
    _fit_source_preprocessor,
    _mean_sem,
    _stable_source_selection,
    _transform_features,
    derive_participant_local_finger_labels,
    load_katja_feature_cache,
)
from neureptrace.decoding.progressive_sequence_finetune import (
    TorchProgressiveSequenceClassifier,
    pack_complete_trial_events,
    permutation_constrained_decode,
    select_nested_trial_calibration_splits,
)
from neureptrace.katja_physical_finger import (
    infer_global_physical_codes,
    participant_physical_finger_maps,
    physical_probabilities_to_local,
)

CALIBRATION_COUNT = 20
JULIA_K20_REFERENCE = 0.594
METHOD_LOCAL = "local_ensemble3"
METHOD_PHYSICAL = "physical_ensemble3_masked"
METHOD_BLEND = "local_physical_blend6"
METHODS = (METHOD_LOCAL, METHOD_PHYSICAL, METHOD_BLEND)


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("Comma-separated values must not be empty.")
    return result


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in _parse_csv(value))
    if any(item < 0 for item in result):
        raise ValueError("Seeds must be non-negative integers.")
    return result


def _prepare_rows(cache: dict[str, np.ndarray]) -> dict[str, Any]:
    subjects = cache["subjects"].astype(str)
    press_positions = cache["press_positions"].astype(int)
    included = (
        cache["correct_order"].astype(bool)
        & np.isin(press_positions, np.asarray((2, 3, 4, 5)))
        & np.isin(subjects, np.asarray(DEFAULT_PARTICIPANTS))
    )
    local_labels = derive_participant_local_finger_labels(
        subjects,
        cache["finger_codes"],
        included_mask=included,
        expected_classes=4,
    )
    physical_labels = np.asarray(cache["finger_codes"])
    global_codes = infer_global_physical_codes(
        physical_labels,
        included_mask=included,
        expected_codes=5,
    )
    physical_maps = participant_physical_finger_maps(
        subjects,
        physical_labels,
        included_mask=included,
        global_codes=global_codes,
        expected_variable_codes=4,
    )
    return {
        "features": cache["features"][included],
        "subjects": subjects[included],
        "trial_ids": cache["trial_ids"][included],
        "press_positions": press_positions[included],
        "sequence_ids": cache["sequence_ids"][included],
        "local_labels": local_labels[included],
        "physical_labels": physical_labels[included],
        "global_codes": global_codes,
        "physical_maps": physical_maps,
    }


def _prepare_target_space(
    rows: dict[str, Any],
    *,
    target: str,
) -> dict[str, Any]:
    subjects = rows["subjects"].astype(str)
    target_mask = subjects == target
    if not np.any(target_mask):
        raise ValueError(f"Target participant {target!r} has no retained rows.")
    selected_sources = _stable_source_selection(
        DEFAULT_PARTICIPANTS,
        target=target,
        n_sources=9,
        seed=13,
    )
    source_mask = np.isin(subjects, np.asarray(selected_sources))
    scaler, pca, source_transformed = _fit_source_preprocessor(
        rows["features"][source_mask],
        pca_components=64,
    )
    target_transformed = _transform_features(
        rows["features"][target_mask],
        scaler=scaler,
        pca=pca,
    )

    source_local = pack_complete_trial_events(
        source_transformed,
        _composite_trial_ids(
            subjects[source_mask],
            rows["trial_ids"][source_mask],
        ),
        rows["press_positions"][source_mask],
        labels=rows["local_labels"][source_mask],
        expected_events=4,
        require_permutation_labels=True,
    )
    target_local = pack_complete_trial_events(
        target_transformed,
        rows["trial_ids"][target_mask],
        rows["press_positions"][target_mask],
        labels=rows["local_labels"][target_mask],
        expected_events=4,
        require_permutation_labels=True,
    )
    if source_local.labels is None or target_local.labels is None:
        raise RuntimeError("Local labels are required for this benchmark.")

    source_physical_rows = rows["physical_labels"][source_mask]
    target_physical_rows = rows["physical_labels"][target_mask]
    source_physical = source_physical_rows[source_local.row_indices]
    target_physical = target_physical_rows[target_local.row_indices]
    source_physical_codes = tuple(np.unique(source_physical).tolist())
    if set(source_physical_codes) != set(rows["global_codes"]):
        raise ValueError(
            f"Target {target!r}'s fixed nine-source set does not cover all five "
            f"physical codes: {source_physical_codes!r}."
        )

    source_subjects = _constant_trial_values(
        subjects[source_mask],
        source_local.row_indices,
        name="source subject",
    )
    target_strata = _constant_trial_values(
        rows["sequence_ids"][target_mask],
        target_local.row_indices,
        name="target sequence ID",
    )
    target_map = rows["physical_maps"][target]
    return {
        "selected_sources": selected_sources,
        "source_features": source_local.features,
        "source_local_labels": source_local.labels,
        "source_physical_labels": source_physical,
        "source_subjects": source_subjects,
        "target_features": target_local.features,
        "target_local_labels": target_local.labels,
        "target_physical_labels": target_physical,
        "target_trial_ids": target_local.trial_ids,
        "target_strata": target_strata,
        "target_variable_codes": target_map.variable_codes,
        "target_fixed_code": target_map.fixed_code,
        "global_codes": rows["global_codes"],
    }


def _local_model(model_seed: int) -> TorchProgressiveSequenceClassifier:
    return TorchProgressiveSequenceClassifier(**_model_kwargs(model_seed))


def _physical_model(model_seed: int) -> TorchProgressiveSequenceClassifier:
    kwargs = _model_kwargs(model_seed)
    kwargs.update(
        {
            "enforce_permutation_labels": False,
            "sinkhorn_loss_weight": 0.0,
            "assignment_loss_weight": 0.0,
        }
    )
    return TorchProgressiveSequenceClassifier(**kwargs)


def _ordered_local_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    ordered_indices: list[int] = []
    for label in range(4):
        matches = np.flatnonzero(classes == label)
        if matches.size != 1:
            raise ValueError(f"Local class {label} is not represented exactly once.")
        ordered_indices.append(int(matches[0]))
    return probabilities[:, :, ordered_indices]


def _evaluate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    independent = np.argmax(probabilities, axis=2)
    constrained = permutation_constrained_decode(
        probabilities,
        temperature=0.5,
        sinkhorn_iterations=20,
    ).assignments
    return (
        float(np.mean(independent == labels)),
        float(np.mean(constrained == labels)),
    )


def run_targets(
    cache: dict[str, np.ndarray],
    *,
    targets: tuple[str, ...],
    calibration_seeds: tuple[int, ...] = DEFAULT_CALIBRATION_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    unknown = [target for target in targets if target not in DEFAULT_PARTICIPANTS]
    if unknown:
        raise ValueError(f"Unknown target participants: {unknown!r}.")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique.")
    rows = _prepare_rows(cache)
    result_rows: list[dict[str, Any]] = []
    target_metadata: dict[str, Any] = {}

    for target in targets:
        target_started = time.time()
        space = _prepare_target_space(rows, target=target)
        splits = {
            int(seed): select_nested_trial_calibration_splits(
                space["target_strata"],
                calibration_counts=(CALIBRATION_COUNT,),
                max_per_stratum=CALIBRATION_COUNT,
                min_evaluation_per_stratum=1,
                seed=int(seed),
                context=("katja_finger", target),
            )[CALIBRATION_COUNT]
            for seed in calibration_seeds
        }

        for calibration_seed in calibration_seeds:
            split = splits[int(calibration_seed)]
            local_members: list[np.ndarray] = []
            physical_members: list[np.ndarray] = []
            for seed_variant in range(3):
                model_seed = _model_seed(calibration_seed, seed_variant)

                local_model = _local_model(model_seed)
                local_model.fit_source(
                    space["source_features"],
                    space["source_local_labels"],
                    source_subjects=space["source_subjects"],
                )
                local_model.adapt_target(
                    space["target_features"][split.calibration_indices],
                    space["target_local_labels"][split.calibration_indices],
                    target_strata=space["target_strata"][split.calibration_indices],
                )
                local_raw = local_model.predict_proba(
                    space["target_features"][split.evaluation_indices],
                    constrained=False,
                )
                local_members.append(
                    _ordered_local_probabilities(local_raw, local_model.classes_)
                )
                del local_model
                torch.cuda.empty_cache()

                physical_model = _physical_model(model_seed)
                physical_model.fit_source(
                    space["source_features"],
                    space["source_physical_labels"],
                    source_subjects=space["source_subjects"],
                )
                if set(physical_model.classes_.tolist()) != set(space["global_codes"]):
                    raise RuntimeError(
                        f"Physical model for {target} does not expose all global classes."
                    )
                physical_model.adapt_target(
                    space["target_features"][split.calibration_indices],
                    space["target_physical_labels"][split.calibration_indices],
                    target_strata=space["target_strata"][split.calibration_indices],
                )
                physical_raw = physical_model.predict_proba(
                    space["target_features"][split.evaluation_indices],
                    constrained=False,
                )
                physical_members.append(
                    physical_probabilities_to_local(
                        physical_raw,
                        model_classes=physical_model.classes_,
                        variable_codes=space["target_variable_codes"],
                    )
                )
                del physical_model
                torch.cuda.empty_cache()

            local_probabilities = np.mean(np.stack(local_members, axis=0), axis=0)
            physical_probabilities = np.mean(
                np.stack(physical_members, axis=0),
                axis=0,
            )
            blend_probabilities = np.mean(
                np.stack((*local_members, *physical_members), axis=0),
                axis=0,
            )
            evaluation_labels = space["target_local_labels"][split.evaluation_indices]
            method_probabilities = {
                METHOD_LOCAL: local_probabilities,
                METHOD_PHYSICAL: physical_probabilities,
                METHOD_BLEND: blend_probabilities,
            }
            for method, probabilities in method_probabilities.items():
                independent_accuracy, permutation_accuracy = _evaluate_probabilities(
                    probabilities,
                    evaluation_labels,
                )
                result_rows.append(
                    {
                        "target": target,
                        "seed": int(calibration_seed),
                        "k": CALIBRATION_COUNT,
                        "method": method,
                        "n_models": 6 if method == METHOD_BLEND else 3,
                        "n_source_participants": 9,
                        "selected_sources": ",".join(space["selected_sources"]),
                        "target_fixed_physical_code": space["target_fixed_code"],
                        "target_variable_physical_codes": ",".join(
                            str(code) for code in space["target_variable_codes"]
                        ),
                        "n_evaluation_trials": int(split.evaluation_indices.size),
                        "n_evaluation_events": int(evaluation_labels.size),
                        "independent_accuracy": independent_accuracy,
                        "permutation_accuracy": permutation_accuracy,
                    }
                )

        target_metadata[target] = {
            "selected_sources": list(space["selected_sources"]),
            "fixed_physical_code": str(space["target_fixed_code"]),
            "variable_physical_codes": [
                str(code) for code in space["target_variable_codes"]
            ],
            "n_target_trials": int(space["target_features"].shape[0]),
            "elapsed_seconds": time.time() - target_started,
        }
        print(
            f"Completed {target} in "
            f"{target_metadata[target]['elapsed_seconds']:.1f} s"
        )

    per_seed = pd.DataFrame(result_rows).sort_values(
        ["method", "target", "seed"]
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
                "k": CALIBRATION_COUNT,
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

    local_target = per_target[per_target["method"] == METHOD_LOCAL].set_index(
        "target"
    )
    paired_rows: list[dict[str, Any]] = []
    for method in (METHOD_PHYSICAL, METHOD_BLEND):
        candidate = per_target[per_target["method"] == method].set_index("target")
        aligned = local_target.join(
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
        if differences.size > 1:
            critical = float(stats.t.ppf(0.975, differences.size - 1))
            lower = mean_delta - critical * sem_delta
            upper = mean_delta + critical * sem_delta
            p_value = float(stats.ttest_1samp(differences, 0.0).pvalue)
        else:
            lower = mean_delta
            upper = mean_delta
            p_value = float("nan")
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
    paired = pd.DataFrame(paired_rows)

    metadata = {
        "analysis_status": "frozen_structural_outer_pseudo_target_evaluation",
        "candidate_selection": (
            "one global five-physical-finger head selected from a label-semantics "
            "audit; no performance hyperparameter sweep"
        ),
        "targets": list(targets),
        "calibration_count": CALIBRATION_COUNT,
        "calibration_seeds": list(calibration_seeds),
        "source_participants_per_target": 9,
        "source_selection_seed": 13,
        "model_seed_variants": [0, 1, 2],
        "methods": list(METHODS),
        "primary_candidate": METHOD_BLEND,
        "reference_method": METHOD_LOCAL,
        "success_rule": (
            "mean paired independent k20 accuracy of local_physical_blend6 "
            "exceeds local_ensemble3 across outer targets"
        ),
        "global_physical_codes": [str(code) for code in rows["global_codes"]],
        "physical_training_constraint": (
            "five-class physical cross-entropy; no permutation loss; target fixed "
            "finger masked and probabilities renormalized at inference"
        ),
        "evaluation_labels_used_for_training_or_weighting": False,
        "ensemble_weights": "uniform and fixed before evaluation",
        "target_metadata": target_metadata,
    }
    return per_seed, per_target, summary, paired, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument(
        "--calibration-seeds",
        default=",".join(str(seed) for seed in DEFAULT_CALIBRATION_SEEDS),
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    targets = _parse_csv(args.targets)
    seeds = _parse_int_csv(args.calibration_seeds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = load_katja_feature_cache(args.feature_cache)
    started = time.time()
    per_seed, per_target, summary, paired, metadata = run_targets(
        cache,
        targets=targets,
        calibration_seeds=seeds,
    )
    metadata["elapsed_seconds"] = time.time() - started
    per_seed.to_csv(output / "katja_physical_per_seed.csv", index=False)
    per_target.to_csv(output / "katja_physical_per_target.csv", index=False)
    summary.to_csv(output / "katja_physical_summary.csv", index=False)
    paired.to_csv(output / "katja_physical_paired.csv", index=False)
    (output / "katja_physical_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Paired comparisons ===")
    print(paired.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
