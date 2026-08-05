"""Run a follow-up Katja population evaluation for development-selected ensembles.

The development participant ``s05`` is excluded. This is explicitly a follow-up
validation after the single-model 16-target result from PR #2109, not a new
untouched confirmatory cohort.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from katja_ensemble_source_sweep import (
    _configurations,
    _member_specs,
    _model_kwargs,
    _model_seed,
    _prepare_rows,
    _source_space_specs,
)
from neureptrace._katja_finger_sequence_support import (
    DEFAULT_CALIBRATION_COUNTS,
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_PARTICIPANTS,
    JULIA_FULL_FINETUNE_ACCURACY,
    _composite_trial_ids,
    _constant_trial_values,
    _fit_source_preprocessor,
    _mean_sem,
    _stable_source_selection,
    _transform_features,
    load_katja_feature_cache,
)
from neureptrace.decoding.progressive_sequence_finetune import (
    TorchProgressiveSequenceClassifier,
    pack_complete_trial_events,
    permutation_constrained_decode,
    select_nested_trial_calibration_splits,
)

DEVELOPMENT_TARGET = "s05"
PRIMARY_CONFIGURATION = "hybrid_source_model_ensemble6"
CONFIGURATION_CATEGORIES = {
    "nine_single_protocol": "strict_nine_source_single_model",
    "nine_model_ensemble3": "strict_nine_source_model_ensemble",
    "nine_source_ensemble3": "expanded_source_subset_ensemble",
    "all16_single_protocol": "all_available_sources_single_model",
    "all16_model_ensemble3": "all_available_sources_model_ensemble",
    "hybrid_source_model_ensemble6": "expanded_source_and_model_ensemble",
}


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("Comma-separated participant list must not be empty.")
    return result


def _parse_int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("Comma-separated integer list must contain positive values.")
    return result


def _prepare_target_source_space(
    rows: dict[str, np.ndarray],
    *,
    target: str,
    n_source_participants: int,
    source_selection_seed: int,
) -> dict[str, Any]:
    subjects = rows["subjects"].astype(str)
    target_mask = subjects == target
    if not np.any(target_mask):
        raise ValueError(f"Target participant {target!r} has no included rows.")
    selected_sources = _stable_source_selection(
        DEFAULT_PARTICIPANTS,
        target=target,
        n_sources=n_source_participants,
        seed=source_selection_seed,
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
    source_packed = pack_complete_trial_events(
        source_transformed,
        _composite_trial_ids(
            subjects[source_mask],
            rows["trial_ids"][source_mask],
        ),
        rows["press_positions"][source_mask],
        labels=rows["labels"][source_mask],
        expected_events=4,
        require_permutation_labels=True,
    )
    target_packed = pack_complete_trial_events(
        target_transformed,
        rows["trial_ids"][target_mask],
        rows["press_positions"][target_mask],
        labels=rows["labels"][target_mask],
        expected_events=4,
        require_permutation_labels=True,
    )
    if source_packed.labels is None or target_packed.labels is None:
        raise RuntimeError("Source and target labels are required for evaluation.")
    source_subjects = _constant_trial_values(
        subjects[source_mask],
        source_packed.row_indices,
        name="source subject",
    )
    target_strata = _constant_trial_values(
        rows["sequence_ids"][target_mask],
        target_packed.row_indices,
        name="target sequence ID",
    )
    return {
        "selected_sources": selected_sources,
        "source_features": source_packed.features,
        "source_labels": source_packed.labels,
        "source_subjects": source_subjects,
        "target_features": target_packed.features,
        "target_labels": target_packed.labels,
        "target_trial_ids": target_packed.trial_ids,
        "target_strata": target_strata,
    }


def _validate_target_spaces(spaces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference = next(iter(spaces.values()))
    for space in spaces.values():
        if not np.array_equal(
            space["target_trial_ids"], reference["target_trial_ids"]
        ):
            raise RuntimeError("Source spaces have different target trial ordering.")
        if not np.array_equal(
            space["target_labels"], reference["target_labels"]
        ):
            raise RuntimeError("Source spaces have different target labels.")
        if not np.array_equal(space["target_strata"], reference["target_strata"]):
            raise RuntimeError("Source spaces have different target strata.")
    return reference


def _fit_source_model(
    space: dict[str, Any],
    *,
    model_seed: int,
) -> TorchProgressiveSequenceClassifier:
    model = TorchProgressiveSequenceClassifier(**_model_kwargs(model_seed))
    model.fit_source(
        space["source_features"],
        space["source_labels"],
        source_subjects=space["source_subjects"],
    )
    return model


def _summarize_population(
    per_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_target = (
        per_seed.groupby(["configuration", "target", "k"], as_index=False)
        .agg(
            independent_accuracy=("independent_accuracy", "mean"),
            permutation_accuracy=("permutation_accuracy", "mean"),
            n_seeds=("seed", "nunique"),
            n_evaluation_trials=("n_evaluation_trials", "first"),
            n_evaluation_events=("n_evaluation_events", "first"),
        )
        .sort_values(["configuration", "target", "k"])
        .reset_index(drop=True)
    )
    summary_rows: list[dict[str, Any]] = []
    for (configuration, calibration_count), frame in per_target.groupby(
        ["configuration", "k"], sort=True
    ):
        independent_mean, independent_sem = _mean_sem(
            frame["independent_accuracy"].to_numpy()
        )
        permutation_mean, permutation_sem = _mean_sem(
            frame["permutation_accuracy"].to_numpy()
        )
        julia = JULIA_FULL_FINETUNE_ACCURACY.get(
            int(calibration_count), float("nan")
        )
        summary_rows.append(
            {
                "configuration": str(configuration),
                "category": CONFIGURATION_CATEGORIES[str(configuration)],
                "primary_configuration": str(configuration)
                == PRIMARY_CONFIGURATION,
                "k": int(calibration_count),
                "n_targets": int(frame.shape[0]),
                "independent_accuracy_mean": independent_mean,
                "independent_accuracy_sem": independent_sem,
                "permutation_accuracy_mean": permutation_mean,
                "permutation_accuracy_sem": permutation_sem,
                "julia_full_finetune_accuracy": julia,
                "independent_delta_vs_julia": independent_mean - julia,
                "permutation_delta_vs_julia": permutation_mean - julia,
                "independent_outperforms_julia": bool(independent_mean > julia),
                "permutation_outperforms_julia": bool(permutation_mean > julia),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["configuration", "k"]
    )
    return per_target, summary.reset_index(drop=True)


def run_targets(
    cache: dict[str, np.ndarray],
    *,
    targets: tuple[str, ...],
    calibration_counts: tuple[int, ...] = DEFAULT_CALIBRATION_COUNTS,
    calibration_seeds: tuple[int, ...] = DEFAULT_CALIBRATION_SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if DEVELOPMENT_TARGET in targets:
        raise ValueError("The development target s05 must be excluded.")
    unknown = [target for target in targets if target not in DEFAULT_PARTICIPANTS]
    if unknown:
        raise ValueError(f"Unknown target participants: {unknown!r}.")
    rows = _prepare_rows(cache)
    source_specs = _source_space_specs()
    member_specs = _member_specs()
    configurations = _configurations()
    needed_members = sorted(
        {member for members in configurations.values() for member in members}
    )
    result_rows: list[dict[str, Any]] = []
    target_metadata: dict[str, Any] = {}

    for target in targets:
        target_started = time.time()
        spaces = {
            name: _prepare_target_source_space(
                rows,
                target=target,
                n_source_participants=spec.n_source_participants,
                source_selection_seed=spec.source_selection_seed,
            )
            for name, spec in source_specs.items()
        }
        reference = _validate_target_spaces(spaces)
        target_metadata[target] = {
            "source_spaces": {
                name: {
                    "selected_sources": list(space["selected_sources"]),
                    "n_source_trials": int(space["source_features"].shape[0]),
                }
                for name, space in spaces.items()
            },
            "n_target_trials": int(reference["target_features"].shape[0]),
        }

        for calibration_seed in calibration_seeds:
            splits = select_nested_trial_calibration_splits(
                reference["target_strata"],
                calibration_counts=calibration_counts,
                max_per_stratum=max(calibration_counts),
                min_evaluation_per_stratum=1,
                seed=int(calibration_seed),
                context=("katja_finger", target),
            )
            fitted_members: dict[str, TorchProgressiveSequenceClassifier] = {}
            for member_name in needed_members:
                member_spec = member_specs[member_name]
                model_seed = _model_seed(
                    calibration_seed, member_spec.seed_variant
                )
                fitted_members[member_name] = _fit_source_model(
                    spaces[member_spec.source_space],
                    model_seed=model_seed,
                )

            for calibration_count in calibration_counts:
                split = splits[int(calibration_count)]
                probabilities_by_member: dict[str, np.ndarray] = {}
                for member_name in needed_members:
                    member_spec = member_specs[member_name]
                    space = spaces[member_spec.source_space]
                    model = copy.deepcopy(fitted_members[member_name])
                    model.adapt_target(
                        space["target_features"][split.calibration_indices],
                        space["target_labels"][split.calibration_indices],
                        target_strata=space["target_strata"][
                            split.calibration_indices
                        ],
                    )
                    probabilities_by_member[member_name] = model.predict_proba(
                        space["target_features"][split.evaluation_indices],
                        constrained=False,
                    )
                    del model
                    torch.cuda.empty_cache()

                evaluation_labels = reference["target_labels"][
                    split.evaluation_indices
                ]
                for configuration, members in configurations.items():
                    probabilities = np.mean(
                        np.stack(
                            [probabilities_by_member[name] for name in members],
                            axis=0,
                        ),
                        axis=0,
                    )
                    independent = np.argmax(probabilities, axis=2)
                    constrained = permutation_constrained_decode(
                        probabilities,
                        temperature=0.5,
                        sinkhorn_iterations=20,
                    ).assignments
                    result_rows.append(
                        {
                            "target": target,
                            "configuration": configuration,
                            "category": CONFIGURATION_CATEGORIES[configuration],
                            "seed": int(calibration_seed),
                            "k": int(calibration_count),
                            "ensemble_size": len(members),
                            "members": ",".join(members),
                            "n_evaluation_trials": int(
                                split.evaluation_indices.size
                            ),
                            "n_evaluation_events": int(evaluation_labels.size),
                            "independent_accuracy": float(
                                np.mean(independent == evaluation_labels)
                            ),
                            "permutation_accuracy": float(
                                np.mean(constrained == evaluation_labels)
                            ),
                        }
                    )

            del fitted_members
            torch.cuda.empty_cache()

        target_metadata[target]["elapsed_seconds"] = time.time() - target_started
        print(
            f"Completed {target} in "
            f"{target_metadata[target]['elapsed_seconds']:.1f} s"
        )

    per_seed = pd.DataFrame(result_rows).sort_values(
        ["configuration", "target", "k", "seed"]
    )
    per_target, summary = _summarize_population(per_seed)
    metadata = {
        "analysis_status": (
            "follow_up_validation_after_prior_16_target_single_model_result"
        ),
        "development_target_excluded": DEVELOPMENT_TARGET,
        "targets": list(targets),
        "calibration_counts": list(calibration_counts),
        "calibration_seeds": list(calibration_seeds),
        "primary_configuration": PRIMARY_CONFIGURATION,
        "configuration_categories": CONFIGURATION_CATEGORIES,
        "target_metadata": target_metadata,
        "evaluation_unit": "finger_event",
        "seed_aggregation": (
            "mean_within_target_then_population_mean_and_sem"
        ),
        "evaluation_pool": (
            "complement_of_maximum_calibration_pool_for_all_k"
        ),
        "sequence_id_used_as_feature": False,
        "sequence_id_used_for_calibration_stratification": True,
    }
    return per_seed, per_target, summary, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument(
        "--calibration-counts",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_COUNTS),
    )
    parser.add_argument(
        "--calibration-seeds",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_SEEDS),
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    targets = _parse_csv(args.targets)
    calibration_counts = _parse_int_csv(args.calibration_counts)
    calibration_seeds = _parse_int_csv(args.calibration_seeds)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = load_katja_feature_cache(args.feature_cache)
    started = time.time()
    per_seed, per_target, summary, metadata = run_targets(
        cache,
        targets=targets,
        calibration_counts=calibration_counts,
        calibration_seeds=calibration_seeds,
    )
    metadata["elapsed_seconds"] = time.time() - started
    per_seed.to_csv(output / "katja_ensemble_population_per_seed.csv", index=False)
    per_target.to_csv(
        output / "katja_ensemble_population_per_target.csv", index=False
    )
    summary.to_csv(output / "katja_ensemble_population_summary.csv", index=False)
    (output / "katja_ensemble_population_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
