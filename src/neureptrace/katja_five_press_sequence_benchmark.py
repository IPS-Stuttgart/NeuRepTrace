"""Evaluate complete five-press global-finger sequence decoding on Katja MEG.

The model consumes all five presses and is trained as a five-class permutation,
but the reported primary endpoint scores only presses 2--5. The fixed first press
is therefore auxiliary sequence context, not an added evaluation target.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from neureptrace._katja_finger_sequence_support import (
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_PARTICIPANTS,
    DEFAULT_SOURCE_SELECTION_SEED,
    JULIA_FULL_FINETUNE_ACCURACY,
    _composite_trial_ids,
    _constant_trial_values,
    _fit_source_preprocessor,
    _load_source_map,
    _mean_sem,
    _parse_csv_values,
    _stable_source_selection,
    _transform_features,
    katja_nested_trial_calibration_indices,
    load_katja_feature_cache,
)
from neureptrace.decoding.progressive_sequence_finetune import (
    TorchProgressiveSequenceClassifier,
    pack_complete_trial_events,
    permutation_constrained_decode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def scored_variable_press_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    press_positions: np.ndarray,
) -> float:
    """Score presses 2--5 while excluding the auxiliary first press."""

    predicted = np.asarray(predictions)
    true = np.asarray(labels)
    positions = np.asarray(press_positions)
    if predicted.shape != true.shape or positions.shape != true.shape:
        raise ValueError("predictions, labels, and press_positions must have matching shapes.")
    mask = positions >= 2
    if not np.any(mask):
        raise ValueError("No scored variable-press positions are present.")
    return float(np.mean(predicted[mask] == true[mask]))


def fixed_first_press_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    press_positions: np.ndarray,
) -> float:
    """Return the auxiliary first-press diagnostic accuracy."""

    predicted = np.asarray(predictions)
    true = np.asarray(labels)
    positions = np.asarray(press_positions)
    if predicted.shape != true.shape or positions.shape != true.shape:
        raise ValueError("predictions, labels, and press_positions must have matching shapes.")
    mask = positions == 1
    if not np.any(mask):
        raise ValueError("No first-press positions are present.")
    return float(np.mean(predicted[mask] == true[mask]))


def run_katja_five_press_sequence_benchmark(
    cache: dict[str, np.ndarray],
    *,
    participants: tuple[str, ...] = DEFAULT_PARTICIPANTS,
    target_participants: tuple[str, ...] | None = None,
    calibration_counts: tuple[int, ...] = (20,),
    calibration_seeds: tuple[int, ...] = DEFAULT_CALIBRATION_SEEDS,
    n_source_participants: int = 9,
    source_selection_seed: int = DEFAULT_SOURCE_SELECTION_SEED,
    source_map: dict[str, tuple[str, ...]] | None = None,
    pca_components: int | None = 256,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run exact nested calibration with a complete five-press sequence model."""

    subjects = cache["subjects"].astype(str)
    trial_ids = cache["trial_ids"]
    positions = cache["press_positions"].astype(int)
    sequence_ids = cache["sequence_ids"]
    physical_labels = np.asarray(cache["finger_codes"])
    correct_order = cache["correct_order"].astype(bool)
    included = (
        correct_order
        & np.isin(positions, np.asarray([1, 2, 3, 4, 5]))
        & np.isin(subjects, np.asarray(participants))
    )
    if not np.any(included):
        raise ValueError("No complete five-press rows remain after filtering.")

    features = cache["features"][included]
    subjects = subjects[included]
    trial_ids = trial_ids[included]
    positions = positions[included]
    sequence_ids = sequence_ids[included]
    physical_labels = physical_labels[included]
    targets = participants if target_participants is None else target_participants
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_participants must contain unique identifiers.")
    if any(target not in participants for target in targets):
        raise ValueError("Every target participant must be in participants.")
    source_map = {} if source_map is None else source_map
    config = {} if model_kwargs is None else dict(model_kwargs)
    if config.get("enforce_permutation_labels") is False:
        raise ValueError("Five-press training requires enforce_permutation_labels=True.")
    config["enforce_permutation_labels"] = True

    rows: list[dict[str, Any]] = []
    source_registry: dict[str, tuple[str, ...]] = {}
    for target in targets:
        target_mask = subjects == target
        if not np.any(target_mask):
            raise ValueError(f"Target participant {target!r} has no rows.")
        selected_sources = source_map.get(target)
        if selected_sources is None:
            selected_sources = _stable_source_selection(
                participants,
                target=target,
                n_sources=n_source_participants,
                seed=source_selection_seed,
            )
        if target in selected_sources or len(set(selected_sources)) != len(selected_sources):
            raise ValueError(f"Invalid source set for target {target!r}: {selected_sources!r}.")
        source_registry[target] = selected_sources
        source_mask = np.isin(subjects, np.asarray(selected_sources))

        scaler, pca, source_transformed = _fit_source_preprocessor(
            features[source_mask],
            pca_components=pca_components,
        )
        target_transformed = _transform_features(
            features[target_mask],
            scaler=scaler,
            pca=pca,
        )
        source_packed = pack_complete_trial_events(
            source_transformed,
            _composite_trial_ids(subjects[source_mask], trial_ids[source_mask]),
            positions[source_mask],
            labels=physical_labels[source_mask],
            expected_events=5,
            require_permutation_labels=True,
        )
        target_packed = pack_complete_trial_events(
            target_transformed,
            trial_ids[target_mask],
            positions[target_mask],
            labels=physical_labels[target_mask],
            expected_events=5,
            require_permutation_labels=True,
        )
        source_trial_subjects = _constant_trial_values(
            subjects[source_mask],
            source_packed.row_indices,
            name="source subject",
        )
        target_trial_strata = _constant_trial_values(
            sequence_ids[target_mask],
            target_packed.row_indices,
            name="target sequence ID",
        )

        for calibration_seed in calibration_seeds:
            calibration, evaluation, pool = katja_nested_trial_calibration_indices(
                target_trial_strata,
                calibration_counts,
                seed=int(calibration_seed),
            )
            base_model = TorchProgressiveSequenceClassifier(
                **config,
                random_state=int(calibration_seed),
            )
            base_model.fit_source(
                source_packed.features,
                source_packed.labels,
                source_subjects=source_trial_subjects,
            )
            for count in calibration_counts:
                calibration_indices = calibration[int(count)]
                model = copy.deepcopy(base_model)
                model.adapt_target(
                    target_packed.features[calibration_indices],
                    target_packed.labels[calibration_indices],
                    target_strata=target_trial_strata[calibration_indices],
                )
                evaluation_features = target_packed.features[evaluation]
                evaluation_labels = target_packed.labels[evaluation]
                evaluation_positions = target_packed.press_positions[evaluation]
                probabilities = model.predict_proba(
                    evaluation_features,
                    constrained=False,
                )
                independent_predictions = model.classes_[
                    np.argmax(probabilities, axis=2)
                ]
                constrained = permutation_constrained_decode(
                    probabilities,
                    temperature=model.sinkhorn_temperature,
                    sinkhorn_iterations=model.sinkhorn_iterations,
                )
                constrained_predictions = model.classes_[constrained.assignments]
                rows.append(
                    {
                        "target": target,
                        "seed": int(calibration_seed),
                        "k": int(count),
                        "n_source_participants": len(selected_sources),
                        "source_participants": ",".join(selected_sources),
                        "n_source_trials": int(source_packed.features.shape[0]),
                        "n_calibration_trials": int(calibration_indices.size),
                        "n_calibration_events": int(calibration_indices.size * 5),
                        "n_evaluation_trials": int(evaluation.size),
                        "n_scored_evaluation_events": int(evaluation.size * 4),
                        "independent_accuracy": scored_variable_press_accuracy(
                            independent_predictions,
                            evaluation_labels,
                            evaluation_positions,
                        ),
                        "permutation_accuracy": scored_variable_press_accuracy(
                            constrained_predictions,
                            evaluation_labels,
                            evaluation_positions,
                        ),
                        "fixed_first_independent_accuracy": fixed_first_press_accuracy(
                            independent_predictions,
                            evaluation_labels,
                            evaluation_positions,
                        ),
                        "fixed_first_permutation_accuracy": fixed_first_press_accuracy(
                            constrained_predictions,
                            evaluation_labels,
                            evaluation_positions,
                        ),
                        "source_validation_mode": model.source_validation_mode_,
                        "target_validation_mode": model.target_validation_mode_,
                        "adaptation_stages": ",".join(
                            item["stage"] for item in model.adaptation_stage_history_
                        ),
                        "pca_components_requested": (
                            None if pca_components is None else int(pca_components)
                        ),
                        "pca_components_effective": int(
                            source_packed.features.shape[2]
                        ),
                        "calibration_pool_trials": int(pool.size),
                    }
                )

    per_seed = pd.DataFrame(rows).sort_values(
        ["target", "k", "seed"]
    ).reset_index(drop=True)
    per_target = (
        per_seed.groupby(["target", "k"], as_index=False)
        .agg(
            independent_accuracy=("independent_accuracy", "mean"),
            permutation_accuracy=("permutation_accuracy", "mean"),
            fixed_first_independent_accuracy=(
                "fixed_first_independent_accuracy",
                "mean",
            ),
            fixed_first_permutation_accuracy=(
                "fixed_first_permutation_accuracy",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
            n_evaluation_trials=("n_evaluation_trials", "first"),
            n_scored_evaluation_events=(
                "n_scored_evaluation_events",
                "first",
            ),
        )
        .sort_values(["target", "k"])
        .reset_index(drop=True)
    )
    summary_rows: list[dict[str, Any]] = []
    for count, frame in per_target.groupby("k", sort=True):
        independent_mean, independent_sem = _mean_sem(
            frame["independent_accuracy"].to_numpy()
        )
        permutation_mean, permutation_sem = _mean_sem(
            frame["permutation_accuracy"].to_numpy()
        )
        fixed_mean, fixed_sem = _mean_sem(
            frame["fixed_first_independent_accuracy"].to_numpy()
        )
        julia_reference = JULIA_FULL_FINETUNE_ACCURACY.get(
            int(count),
            float("nan"),
        )
        summary_rows.append(
            {
                "k": int(count),
                "n_targets": int(frame.shape[0]),
                "independent_accuracy_mean": independent_mean,
                "independent_accuracy_sem": independent_sem,
                "permutation_accuracy_mean": permutation_mean,
                "permutation_accuracy_sem": permutation_sem,
                "fixed_first_independent_accuracy_mean": fixed_mean,
                "fixed_first_independent_accuracy_sem": fixed_sem,
                "julia_full_finetune_accuracy": julia_reference,
                "independent_delta_vs_julia": independent_mean - julia_reference,
                "permutation_delta_vs_julia": permutation_mean - julia_reference,
                "independent_outperforms_julia": bool(
                    independent_mean > julia_reference
                ),
                "permutation_outperforms_julia": bool(
                    permutation_mean > julia_reference
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("k").reset_index(drop=True)
    metadata = {
        "participants": list(participants),
        "target_participants": list(targets),
        "calibration_counts": [int(value) for value in calibration_counts],
        "calibration_seeds": [int(value) for value in calibration_seeds],
        "source_selection_seed": int(source_selection_seed),
        "source_selection": {
            target: list(sources) for target, sources in source_registry.items()
        },
        "pca_components": (
            None if pca_components is None else int(pca_components)
        ),
        "model_kwargs": config,
        "training_event_positions": [1, 2, 3, 4, 5],
        "scored_event_positions": [2, 3, 4, 5],
        "training_label_space": "global_physical_finger",
        "evaluation_label_space": "global_physical_finger",
        "fixed_first_press_role": "auxiliary_sequence_context_only",
        "evaluation_labels_used_for_fitting": False,
        "independent_endpoint": "eventwise_argmax_on_presses_2_to_5",
        "permutation_endpoint": "five_press_hungarian_scored_on_presses_2_to_5",
        "sequence_id_used_as_feature": False,
        "sequence_id_used_for_calibration_stratification": True,
        "comparison_split_implementation": "recovered_sequential_rng",
    }
    return per_seed, per_target, summary, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--targets")
    parser.add_argument("--calibration-counts", default="20")
    parser.add_argument(
        "--calibration-seeds",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_SEEDS),
    )
    parser.add_argument("--n-source-participants", type=int, default=9)
    parser.add_argument(
        "--source-selection-seed",
        type=int,
        default=DEFAULT_SOURCE_SELECTION_SEED,
    )
    parser.add_argument("--source-map-json")
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--hidden-units", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--source-max-epochs", type=int, default=120)
    parser.add_argument("--meta-epochs", type=int, default=2)
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--last-block-steps", type=int, default=60)
    parser.add_argument("--full-finetune-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cache = load_katja_feature_cache(args.feature_cache)
    per_seed, per_target, summary, metadata = (
        run_katja_five_press_sequence_benchmark(
            cache,
            participants=_parse_csv_values(args.participants),
            target_participants=(
                None if args.targets is None else _parse_csv_values(args.targets)
            ),
            calibration_counts=_parse_csv_values(
                args.calibration_counts,
                cast=int,
            ),
            calibration_seeds=_parse_csv_values(
                args.calibration_seeds,
                cast=int,
            ),
            n_source_participants=args.n_source_participants,
            source_selection_seed=args.source_selection_seed,
            source_map=_load_source_map(args.source_map_json),
            pca_components=(None if args.no_pca else args.pca_components),
            model_kwargs={
                "hidden_units": args.hidden_units,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "adapter_rank": args.adapter_rank,
                "source_max_epochs": args.source_max_epochs,
                "meta_epochs": args.meta_epochs,
                "adapter_steps": args.adapter_steps,
                "last_block_steps": args.last_block_steps,
                "full_finetune_steps": args.full_finetune_steps,
                "batch_size": args.batch_size,
                "device": args.device,
            },
        )
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output / "katja_five_press_per_seed.csv", index=False)
    per_target.to_csv(output / "katja_five_press_per_target.csv", index=False)
    summary.to_csv(output / "katja_five_press_summary.csv", index=False)
    (output / "katja_five_press_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "fixed_first_press_accuracy",
    "run_katja_five_press_sequence_benchmark",
    "scored_variable_press_accuracy",
)
