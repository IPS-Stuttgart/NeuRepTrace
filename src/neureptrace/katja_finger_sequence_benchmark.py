"""Run the Katja four-variable-finger calibrated sequence benchmark.

The runner consumes a feature-level cache and performs fold-local preprocessing,
nested complete-trial calibration, progressive neural adaptation, and reporting.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from neureptrace._katja_finger_sequence_support import (
    DEFAULT_CALIBRATION_COUNTS,
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_PARTICIPANTS,
    JULIA_FULL_FINETUNE_ACCURACY,
    _composite_trial_ids,
    _constant_trial_values,
    _fit_source_preprocessor,
    _load_source_map,
    _mean_sem,
    _parse_csv_values,
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

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

__all__ = (
    "DEFAULT_CALIBRATION_COUNTS",
    "DEFAULT_CALIBRATION_SEEDS",
    "DEFAULT_PARTICIPANTS",
    "JULIA_FULL_FINETUNE_ACCURACY",
    "build_arg_parser",
    "derive_participant_local_finger_labels",
    "load_katja_feature_cache",
    "main",
    "run_katja_finger_sequence_benchmark",
)


def _integer_control(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Normalize one integer experiment control without lossy coercion."""

    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, (int, np.integer)):
        integer = int(value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if not np.isfinite(number) or number % 1.0 != 0.0:
            raise ValueError(f"{name} must be an integer.")
        integer = int(number)
    if minimum is not None and integer < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        if minimum == 1:
            raise ValueError(f"{name} must be a positive integer.")
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return integer


def _integer_registry(
    values: Iterable[Any],
    *,
    name: str,
    minimum: int | None = None,
) -> tuple[int, ...]:
    """Normalize a non-empty registry of unique integer controls."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of integers.")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence of integers.") from exc
    if not items:
        raise ValueError(f"{name} must not be empty.")
    normalized = tuple(
        _integer_control(value, name=f"{name} value", minimum=minimum)
        for value in items
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique values.")
    return normalized


def run_katja_finger_sequence_benchmark(
    cache: dict[str, np.ndarray],
    *,
    participants: tuple[str, ...] = DEFAULT_PARTICIPANTS,
    target_participants: tuple[str, ...] | None = None,
    calibration_counts: tuple[int, ...] = DEFAULT_CALIBRATION_COUNTS,
    calibration_seeds: tuple[int, ...] = DEFAULT_CALIBRATION_SEEDS,
    event_positions: tuple[int, ...] = (2, 3, 4, 5),
    n_source_participants: int = 9,
    source_selection_seed: int = 13,
    source_map: dict[str, tuple[str, ...]] | None = None,
    pca_components: int | None = 64,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the complete target-by-seed-by-k benchmark from event-row features."""

    calibration_counts = _integer_registry(
        calibration_counts,
        name="calibration_counts",
        minimum=1,
    )
    calibration_seeds = _integer_registry(
        calibration_seeds,
        name="calibration_seeds",
        minimum=0,
    )
    event_positions = _integer_registry(event_positions, name="event_positions")
    n_source_participants = _integer_control(
        n_source_participants,
        name="n_source_participants",
        minimum=1,
    )
    source_selection_seed = _integer_control(
        source_selection_seed,
        name="source_selection_seed",
        minimum=0,
    )
    pca_components = (
        None
        if pca_components is None
        else _integer_control(pca_components, name="pca_components", minimum=1)
    )

    subjects = cache["subjects"].astype(str)
    trial_ids = cache["trial_ids"]
    press_positions = cache["press_positions"]
    sequence_ids = cache["sequence_ids"]
    correct_order = cache["correct_order"].astype(bool)
    variable_mask = np.isin(press_positions.astype(int), np.asarray(event_positions, dtype=int))
    included = correct_order & variable_mask & np.isin(subjects, np.asarray(participants))
    if not np.any(included):
        raise ValueError("No cache rows remain after participant, correct-order, and event-position filters.")
    if "labels" in cache:
        labels = np.asarray(cache["labels"])
    else:
        labels = derive_participant_local_finger_labels(
            subjects,
            cache["finger_codes"],
            included_mask=included,
            expected_classes=len(event_positions),
        )

    features = cache["features"][included]
    subjects = subjects[included]
    trial_ids = trial_ids[included]
    press_positions = press_positions[included]
    sequence_ids = sequence_ids[included]
    labels = labels[included]
    available_participants = set(subjects.tolist())
    source_map = {} if source_map is None else source_map
    model_config = {} if model_kwargs is None else dict(model_kwargs)
    rows: list[dict[str, Any]] = []
    source_selection_registry: dict[str, tuple[str, ...]] = {}
    targets = participants if target_participants is None else target_participants
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_participants must contain unique identifiers.")
    unknown_targets = [target for target in targets if target not in participants]
    if unknown_targets:
        raise ValueError(f"Target participants are absent from the participant pool: {unknown_targets!r}.")

    for target in targets:
        target_mask = subjects == target
        if not np.any(target_mask):
            raise ValueError(f"Target participant {target!r} has no included rows.")
        selected_sources = source_map.get(target)
        if selected_sources is None:
            selected_sources = _stable_source_selection(
                participants,
                target=target,
                n_sources=n_source_participants,
                seed=source_selection_seed,
            )
        selected_sources = tuple(str(source) for source in selected_sources)
        if target in selected_sources or len(set(selected_sources)) != len(selected_sources):
            raise ValueError(f"Invalid source participant set for target {target!r}: {selected_sources!r}.")
        missing_sources = tuple(source for source in selected_sources if source not in available_participants)
        if missing_sources:
            raise ValueError(
                f"Selected source participants for target {target!r} lack included rows: {missing_sources!r}."
            )
        source_selection_registry[target] = selected_sources
        source_mask = np.isin(subjects, np.asarray(selected_sources))
        if not np.any(source_mask):
            raise ValueError(f"Target {target!r} has no rows from its selected source participants.")

        scaler, pca, source_transformed = _fit_source_preprocessor(features[source_mask], pca_components=pca_components)
        target_transformed = _transform_features(features[target_mask], scaler=scaler, pca=pca)
        source_packed = pack_complete_trial_events(
            source_transformed,
            _composite_trial_ids(subjects[source_mask], trial_ids[source_mask]),
            press_positions[source_mask],
            labels=labels[source_mask],
            expected_events=len(event_positions),
            require_permutation_labels=True,
        )
        target_packed = pack_complete_trial_events(
            target_transformed,
            trial_ids[target_mask],
            press_positions[target_mask],
            labels=labels[target_mask],
            expected_events=len(event_positions),
            require_permutation_labels=True,
        )
        source_trial_subjects = _constant_trial_values(subjects[source_mask], source_packed.row_indices, name="source subject")
        target_trial_strata = _constant_trial_values(sequence_ids[target_mask], target_packed.row_indices, name="target sequence ID")

        for calibration_seed in calibration_seeds:
            splits = select_nested_trial_calibration_splits(
                target_trial_strata,
                calibration_counts=calibration_counts,
                max_per_stratum=max(calibration_counts),
                min_evaluation_per_stratum=1,
                seed=calibration_seed,
                context=("katja_finger", target),
            )
            seed_model_config = {**model_config, "random_state": calibration_seed}
            base_model = TorchProgressiveSequenceClassifier(**seed_model_config)
            base_model.fit_source(source_packed.features, source_packed.labels, source_subjects=source_trial_subjects)
            for calibration_count in calibration_counts:
                split = splits[calibration_count]
                model = copy.deepcopy(base_model)
                model.adapt_target(
                    target_packed.features[split.calibration_indices],
                    target_packed.labels[split.calibration_indices],
                    target_strata=target_trial_strata[split.calibration_indices],
                )
                evaluation_features = target_packed.features[split.evaluation_indices]
                evaluation_labels = target_packed.labels[split.evaluation_indices]
                probabilities = model.predict_proba(evaluation_features, constrained=False)
                independent_predictions = model.classes_[np.argmax(probabilities, axis=2)]
                constrained = permutation_constrained_decode(
                    probabilities,
                    temperature=model.sinkhorn_temperature,
                    sinkhorn_iterations=model.sinkhorn_iterations,
                )
                constrained_predictions = model.classes_[constrained.assignments]
                rows.append(
                    {
                        "target": target,
                        "seed": calibration_seed,
                        "k": calibration_count,
                        "n_source_participants": len(selected_sources),
                        "source_participants": ",".join(selected_sources),
                        "n_source_trials": int(source_packed.features.shape[0]),
                        "n_calibration_trials": int(split.calibration_indices.size),
                        "n_evaluation_trials": int(split.evaluation_indices.size),
                        "n_evaluation_events": int(evaluation_labels.size),
                        "independent_accuracy": float(np.mean(independent_predictions == evaluation_labels)),
                        "permutation_accuracy": float(np.mean(constrained_predictions == evaluation_labels)),
                        "source_validation_mode": model.source_validation_mode_,
                        "target_validation_mode": model.target_validation_mode_,
                        "adaptation_stages": ",".join(item["stage"] for item in model.adaptation_stage_history_),
                        "pca_components_requested": pca_components,
                        "pca_components_effective": int(source_packed.features.shape[2]),
                    }
                )

    per_seed = pd.DataFrame(rows).sort_values(["target", "k", "seed"]).reset_index(drop=True)
    per_target = (
        per_seed.groupby(["target", "k"], as_index=False)
        .agg(
            independent_accuracy=("independent_accuracy", "mean"),
            permutation_accuracy=("permutation_accuracy", "mean"),
            n_seeds=("seed", "nunique"),
            n_evaluation_trials=("n_evaluation_trials", "first"),
            n_evaluation_events=("n_evaluation_events", "first"),
        )
        .sort_values(["target", "k"])
        .reset_index(drop=True)
    )
    summary_rows = []
    for calibration_count, frame in per_target.groupby("k", sort=True):
        independent_mean, independent_sem = _mean_sem(frame["independent_accuracy"].to_numpy())
        permutation_mean, permutation_sem = _mean_sem(frame["permutation_accuracy"].to_numpy())
        julia_reference = JULIA_FULL_FINETUNE_ACCURACY.get(int(calibration_count), float("nan"))
        summary_rows.append(
            {
                "k": int(calibration_count),
                "n_targets": int(frame.shape[0]),
                "independent_accuracy_mean": independent_mean,
                "independent_accuracy_sem": independent_sem,
                "permutation_accuracy_mean": permutation_mean,
                "permutation_accuracy_sem": permutation_sem,
                "julia_full_finetune_accuracy": julia_reference,
                "independent_delta_vs_julia": independent_mean - julia_reference,
                "permutation_delta_vs_julia": permutation_mean - julia_reference,
                "independent_outperforms_julia": bool(independent_mean > julia_reference),
                "permutation_outperforms_julia": bool(permutation_mean > julia_reference),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("k").reset_index(drop=True)
    metadata = {
        "participants": list(participants),
        "target_participants": list(targets),
        "calibration_counts": list(calibration_counts),
        "calibration_seeds": list(calibration_seeds),
        "event_positions": list(event_positions),
        "source_selection_seed": source_selection_seed,
        "source_selection": {target: list(sources) for target, sources in source_selection_registry.items()},
        "pca_components": pca_components,
        "model_kwargs": model_config,
        "evaluation_unit": "finger_event",
        "seed_aggregation": "mean_within_target_then_population_mean_and_sem",
        "evaluation_pool": "complement_of_maximum_calibration_pool_for_all_k",
        "sequence_id_used_as_feature": False,
        "sequence_id_used_for_calibration_stratification": True,
    }
    return per_seed, per_target, summary, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True, help="NPZ event-row feature cache.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV and JSON outputs.")
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--targets", help="Optional comma-separated target subset; sources still use --participants.")
    parser.add_argument("--calibration-counts", default=",".join(str(value) for value in DEFAULT_CALIBRATION_COUNTS))
    parser.add_argument("--calibration-seeds", default=",".join(str(value) for value in DEFAULT_CALIBRATION_SEEDS))
    parser.add_argument("--event-positions", default="2,3,4,5")
    parser.add_argument("--n-source-participants", type=int, default=9)
    parser.add_argument("--source-selection-seed", type=int, default=13)
    parser.add_argument("--source-map-json")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--hidden-units", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--source-max-epochs", type=int, default=120)
    parser.add_argument("--meta-epochs", type=int, default=2)
    parser.add_argument("--meta-support-trials", type=int, default=4)
    parser.add_argument("--meta-query-trials", type=int, default=4)
    parser.add_argument("--meta-inner-steps", type=int, default=5)
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--last-block-steps", type=int, default=60)
    parser.add_argument("--full-finetune-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cache = load_katja_feature_cache(args.feature_cache)
    per_seed, per_target, summary, metadata = run_katja_finger_sequence_benchmark(
        cache,
        participants=_parse_csv_values(args.participants),
        target_participants=None if args.targets is None else _parse_csv_values(args.targets),
        calibration_counts=_parse_csv_values(args.calibration_counts, cast=int),
        calibration_seeds=_parse_csv_values(args.calibration_seeds, cast=int),
        event_positions=_parse_csv_values(args.event_positions, cast=int),
        n_source_participants=args.n_source_participants,
        source_selection_seed=args.source_selection_seed,
        source_map=_load_source_map(args.source_map_json),
        pca_components=None if args.no_pca else args.pca_components,
        model_kwargs={
            "hidden_units": args.hidden_units,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "adapter_rank": args.adapter_rank,
            "source_max_epochs": args.source_max_epochs,
            "meta_epochs": args.meta_epochs,
            "meta_support_trials": args.meta_support_trials,
            "meta_query_trials": args.meta_query_trials,
            "meta_inner_steps": args.meta_inner_steps,
            "adapter_steps": args.adapter_steps,
            "last_block_steps": args.last_block_steps,
            "full_finetune_steps": args.full_finetune_steps,
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "katja_finger_sequence_per_seed.csv", index=False)
    per_target.to_csv(output_dir / "katja_finger_sequence_per_target.csv", index=False)
    summary.to_csv(output_dir / "katja_finger_sequence_summary.csv", index=False)
    (output_dir / "katja_finger_sequence_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
