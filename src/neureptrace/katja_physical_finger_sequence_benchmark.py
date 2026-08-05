"""Run Katja sequence decoding with global physical-finger source pretraining."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from neureptrace.decoding.physical_finger_sequence_finetune import TorchPhysicalFingerSequenceClassifier
from neureptrace.decoding.progressive_sequence_finetune import (
    pack_complete_trial_events,
    permutation_constrained_decode,
    select_nested_trial_calibration_splits,
)

if TYPE_CHECKING:
    from typing import Any

SOURCE_SELECTION_MODES = ("deterministic", "all", "same_variable_set")


def _repeat_seed(calibration_seed: int, target: str, repeat: int) -> int:
    payload = f"katja-physical|{calibration_seed}|{target}|{repeat}".encode()
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) % (2**32)


def _participant_variable_codes(subjects: np.ndarray, physical_codes: np.ndarray) -> dict[str, tuple[Any, ...]]:
    result: dict[str, tuple[Any, ...]] = {}
    for subject in dict.fromkeys(subjects.tolist()):
        values = np.unique(physical_codes[subjects == subject])
        try:
            values = np.sort(values)
        except TypeError:
            values = np.asarray(sorted(values.tolist(), key=str), dtype=object)
        result[str(subject)] = tuple(values.tolist())
    return result


def _select_sources(
    *,
    participants: tuple[str, ...],
    target: str,
    mode: str,
    n_sources: int,
    seed: int,
    source_map: dict[str, tuple[str, ...]],
    variable_codes_by_subject: dict[str, tuple[Any, ...]],
) -> tuple[str, ...]:
    explicit = source_map.get(target)
    if explicit is not None:
        selected = explicit
    elif mode == "deterministic":
        selected = _stable_source_selection(
            participants,
            target=target,
            n_sources=n_sources,
            seed=seed,
        )
    elif mode == "all":
        selected = tuple(participant for participant in participants if participant != target)
    elif mode == "same_variable_set":
        target_codes = variable_codes_by_subject[target]
        selected = tuple(
            participant
            for participant in participants
            if participant != target and variable_codes_by_subject[participant] == target_codes
        )
    else:
        raise ValueError(f"source_selection_mode must be one of {SOURCE_SELECTION_MODES}; got {mode!r}.")
    if not selected:
        raise ValueError(f"Source selection mode {mode!r} produced no source participants for {target!r}.")
    if target in selected or len(set(selected)) != len(selected):
        raise ValueError(f"Invalid source participant set for target {target!r}: {selected!r}.")
    unknown = [subject for subject in selected if subject not in participants]
    if unknown:
        raise ValueError(f"Source participants are absent from the participant pool: {unknown!r}.")
    return tuple(selected)


def run_katja_physical_finger_sequence_benchmark(
    cache: dict[str, np.ndarray],
    *,
    participants: tuple[str, ...] = DEFAULT_PARTICIPANTS,
    target_participants: tuple[str, ...] | None = None,
    calibration_counts: tuple[int, ...] = DEFAULT_CALIBRATION_COUNTS,
    calibration_seeds: tuple[int, ...] = DEFAULT_CALIBRATION_SEEDS,
    event_positions: tuple[int, ...] = (2, 3, 4, 5),
    source_selection_mode: str = "deterministic",
    n_source_participants: int = 9,
    source_selection_seed: int = 13,
    source_map: dict[str, tuple[str, ...]] | None = None,
    pca_components: int | None = 64,
    model_repeats: int = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate physical-head pretraining with nested target calibration."""

    if model_repeats < 1:
        raise ValueError("model_repeats must be positive.")
    subjects = cache["subjects"].astype(str)
    trial_ids = cache["trial_ids"]
    press_positions = cache["press_positions"]
    sequence_ids = cache["sequence_ids"]
    physical_codes = np.asarray(cache["finger_codes"])
    correct_order = cache["correct_order"].astype(bool)
    variable_mask = np.isin(press_positions.astype(int), np.asarray(event_positions, dtype=int))
    included = correct_order & variable_mask & np.isin(subjects, np.asarray(participants))
    if not np.any(included):
        raise ValueError("No cache rows remain after participant, correct-order, and event-position filters.")
    if "labels" in cache:
        local_labels = np.asarray(cache["labels"])
    else:
        local_labels = derive_participant_local_finger_labels(
            subjects,
            physical_codes,
            included_mask=included,
            expected_classes=len(event_positions),
        )

    features = cache["features"][included]
    subjects = subjects[included]
    trial_ids = trial_ids[included]
    press_positions = press_positions[included]
    sequence_ids = sequence_ids[included]
    physical_codes = physical_codes[included]
    local_labels = local_labels[included]
    variable_codes_by_subject = _participant_variable_codes(subjects, physical_codes)
    targets = participants if target_participants is None else target_participants
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_participants must contain unique identifiers.")
    unknown_targets = [target for target in targets if target not in participants]
    if unknown_targets:
        raise ValueError(f"Target participants are absent from the participant pool: {unknown_targets!r}.")

    source_map = {} if source_map is None else source_map
    model_config = {} if model_kwargs is None else dict(model_kwargs)
    rows: list[dict[str, Any]] = []
    source_selection_registry: dict[str, tuple[str, ...]] = {}

    for target in targets:
        target_mask = subjects == target
        if not np.any(target_mask):
            raise ValueError(f"Target participant {target!r} has no included rows.")
        selected_sources = _select_sources(
            participants=participants,
            target=target,
            mode=source_selection_mode,
            n_sources=n_source_participants,
            seed=source_selection_seed,
            source_map=source_map,
            variable_codes_by_subject=variable_codes_by_subject,
        )
        source_selection_registry[target] = selected_sources
        source_mask = np.isin(subjects, np.asarray(selected_sources))
        scaler, pca, source_transformed = _fit_source_preprocessor(
            features[source_mask],
            pca_components=pca_components,
        )
        target_transformed = _transform_features(features[target_mask], scaler=scaler, pca=pca)
        source_packed = pack_complete_trial_events(
            source_transformed,
            _composite_trial_ids(subjects[source_mask], trial_ids[source_mask]),
            press_positions[source_mask],
            labels=physical_codes[source_mask],
            expected_events=len(event_positions),
            require_permutation_labels=False,
        )
        target_packed = pack_complete_trial_events(
            target_transformed,
            trial_ids[target_mask],
            press_positions[target_mask],
            labels=local_labels[target_mask],
            expected_events=len(event_positions),
            require_permutation_labels=True,
        )
        target_physical_labels = physical_codes[target_mask][target_packed.row_indices]
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
            splits = select_nested_trial_calibration_splits(
                target_trial_strata,
                calibration_counts=calibration_counts,
                max_per_stratum=max(calibration_counts),
                min_evaluation_per_stratum=1,
                seed=int(calibration_seed),
                context=("katja_physical_finger", target),
            )
            base_models = []
            for repeat in range(model_repeats):
                repeat_config = {
                    **model_config,
                    "random_state": _repeat_seed(int(calibration_seed), target, repeat),
                }
                model = TorchPhysicalFingerSequenceClassifier(**repeat_config)
                model.fit_source(
                    source_packed.features,
                    source_packed.labels,
                    source_subjects=source_trial_subjects,
                )
                base_models.append(model)

            for calibration_count in calibration_counts:
                split = splits[int(calibration_count)]
                evaluation_features = target_packed.features[split.evaluation_indices]
                evaluation_labels = target_packed.labels[split.evaluation_indices]
                repeat_probabilities = []
                stage_histories = []
                for base_model in base_models:
                    model = copy.deepcopy(base_model)
                    model.adapt_target(
                        target_packed.features[split.calibration_indices],
                        target_packed.labels[split.calibration_indices],
                        target_calibration_physical_labels=target_physical_labels[split.calibration_indices],
                        target_strata=target_trial_strata[split.calibration_indices],
                    )
                    repeat_probabilities.append(model.predict_proba(evaluation_features, constrained=False))
                    stage_histories.append(
                        ",".join(item["stage"] for item in model.adaptation_stage_history_)
                    )
                probabilities = np.mean(np.stack(repeat_probabilities, axis=0), axis=0)
                independent_predictions = model.classes_[np.argmax(probabilities, axis=2)]
                constrained = permutation_constrained_decode(
                    probabilities,
                    temperature=model.sinkhorn_temperature,
                    sinkhorn_iterations=model.sinkhorn_iterations,
                )
                soft_predictions = model.classes_[np.argmax(constrained.probabilities, axis=2)]
                constrained_predictions = model.classes_[constrained.assignments]
                rows.append(
                    {
                        "target": target,
                        "seed": int(calibration_seed),
                        "k": int(calibration_count),
                        "source_selection_mode": source_selection_mode,
                        "n_source_participants": len(selected_sources),
                        "source_participants": ",".join(selected_sources),
                        "source_physical_classes": ",".join(
                            str(value) for value in base_models[0].physical_classes_.tolist()
                        ),
                        "target_physical_classes": ",".join(
                            str(value) for value in model.target_physical_codes_.tolist()
                        ),
                        "n_model_repeats": int(model_repeats),
                        "n_source_trials": int(source_packed.features.shape[0]),
                        "n_calibration_trials": int(split.calibration_indices.size),
                        "n_evaluation_trials": int(split.evaluation_indices.size),
                        "n_evaluation_events": int(evaluation_labels.size),
                        "independent_accuracy": float(np.mean(independent_predictions == evaluation_labels)),
                        "soft_assignment_accuracy": float(np.mean(soft_predictions == evaluation_labels)),
                        "permutation_accuracy": float(np.mean(constrained_predictions == evaluation_labels)),
                        "source_validation_mode": model.source_validation_mode_,
                        "target_validation_mode": model.target_validation_mode_,
                        "adaptation_stages": "|".join(stage_histories),
                        "pca_components_requested": None if pca_components is None else int(pca_components),
                        "pca_components_effective": int(source_packed.features.shape[2]),
                    }
                )

    per_seed = pd.DataFrame(rows).sort_values(["target", "k", "seed"]).reset_index(drop=True)
    per_target = (
        per_seed.groupby(["target", "k"], as_index=False)
        .agg(
            independent_accuracy=("independent_accuracy", "mean"),
            soft_assignment_accuracy=("soft_assignment_accuracy", "mean"),
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
        soft_mean, soft_sem = _mean_sem(frame["soft_assignment_accuracy"].to_numpy())
        permutation_mean, permutation_sem = _mean_sem(frame["permutation_accuracy"].to_numpy())
        julia_reference = JULIA_FULL_FINETUNE_ACCURACY.get(int(calibration_count), float("nan"))
        summary_rows.append(
            {
                "k": int(calibration_count),
                "n_targets": int(frame.shape[0]),
                "independent_accuracy_mean": independent_mean,
                "independent_accuracy_sem": independent_sem,
                "soft_assignment_accuracy_mean": soft_mean,
                "soft_assignment_accuracy_sem": soft_sem,
                "permutation_accuracy_mean": permutation_mean,
                "permutation_accuracy_sem": permutation_sem,
                "julia_full_finetune_accuracy": julia_reference,
                "independent_delta_vs_julia": independent_mean - julia_reference,
                "soft_assignment_delta_vs_julia": soft_mean - julia_reference,
                "permutation_delta_vs_julia": permutation_mean - julia_reference,
                "independent_outperforms_julia": bool(independent_mean > julia_reference),
                "soft_assignment_outperforms_julia": bool(soft_mean > julia_reference),
                "permutation_outperforms_julia": bool(permutation_mean > julia_reference),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("k").reset_index(drop=True)
    metadata = {
        "participants": list(participants),
        "target_participants": list(targets),
        "calibration_counts": [int(value) for value in calibration_counts],
        "calibration_seeds": [int(value) for value in calibration_seeds],
        "event_positions": [int(value) for value in event_positions],
        "source_selection_mode": source_selection_mode,
        "source_selection_seed": int(source_selection_seed),
        "source_selection": {
            target: list(sources) for target, sources in source_selection_registry.items()
        },
        "variable_physical_codes": {
            subject: list(values) for subject, values in variable_codes_by_subject.items()
        },
        "pca_components": None if pca_components is None else int(pca_components),
        "model_repeats": int(model_repeats),
        "model_kwargs": model_config,
        "source_objective": "global_physical_finger_identity",
        "target_objective": "participant_local_four_class_finger",
        "target_physical_mapping_source": "target_calibration_rows_only",
        "evaluation_unit": "finger_event",
        "seed_aggregation": "mean_within_target_then_population_mean_and_sem",
        "evaluation_pool": "complement_of_maximum_calibration_pool_for_all_k",
        "sequence_id_used_as_feature": False,
        "sequence_id_used_for_calibration_stratification": True,
    }
    return per_seed, per_target, summary, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--targets")
    parser.add_argument(
        "--calibration-counts",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_COUNTS),
    )
    parser.add_argument(
        "--calibration-seeds",
        default=",".join(str(value) for value in DEFAULT_CALIBRATION_SEEDS),
    )
    parser.add_argument("--event-positions", default="2,3,4,5")
    parser.add_argument(
        "--source-selection-mode",
        choices=SOURCE_SELECTION_MODES,
        default="deterministic",
    )
    parser.add_argument("--n-source-participants", type=int, default=9)
    parser.add_argument("--source-selection-seed", type=int, default=13)
    parser.add_argument("--source-map-json")
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--model-repeats", type=int, default=1)
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
    parser.add_argument("--sinkhorn-loss-weight", type=float, default=0.5)
    parser.add_argument("--assignment-loss-weight", type=float, default=0.1)
    parser.add_argument("--sinkhorn-temperature", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cache = load_katja_feature_cache(args.feature_cache)
    per_seed, per_target, summary, metadata = run_katja_physical_finger_sequence_benchmark(
        cache,
        participants=_parse_csv_values(args.participants),
        target_participants=None if args.targets is None else _parse_csv_values(args.targets),
        calibration_counts=_parse_csv_values(args.calibration_counts, cast=int),
        calibration_seeds=_parse_csv_values(args.calibration_seeds, cast=int),
        event_positions=_parse_csv_values(args.event_positions, cast=int),
        source_selection_mode=args.source_selection_mode,
        n_source_participants=args.n_source_participants,
        source_selection_seed=args.source_selection_seed,
        source_map=_load_source_map(args.source_map_json),
        pca_components=None if args.no_pca else args.pca_components,
        model_repeats=args.model_repeats,
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
            "sinkhorn_loss_weight": args.sinkhorn_loss_weight,
            "assignment_loss_weight": args.assignment_loss_weight,
            "sinkhorn_temperature": args.sinkhorn_temperature,
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "katja_physical_finger_per_seed.csv", index=False)
    per_target.to_csv(output_dir / "katja_physical_finger_per_target.csv", index=False)
    summary.to_csv(output_dir / "katja_physical_finger_summary.csv", index=False)
    (output_dir / "katja_physical_finger_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
