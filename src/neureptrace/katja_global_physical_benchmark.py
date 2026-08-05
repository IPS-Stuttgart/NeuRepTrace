"""Katja calibrated decoding with globally consistent physical-finger labels.

The participant-local four-class reconstruction changes class semantics when a
participant has a different fixed first finger. This experiment instead trains a
shared five-class physical-finger head, then uses only labeled target calibration
rows to identify the four valid target classes. The excluded target class is
masked before independent or permutation-constrained evaluation.
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
    load_katja_feature_cache,
)
from neureptrace.decoding.progressive_sequence_finetune import (
    TorchProgressiveSequenceClassifier,
    pack_complete_trial_events,
    permutation_constrained_decode,
    select_nested_trial_calibration_splits,
)

if TYPE_CHECKING:
    from typing import Any

__all__ = (
    "build_arg_parser",
    "main",
    "restrict_probabilities_to_calibration_classes",
    "run_katja_global_physical_benchmark",
)


def _validate_trial_label_uniqueness(
    labels: np.ndarray,
    *,
    expected_classes: int,
    name: str,
) -> None:
    matrix = np.asarray(labels)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (trials, events).")
    if matrix.shape[1] != expected_classes:
        raise ValueError(
            f"{name} contains {matrix.shape[1]} events per trial; expected "
            f"{expected_classes}."
        )
    for trial_index, row in enumerate(matrix):
        if np.unique(row).shape[0] != expected_classes:
            raise ValueError(
                f"{name} trial {trial_index} does not contain "
                f"{expected_classes} unique physical fingers."
            )


def restrict_probabilities_to_calibration_classes(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
    calibration_labels: np.ndarray,
    *,
    expected_classes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask a global head to classes identified from calibration labels only.

    No evaluation labels are accepted by this function. Returned probabilities
    are normalized over the calibration-observed classes and preserve the input
    trial/event dimensions.
    """

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 3 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            "probabilities must be finite non-negative trial/event/class rows."
        )
    classes = np.asarray(model_classes)
    if classes.ndim != 1 or classes.shape[0] != values.shape[2]:
        raise ValueError("model_classes must match the probability columns.")
    calibration = np.asarray(calibration_labels).reshape(-1)
    allowed = np.unique(calibration)
    if allowed.shape[0] != expected_classes:
        raise ValueError(
            f"Target calibration exposes {allowed.shape[0]} physical classes; "
            f"expected {expected_classes}."
        )

    column_indices: list[int] = []
    for label in allowed.tolist():
        matches = np.flatnonzero(classes == label)
        if matches.shape[0] != 1:
            raise ValueError(
                f"Calibration class {label!r} is not represented exactly once "
                "in the fitted global head."
            )
        column_indices.append(int(matches[0]))
    restricted = values[..., np.asarray(column_indices, dtype=int)]
    row_sums = restricted.sum(axis=2, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("Restricted target probabilities must have positive mass.")
    return allowed, restricted / row_sums


def run_katja_global_physical_benchmark(
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
    """Run target-calibrated decoding in a global physical-finger label space."""

    if "finger_codes" not in cache:
        raise ValueError("Global physical-finger decoding requires finger_codes.")
    subjects = cache["subjects"].astype(str)
    trial_ids = cache["trial_ids"]
    press_positions = cache["press_positions"].astype(int)
    sequence_ids = cache["sequence_ids"]
    physical_labels = np.asarray(cache["finger_codes"])
    correct_order = cache["correct_order"].astype(bool)
    included = (
        correct_order
        & np.isin(press_positions, np.asarray(event_positions, dtype=int))
        & np.isin(subjects, np.asarray(participants))
    )
    if not np.any(included):
        raise ValueError("No rows remain after participant and event filtering.")

    features = cache["features"][included]
    subjects = subjects[included]
    trial_ids = trial_ids[included]
    press_positions = press_positions[included]
    sequence_ids = sequence_ids[included]
    physical_labels = physical_labels[included]
    source_map = {} if source_map is None else source_map
    model_config = {} if model_kwargs is None else dict(model_kwargs)
    if bool(model_config.get("enforce_permutation_labels", False)):
        raise ValueError(
            "Global five-class source training requires "
            "enforce_permutation_labels=False."
        )
    model_config["enforce_permutation_labels"] = False

    targets = participants if target_participants is None else target_participants
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("target_participants must contain unique identifiers.")
    unknown_targets = [target for target in targets if target not in participants]
    if unknown_targets:
        raise ValueError(
            f"Target participants are absent from the participant pool: "
            f"{unknown_targets!r}."
        )

    rows: list[dict[str, Any]] = []
    source_selection_registry: dict[str, tuple[str, ...]] = {}
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
        if target in selected_sources or len(set(selected_sources)) != len(
            selected_sources
        ):
            raise ValueError(
                f"Invalid source participant set for target {target!r}: "
                f"{selected_sources!r}."
            )
        source_selection_registry[target] = selected_sources
        source_mask = np.isin(subjects, np.asarray(selected_sources))
        if not np.any(source_mask):
            raise ValueError(
                f"Target {target!r} has no rows from its selected sources."
            )

        scaler, pca, source_transformed = _fit_source_preprocessor(
            features[source_mask], pca_components=pca_components
        )
        target_transformed = _transform_features(
            features[target_mask], scaler=scaler, pca=pca
        )
        source_packed = pack_complete_trial_events(
            source_transformed,
            _composite_trial_ids(subjects[source_mask], trial_ids[source_mask]),
            press_positions[source_mask],
            labels=physical_labels[source_mask],
            expected_events=len(event_positions),
            require_permutation_labels=False,
        )
        target_packed = pack_complete_trial_events(
            target_transformed,
            trial_ids[target_mask],
            press_positions[target_mask],
            labels=physical_labels[target_mask],
            expected_events=len(event_positions),
            require_permutation_labels=False,
        )
        _validate_trial_label_uniqueness(
            source_packed.labels,
            expected_classes=len(event_positions),
            name="source physical labels",
        )
        _validate_trial_label_uniqueness(
            target_packed.labels,
            expected_classes=len(event_positions),
            name="target physical labels",
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
            splits = select_nested_trial_calibration_splits(
                target_trial_strata,
                calibration_counts=calibration_counts,
                max_per_stratum=max(calibration_counts),
                min_evaluation_per_stratum=1,
                seed=int(calibration_seed),
                context=("katja_global_physical", target),
            )
            seed_model_config = {
                **model_config,
                "random_state": int(calibration_seed),
            }
            base_model = TorchProgressiveSequenceClassifier(**seed_model_config)
            base_model.fit_source(
                source_packed.features,
                source_packed.labels,
                source_subjects=source_trial_subjects,
            )
            for calibration_count in calibration_counts:
                split = splits[int(calibration_count)]
                model = copy.deepcopy(base_model)
                model.adapt_target(
                    target_packed.features[split.calibration_indices],
                    target_packed.labels[split.calibration_indices],
                    target_strata=target_trial_strata[split.calibration_indices],
                )
                evaluation_features = target_packed.features[split.evaluation_indices]
                evaluation_labels = target_packed.labels[split.evaluation_indices]
                probabilities = model.predict_proba(
                    evaluation_features, constrained=False
                )
                calibration_labels = target_packed.labels[
                    split.calibration_indices
                ]
                allowed_classes, restricted = (
                    restrict_probabilities_to_calibration_classes(
                        probabilities,
                        model.classes_,
                        calibration_labels,
                        expected_classes=len(event_positions),
                    )
                )
                independent_predictions = allowed_classes[
                    np.argmax(restricted, axis=2)
                ]
                unmasked_predictions = model.classes_[
                    np.argmax(probabilities, axis=2)
                ]
                constrained = permutation_constrained_decode(
                    restricted,
                    temperature=model.sinkhorn_temperature,
                    sinkhorn_iterations=model.sinkhorn_iterations,
                )
                constrained_predictions = allowed_classes[
                    constrained.assignments
                ]
                rows.append(
                    {
                        "target": target,
                        "seed": int(calibration_seed),
                        "k": int(calibration_count),
                        "n_source_participants": len(selected_sources),
                        "source_participants": ",".join(selected_sources),
                        "n_source_trials": int(source_packed.features.shape[0]),
                        "n_source_global_classes": int(model.classes_.shape[0]),
                        "target_allowed_physical_classes": ",".join(
                            map(str, allowed_classes.tolist())
                        ),
                        "n_calibration_trials": int(
                            split.calibration_indices.size
                        ),
                        "n_evaluation_trials": int(
                            split.evaluation_indices.size
                        ),
                        "n_evaluation_events": int(evaluation_labels.size),
                        "independent_accuracy": float(
                            np.mean(independent_predictions == evaluation_labels)
                        ),
                        "global_unmasked_accuracy": float(
                            np.mean(unmasked_predictions == evaluation_labels)
                        ),
                        "permutation_accuracy": float(
                            np.mean(constrained_predictions == evaluation_labels)
                        ),
                        "source_validation_mode": model.source_validation_mode_,
                        "target_validation_mode": model.target_validation_mode_,
                        "adaptation_stages": ",".join(
                            item["stage"]
                            for item in model.adaptation_stage_history_
                        ),
                        "pca_components_requested": (
                            None
                            if pca_components is None
                            else int(pca_components)
                        ),
                        "pca_components_effective": int(
                            source_packed.features.shape[2]
                        ),
                    }
                )

    per_seed = pd.DataFrame(rows).sort_values(
        ["target", "k", "seed"]
    ).reset_index(drop=True)
    per_target = (
        per_seed.groupby(["target", "k"], as_index=False)
        .agg(
            independent_accuracy=("independent_accuracy", "mean"),
            global_unmasked_accuracy=("global_unmasked_accuracy", "mean"),
            permutation_accuracy=("permutation_accuracy", "mean"),
            n_seeds=("seed", "nunique"),
            n_evaluation_trials=("n_evaluation_trials", "first"),
            n_evaluation_events=("n_evaluation_events", "first"),
        )
        .sort_values(["target", "k"])
        .reset_index(drop=True)
    )
    summary_rows: list[dict[str, Any]] = []
    for calibration_count, frame in per_target.groupby("k", sort=True):
        independent_mean, independent_sem = _mean_sem(
            frame["independent_accuracy"].to_numpy()
        )
        unmasked_mean, unmasked_sem = _mean_sem(
            frame["global_unmasked_accuracy"].to_numpy()
        )
        permutation_mean, permutation_sem = _mean_sem(
            frame["permutation_accuracy"].to_numpy()
        )
        julia_reference = JULIA_FULL_FINETUNE_ACCURACY.get(
            int(calibration_count), float("nan")
        )
        summary_rows.append(
            {
                "k": int(calibration_count),
                "n_targets": int(frame.shape[0]),
                "independent_accuracy_mean": independent_mean,
                "independent_accuracy_sem": independent_sem,
                "global_unmasked_accuracy_mean": unmasked_mean,
                "global_unmasked_accuracy_sem": unmasked_sem,
                "permutation_accuracy_mean": permutation_mean,
                "permutation_accuracy_sem": permutation_sem,
                "julia_full_finetune_accuracy": julia_reference,
                "independent_delta_vs_julia": (
                    independent_mean - julia_reference
                ),
                "permutation_delta_vs_julia": (
                    permutation_mean - julia_reference
                ),
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
        "event_positions": [int(value) for value in event_positions],
        "source_selection_seed": int(source_selection_seed),
        "source_selection": {
            target: list(sources)
            for target, sources in source_selection_registry.items()
        },
        "pca_components": (
            None if pca_components is None else int(pca_components)
        ),
        "model_kwargs": model_config,
        "training_label_space": "global_physical_finger",
        "evaluation_label_space": "global_physical_finger",
        "target_class_mask_source": "target_calibration_labels_only",
        "evaluation_labels_used_for_target_class_mask": False,
        "independent_endpoint": (
            "eventwise_argmax_after_calibration_only_target_class_mask"
        ),
        "permutation_endpoint": (
            "hungarian_assignment_after_calibration_only_target_class_mask"
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
        run_katja_global_physical_benchmark(
            cache,
            participants=_parse_csv_values(args.participants),
            target_participants=(
                None
                if args.targets is None
                else _parse_csv_values(args.targets)
            ),
            calibration_counts=_parse_csv_values(
                args.calibration_counts, cast=int
            ),
            calibration_seeds=_parse_csv_values(
                args.calibration_seeds, cast=int
            ),
            n_source_participants=args.n_source_participants,
            source_selection_seed=args.source_selection_seed,
            source_map=_load_source_map(args.source_map_json),
            pca_components=(
                None if args.no_pca else args.pca_components
            ),
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "katja_global_physical_per_seed.csv", index=False)
    per_target.to_csv(
        output_dir / "katja_global_physical_per_target.csv", index=False
    )
    summary.to_csv(output_dir / "katja_global_physical_summary.csv", index=False)
    (output_dir / "katja_global_physical_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
