"""Evaluate source/model probability ensembles on the Katja development target.

This script deliberately uses only the designated development participant. It
averages unconstrained class probabilities from independently trained source
models before computing either independent event predictions or one-to-one
Hungarian trial assignments.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from neureptrace._katja_finger_sequence_support import (
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_PARTICIPANTS,
    _composite_trial_ids,
    _constant_trial_values,
    _fit_source_preprocessor,
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

DEVELOPMENT_TARGET = "s05"
CALIBRATION_COUNTS = (15, 20)
CURRENT_SELECTED_K20 = 0.5251572327044025
BASELINE_REPRODUCTION_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class SourceSpaceSpec:
    """Source-selection and fold-local preprocessing specification."""

    name: str
    n_source_participants: int
    source_selection_seed: int


@dataclass(slots=True)
class PreparedSourceSpace:
    """Packed source and target tensors in one source-fitted feature space."""

    spec: SourceSpaceSpec
    selected_sources: tuple[str, ...]
    source_features: np.ndarray
    source_labels: np.ndarray
    source_subjects: np.ndarray
    target_features: np.ndarray
    target_labels: np.ndarray
    target_trial_ids: np.ndarray
    target_trial_strata: np.ndarray


@dataclass(frozen=True, slots=True)
class MemberSpec:
    """One model seed within a prepared source feature space."""

    name: str
    source_space: str
    seed_variant: int


def _source_space_specs() -> dict[str, SourceSpaceSpec]:
    return {
        "nine_s13": SourceSpaceSpec("nine_s13", 9, 13),
        "nine_s29": SourceSpaceSpec("nine_s29", 9, 29),
        "nine_s47": SourceSpaceSpec("nine_s47", 9, 47),
        "all16": SourceSpaceSpec("all16", 16, 13),
    }


def _member_specs() -> dict[str, MemberSpec]:
    return {
        "nine_s13_v0": MemberSpec("nine_s13_v0", "nine_s13", 0),
        "nine_s13_v1": MemberSpec("nine_s13_v1", "nine_s13", 1),
        "nine_s13_v2": MemberSpec("nine_s13_v2", "nine_s13", 2),
        "nine_s29_v0": MemberSpec("nine_s29_v0", "nine_s29", 0),
        "nine_s47_v0": MemberSpec("nine_s47_v0", "nine_s47", 0),
        "all16_v0": MemberSpec("all16_v0", "all16", 0),
        "all16_v1": MemberSpec("all16_v1", "all16", 1),
        "all16_v2": MemberSpec("all16_v2", "all16", 2),
    }


def _configurations() -> dict[str, tuple[str, ...]]:
    return {
        "nine_single_protocol": ("nine_s13_v0",),
        "all16_single_protocol": ("all16_v0",),
        "nine_model_ensemble3": (
            "nine_s13_v0",
            "nine_s13_v1",
            "nine_s13_v2",
        ),
        "nine_source_ensemble3": (
            "nine_s13_v0",
            "nine_s29_v0",
            "nine_s47_v0",
        ),
        "all16_model_ensemble3": (
            "all16_v0",
            "all16_v1",
            "all16_v2",
        ),
        "hybrid_source_model_ensemble6": (
            "nine_s13_v0",
            "nine_s29_v0",
            "nine_s47_v0",
            "all16_v0",
            "all16_v1",
            "all16_v2",
        ),
    }


def _prepare_rows(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    subjects = cache["subjects"].astype(str)
    press_positions = cache["press_positions"].astype(int)
    included = (
        cache["correct_order"].astype(bool)
        & np.isin(press_positions, np.asarray((2, 3, 4, 5)))
        & np.isin(subjects, np.asarray(DEFAULT_PARTICIPANTS))
    )
    labels = derive_participant_local_finger_labels(
        subjects,
        cache["finger_codes"],
        included_mask=included,
        expected_classes=4,
    )
    return {
        "features": cache["features"][included],
        "subjects": subjects[included],
        "trial_ids": cache["trial_ids"][included],
        "press_positions": press_positions[included],
        "sequence_ids": cache["sequence_ids"][included],
        "labels": labels[included],
    }


def _model_seed(calibration_seed: int, seed_variant: int) -> int:
    """Keep variant zero identical to the established benchmark protocol."""

    return int(calibration_seed) + 10_000 * int(seed_variant)


def _model_kwargs(model_seed: int) -> dict[str, Any]:
    return {
        "hidden_units": 96,
        "num_layers": 2,
        "num_heads": 4,
        "adapter_rank": 8,
        "source_max_epochs": 120,
        "meta_epochs": 2,
        "meta_support_trials": 4,
        "meta_query_trials": 4,
        "meta_inner_steps": 5,
        "adapter_steps": 80,
        "last_block_steps": 60,
        "full_finetune_steps": 60,
        "batch_size": 64,
        "device": "cuda",
        "refit_source_on_all": True,
        "refit_target_on_all": True,
        "sinkhorn_loss_weight": 0.5,
        "assignment_loss_weight": 0.1,
        "sinkhorn_temperature": 0.5,
        "random_state": int(model_seed),
    }


def _prepare_source_space(
    rows: dict[str, np.ndarray],
    spec: SourceSpaceSpec,
) -> PreparedSourceSpace:
    subjects = rows["subjects"].astype(str)
    target_mask = subjects == DEVELOPMENT_TARGET
    selected_sources = _stable_source_selection(
        DEFAULT_PARTICIPANTS,
        target=DEVELOPMENT_TARGET,
        n_sources=spec.n_source_participants,
        seed=spec.source_selection_seed,
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
        raise RuntimeError("Source and target labels are required for development scoring.")
    source_trial_subjects = _constant_trial_values(
        subjects[source_mask],
        source_packed.row_indices,
        name="source subject",
    )
    target_trial_strata = _constant_trial_values(
        rows["sequence_ids"][target_mask],
        target_packed.row_indices,
        name="target sequence ID",
    )
    return PreparedSourceSpace(
        spec=spec,
        selected_sources=selected_sources,
        source_features=source_packed.features,
        source_labels=source_packed.labels,
        source_subjects=source_trial_subjects,
        target_features=target_packed.features,
        target_labels=target_packed.labels,
        target_trial_ids=target_packed.trial_ids,
        target_trial_strata=target_trial_strata,
    )


def _validate_source_space_alignment(
    spaces: dict[str, PreparedSourceSpace],
) -> PreparedSourceSpace:
    reference = next(iter(spaces.values()))
    for space in spaces.values():
        if not np.array_equal(space.target_trial_ids, reference.target_trial_ids):
            raise RuntimeError("Source spaces have different target trial ordering.")
        if not np.array_equal(space.target_labels, reference.target_labels):
            raise RuntimeError("Source spaces have different target labels.")
        if not np.array_equal(
            space.target_trial_strata,
            reference.target_trial_strata,
        ):
            raise RuntimeError("Source spaces have different target strata.")
    return reference


def _fit_source_member(
    space: PreparedSourceSpace,
    *,
    model_seed: int,
) -> TorchProgressiveSequenceClassifier:
    model = TorchProgressiveSequenceClassifier(**_model_kwargs(model_seed))
    model.fit_source(
        space.source_features,
        space.source_labels,
        source_subjects=space.source_subjects,
    )
    return model


def run_sweep(
    cache: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = _prepare_rows(cache)
    space_specs = _source_space_specs()
    spaces = {
        name: _prepare_source_space(rows, spec)
        for name, spec in space_specs.items()
    }
    reference = _validate_source_space_alignment(spaces)
    member_specs = _member_specs()
    configurations = _configurations()
    needed_names = sorted({name for names in configurations.values() for name in names})

    result_rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}
    model_seeds: dict[str, int] = {}

    for calibration_seed in DEFAULT_CALIBRATION_SEEDS:
        splits = select_nested_trial_calibration_splits(
            reference.target_trial_strata,
            calibration_counts=CALIBRATION_COUNTS,
            max_per_stratum=max(CALIBRATION_COUNTS),
            min_evaluation_per_stratum=1,
            seed=int(calibration_seed),
            context=("katja_finger", DEVELOPMENT_TARGET),
        )
        fitted_members: dict[str, TorchProgressiveSequenceClassifier] = {}
        for name in needed_names:
            spec = member_specs[name]
            model_seed = _model_seed(calibration_seed, spec.seed_variant)
            started = time.time()
            fitted_members[name] = _fit_source_member(
                spaces[spec.source_space],
                model_seed=model_seed,
            )
            key = f"seed{calibration_seed}:{name}"
            training_seconds[key] = time.time() - started
            model_seeds[key] = model_seed
            print(
                f"Prepared {key} in {training_seconds[key]:.1f} s; "
                f"sources={','.join(spaces[spec.source_space].selected_sources)}"
            )

        for calibration_count in CALIBRATION_COUNTS:
            split = splits[int(calibration_count)]
            member_probabilities: dict[str, np.ndarray] = {}
            for name in needed_names:
                spec = member_specs[name]
                space = spaces[spec.source_space]
                model = copy.deepcopy(fitted_members[name])
                model.adapt_target(
                    space.target_features[split.calibration_indices],
                    space.target_labels[split.calibration_indices],
                    target_strata=space.target_trial_strata[
                        split.calibration_indices
                    ],
                )
                member_probabilities[name] = model.predict_proba(
                    space.target_features[split.evaluation_indices],
                    constrained=False,
                )
                del model
                torch.cuda.empty_cache()

            evaluation_labels = reference.target_labels[split.evaluation_indices]
            for configuration, member_names in configurations.items():
                probabilities = np.mean(
                    np.stack(
                        [member_probabilities[name] for name in member_names],
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
                        "configuration": configuration,
                        "seed": int(calibration_seed),
                        "k": int(calibration_count),
                        "ensemble_size": len(member_names),
                        "members": ",".join(member_names),
                        "n_evaluation_trials": int(split.evaluation_indices.size),
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

    per_seed = pd.DataFrame(result_rows).sort_values(
        ["configuration", "k", "seed"]
    )
    summary = (
        per_seed.groupby(["configuration", "k"], as_index=False)
        .agg(
            independent_accuracy=("independent_accuracy", "mean"),
            independent_std=("independent_accuracy", "std"),
            permutation_accuracy=("permutation_accuracy", "mean"),
            permutation_std=("permutation_accuracy", "std"),
            ensemble_size=("ensemble_size", "first"),
        )
        .sort_values(["configuration", "k"])
        .reset_index(drop=True)
    )
    pivot = summary.pivot(
        index="configuration",
        columns="k",
        values=["independent_accuracy", "permutation_accuracy"],
    )
    ranking = pd.DataFrame(
        {
            "configuration": pivot.index,
            "independent_k15": pivot["independent_accuracy"][15].to_numpy(),
            "independent_k20": pivot["independent_accuracy"][20].to_numpy(),
            "permutation_k20": pivot["permutation_accuracy"][20].to_numpy(),
        }
    ).sort_values(
        ["independent_k20", "independent_k15", "permutation_k20"],
        ascending=False,
    )
    selected = str(ranking.iloc[0]["configuration"])
    baseline_k20 = float(
        ranking.loc[
            ranking["configuration"] == "nine_single_protocol",
            "independent_k20",
        ].iloc[0]
    )
    baseline_error = baseline_k20 - CURRENT_SELECTED_K20
    if abs(baseline_error) > BASELINE_REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            "The single-member protocol baseline did not reproduce the prior "
            f"s05 result: observed={baseline_k20:.6f}, "
            f"expected={CURRENT_SELECTED_K20:.6f}."
        )
    metadata = {
        "development_target": DEVELOPMENT_TARGET,
        "calibration_counts": list(CALIBRATION_COUNTS),
        "calibration_seeds": list(DEFAULT_CALIBRATION_SEEDS),
        "current_selected_independent_k20": CURRENT_SELECTED_K20,
        "reproduced_baseline_independent_k20": baseline_k20,
        "baseline_reproduction_error": baseline_error,
        "baseline_reproduction_tolerance": BASELINE_REPRODUCTION_TOLERANCE,
        "selection_endpoint": (
            "five-seed mean independent k20 accuracy; independent k15 and "
            "permutation k20 as tie-breakers"
        ),
        "selected_configuration": selected,
        "selected_members": list(configurations[selected]),
        "selected_independent_k20": float(ranking.iloc[0]["independent_k20"]),
        "improvement_over_reproduced_baseline_k20": float(
            ranking.iloc[0]["independent_k20"] - baseline_k20
        ),
        "member_training_seconds": training_seconds,
        "member_model_seeds": model_seeds,
        "source_spaces": {
            name: {
                "n_source_participants": spec.n_source_participants,
                "source_selection_seed": spec.source_selection_seed,
                "selected_sources": list(spaces[name].selected_sources),
            }
            for name, spec in space_specs.items()
        },
        "members": {
            name: {
                "source_space": spec.source_space,
                "seed_variant": spec.seed_variant,
            }
            for name, spec in member_specs.items()
        },
        "configurations": {
            name: list(member_names)
            for name, member_names in configurations.items()
        },
    }
    return per_seed, summary, ranking, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = load_katja_feature_cache(args.feature_cache)
    started = time.time()
    per_seed, summary, ranking, metadata = run_sweep(cache)
    metadata["elapsed_seconds"] = time.time() - started
    per_seed.to_csv(output / "katja_ensemble_per_seed.csv", index=False)
    summary.to_csv(output / "katja_ensemble_summary.csv", index=False)
    ranking.to_csv(output / "katja_ensemble_ranking.csv", index=False)
    (output / "katja_ensemble_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("\n=== Ensemble development ranking ===")
    print(ranking.to_string(index=False))
    print("\n=== Selection metadata ===")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
