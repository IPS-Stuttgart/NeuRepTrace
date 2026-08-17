"""Maximum-accuracy Katja sliding-window benchmark with strict label boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neureptrace.decoding.katja_trial_context import TorchKatjaTrialContextRefiner
from neureptrace.decoding.katja_window_structure import (
    TargetCalibrationPartition,
    causal_trial_decode,
    ensemble_prediction_bundles,
    estimate_state_stay_probabilities,
    learn_finger_templates,
    structured_trial_decode,
    write_partition_audit,
    write_prediction_bundle,
)
from neureptrace.decoding.progressive_temporal_window_finetune import (
    TorchProgressiveTemporalWindowClassifier,
)
from neureptrace.katja_julia_window_benchmark import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    JULIA_SUBJECTS,
    _append_row,
    _load_npz_metadata,
    _metric_row,
    _parse_csv,
    _stable_seed,
    _utc_timestamp,
    _write_status,
    prepare_raw_window_memmap,
    prepare_subject_sensor_moments,
    relabel_minimum_overlap,
    select_nested_trial_splits,
    summarize_results,
)


PUSH_METHODS = (
    "hierarchical_source_only",
    "hierarchical_tcn_single",
    "hierarchical_tcn_ensemble",
    "hierarchical_tcn_ensemble_structured",
    "hierarchical_tcn_ensemble_causal",
    "trial_transformer_single",
    "trial_transformer_ensemble",
    "trial_transformer_ensemble_structured",
    "trial_transformer_causal_single",
    "trial_transformer_causal_ensemble",
    "hybrid_ensemble",
    "hybrid_ensemble_structured",
)

ENSEMBLE_METHODS = (
    "hierarchical_source_only",
    "hierarchical_tcn_ensemble",
    "hierarchical_tcn_ensemble_structured",
    "hierarchical_tcn_ensemble_causal",
    "trial_transformer_ensemble",
    "trial_transformer_ensemble_structured",
    "trial_transformer_causal_ensemble",
    "hybrid_ensemble",
    "hybrid_ensemble_structured",
)

MEMBER_METHODS = (
    "hierarchical_tcn_single",
    "trial_transformer_single",
    "trial_transformer_causal_single",
)

PREDICTION_FAMILIES = (
    "hierarchical_source_only",
    "hierarchical_tcn",
    "trial_transformer_offline",
    "trial_transformer_causal",
)

DIRECT_JULIA_COMPARISON_METHODS = (
    "hierarchical_source_only",
    "hierarchical_tcn_single",
    "hierarchical_tcn_ensemble",
)

CLASSIFICATION_INFORMATION_BOUNDARY = {
    "evaluation_inputs": [
        "Julia-supplied cached MEG window samples",
        "held-out participant identity for selecting the target adapter",
        "trial membership",
        "chronological cache-row order within each trial",
    ],
    "source_and_calibration_supervision": [
        "finger identity",
        "sequence identity",
        "press order",
        "press overlap and per-finger occupancy",
    ],
    "forbidden_evaluation_inputs": [
        "target evaluation finger labels",
        "target evaluation sequence labels",
        "target evaluation press-order labels",
        "target evaluation overlap labels or press ratios",
        "ground-truth press timestamps or cue durations outside the supplied cache",
    ],
    "structured_inference": (
        "uses calibration-observed templates, source-plus-calibration duration priors, "
        "and model-predicted auxiliary heads; it does not use evaluation annotations"
    ),
}

INDEPENDENT_DESIGN_PROVENANCE = {
    "shared_external_artifacts": [
        "Julia-supplied sliding-window cache",
        "cache schema and task documentation",
        "documented minimum-overlap relabel rule at tau=0.2",
    ],
    "collaborator_model_architecture_received": False,
    "collaborator_training_or_adaptation_code_received": False,
    "collaborator_split_function_received": False,
    "copied_collaborator_model_components": False,
    "neureptrace_method_components": [
        "hierarchical press/finger TCN",
        "source-specific and target low-rank adapters",
        "calibration-only adaptation selection",
        "trial-context Transformer",
        "five-seed probability ensemble",
        "calibration-observed-template Viterbi decoder",
    ],
    "scope": (
        "The cache and documented endpoint rule are shared to make the evaluation "
        "comparable; model architecture, optimization, adaptation, splitting, ensembling, "
        "and structured decoding are independent NeuRepTrace implementations."
    ),
}


def validate_baseline_reproduction(path: str | Path) -> dict[str, Any]:
    """Require the validated 53.36/56.03% baseline before accepting a push run."""

    root = Path(path).expanduser().resolve()
    validation_path = root / "validation.json"
    summary_path = root / "summary_subject_sem.csv"
    if not validation_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Baseline reproduction requires validation.json and summary_subject_sem.csv in {root}"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("all_required_checks_pass") is not True:
        raise RuntimeError("Baseline reproduction validation did not pass")
    rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))

    def accuracy(k: int) -> tuple[float, int]:
        matches = [
            row
            for row in rows
            if row.get("method") == "progressive_full"
            and int(row.get("k_trials_per_sequence", -1)) == int(k)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Baseline summary has {len(matches)} progressive_full rows at k={k}")
        return float(matches[0]["mean_accuracy_raw_labels"]), int(matches[0]["n_subjects"])

    k10, n10 = accuracy(10)
    k20, n20 = accuracy(20)
    expected = {"k10": 0.5335594321714971, "k20": 0.5602636869077369}
    if n10 != 10 or n20 != 9 or abs(k10 - expected["k10"]) > 1e-12 or abs(k20 - expected["k20"]) > 1e-12:
        raise RuntimeError(
            "Baseline reproduction differs from the validated reference: "
            f"k10={k10:.15f} n={n10}, k20={k20:.15f} n={n20}"
        )
    return {
        "path": str(root),
        "validation_passed": True,
        "k10_all_ten_accuracy": k10,
        "k20_common_nine_accuracy": k20,
    }


def _write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    payload: dict[str, Any] = {}
    for name, value in sorted(vars(args).items()):
        if isinstance(value, Path):
            payload[name] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            payload[name] = value
        elif isinstance(value, (list, tuple)):
            payload[name] = list(value)
        else:
            payload[name] = repr(value)
    payload["written_at"] = _utc_timestamp()
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _subject_moment_arrays(moments: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    domains = np.asarray(moments["subjects"]).reshape(-1)
    counts = np.asarray(moments["counts"], dtype=np.float64).reshape(-1, 1)
    means = np.asarray(moments["sums"], dtype=np.float64) / counts
    variance = np.maximum(
        np.asarray(moments["squared_sums"], dtype=np.float64) / counts - np.square(means),
        1e-12,
    )
    return domains, means.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _source_pooled_moments(
    moments: dict[str, np.ndarray], source_domains: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positions = {int(value): index for index, value in enumerate(moments["subjects"].tolist())}
    rows = np.asarray([positions[int(value)] for value in source_domains], dtype=int)
    count = float(np.sum(moments["counts"][rows]))
    total = np.sum(moments["sums"][rows], axis=0)
    squared = np.sum(moments["squared_sums"][rows], axis=0)
    mean = total / count
    variance = np.maximum(squared / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _masked_source_labels(
    source_indices: np.ndarray,
    *,
    training_labels: np.ndarray,
    raw_finger_labels: np.ndarray,
    sequence_ids: np.ndarray,
    order_labels: np.ndarray,
    overlap_targets: np.ndarray,
    press_ratios: np.ndarray,
) -> dict[str, np.ndarray]:
    finger = np.zeros_like(training_labels)
    validation_finger = np.zeros_like(raw_finger_labels)
    sequence = np.zeros_like(sequence_ids)
    order = np.zeros_like(order_labels)
    overlap = np.zeros_like(overlap_targets)
    ratios = np.zeros_like(press_ratios, dtype=np.float32)
    ratios[:, 0] = 1.0
    finger[source_indices] = training_labels[source_indices]
    validation_finger[source_indices] = raw_finger_labels[source_indices]
    sequence[source_indices] = sequence_ids[source_indices]
    order[source_indices] = order_labels[source_indices]
    overlap[source_indices] = overlap_targets[source_indices]
    ratios[source_indices] = press_ratios[source_indices]
    return {
        "finger": finger,
        "validation_finger": validation_finger,
        "sequence": sequence,
        "order": order,
        "overlap": overlap,
        "press_ratios": ratios,
    }


def _fit_hierarchical_source_model(
    args: argparse.Namespace,
    *,
    adapter_rank: int,
    adapter_kind: str,
    random_state: int,
    window_store: np.ndarray,
    source_indices: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    source_labels: dict[str, np.ndarray],
    pooled_mean: np.ndarray,
    pooled_std: np.ndarray,
    moment_domains: np.ndarray,
    subject_means: np.ndarray,
    subject_stds: np.ndarray,
    source_epochs: int | None = None,
    source_refit_all: bool = True,
    source_validation_domain: Any | None = None,
) -> TorchProgressiveTemporalWindowClassifier:
    source_domain_values = np.unique(subject_ids[source_indices])
    source_moment_mask = np.isin(moment_domains, source_domain_values)
    return TorchProgressiveTemporalWindowClassifier(
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        adapter_rank=int(adapter_rank),
        adapter_alpha=float(adapter_rank),
        adapter_kind=str(adapter_kind),
        source_epochs=args.source_epochs if source_epochs is None else int(source_epochs),
        source_validation_patience=args.source_validation_patience,
        source_refit_all=bool(source_refit_all),
        source_validation_domain=source_validation_domain,
        adapter_steps=args.adapter_steps,
        last_block_steps=args.last_block_steps,
        full_finetune_steps=args.full_finetune_steps,
        batch_size=args.batch_size,
        sequence_loss_weight=args.sequence_loss_weight,
        order_loss_weight=args.order_loss_weight,
        overlap_loss_weight=args.overlap_loss_weight,
        hierarchical=True,
        balanced_sampling=True,
        subject_specific_normalization=True,
        source_specific_adapters=True,
        source_selection_metric="finger_accuracy",
        random_state=int(random_state),
        device=args.device,
    ).fit_source(
        window_store,
        source_indices=source_indices,
        source_domains=subject_ids,
        finger_labels=source_labels["finger"],
        sequence_labels=source_labels["sequence"],
        order_labels=source_labels["order"],
        overlap_targets=source_labels["overlap"],
        sensor_mean=pooled_mean,
        sensor_std=pooled_std,
        press_ratios=source_labels["press_ratios"],
        trial_ids=trial_ids,
        subject_sensor_domains=moment_domains[source_moment_mask],
        subject_sensor_means=subject_means[source_moment_mask],
        subject_sensor_stds=subject_stds[source_moment_mask],
        validation_finger_labels=source_labels["validation_finger"],
    )


def _parse_adapter_configurations(raw: str) -> tuple[tuple[str, int], ...]:
    configurations: list[tuple[str, int]] = []
    for value in _parse_csv(raw):
        try:
            kind, rank = value.split(":", maxsplit=1)
        except ValueError as error:
            raise ValueError("Adapter configurations must use kind:rank") from error
        if kind not in {"low_rank", "channel_affine_residual"} or int(rank) not in {8, 16, 32}:
            raise ValueError(f"Unsupported adapter configuration {value!r}")
        configuration = (kind, int(rank))
        if configuration not in configurations:
            configurations.append(configuration)
    return tuple(configurations)


def _screen_adapter_for_outer_target(
    args: argparse.Namespace,
    *,
    target: str,
    output_dir: Path,
    model_seed: int,
    window_store: np.ndarray,
    subject_ids: np.ndarray,
    trial_ids: np.ndarray,
    training_labels: np.ndarray,
    raw_finger_labels: np.ndarray,
    sequence_ids: np.ndarray,
    order_labels: np.ndarray,
    overlap_targets: np.ndarray,
    press_ratios: np.ndarray,
    moments: dict[str, np.ndarray],
    moment_domains: np.ndarray,
    subject_means: np.ndarray,
    subject_stds: np.ndarray,
) -> dict[str, Any]:
    """Choose an adapter by nested source LOSO without touching the outer target."""

    screen_dir = output_dir / "adapter_screens" / f"target={target}"
    screen_dir.mkdir(parents=True, exist_ok=True)
    selected_path = screen_dir / "selected_adapter_config.json"
    partial_path = screen_dir / "source_adapter_screen.partial.csv"
    screen_status_path = screen_dir / "status.json"
    if bool(args.resume) and selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if selected.get("outer_test_subject") != target:
            raise RuntimeError(f"Resumed adapter screen does not belong to {target}")
        return selected
    if not bool(args.resume) and partial_path.exists():
        partial_path.unlink()

    target_index = JULIA_SUBJECTS.index(target)
    available_source_domains = np.asarray(
        [index for index in range(len(JULIA_SUBJECTS)) if index != target_index]
    )
    validation_domains = available_source_domains.tolist()
    screen_fold_limit = getattr(args, "adapter_screen_fold_limit", None)
    if screen_fold_limit is not None:
        validation_domains = validation_domains[: int(screen_fold_limit)]

    completed: set[tuple[str, int, int]] = set()
    if bool(args.resume) and partial_path.exists():
        prior = pd.read_csv(partial_path)
        completed = set(
            zip(
                prior["adapter_kind"].astype(str),
                prior["adapter_rank"].astype(int),
                prior["heldout_source_domain"].astype(int),
                strict=True,
            )
        )
    for adapter_kind, adapter_rank in _parse_adapter_configurations(
        getattr(
            args,
            "adapter_configurations",
            "low_rank:8,low_rank:16,low_rank:32,channel_affine_residual:16",
        )
    ):
        for validation_domain in validation_domains:
            completion_key = (adapter_kind, int(adapter_rank), int(validation_domain))
            if completion_key in completed:
                continue
            _write_status(
                screen_status_path,
                state="fit_start",
                outer_test_subject=target,
                adapter_kind=adapter_kind,
                adapter_rank=int(adapter_rank),
                heldout_source_domain=int(validation_domain),
                n_completed=len(completed),
            )
            started = time.monotonic()
            source_domains = available_source_domains[
                available_source_domains != validation_domain
            ]
            source_indices = np.flatnonzero(np.isin(subject_ids, source_domains))
            pseudo_target_indices = np.flatnonzero(subject_ids == validation_domain)
            if (
                np.any(subject_ids[source_indices] == target_index)
                or np.any(subject_ids[source_indices] == validation_domain)
            ):
                raise RuntimeError(
                    "Outer-target or pseudo-target rows entered source fitting"
                )
            pooled_mean, pooled_std = _source_pooled_moments(moments, source_domains)
            source_labels = _masked_source_labels(
                source_indices,
                training_labels=training_labels,
                raw_finger_labels=raw_finger_labels,
                sequence_ids=sequence_ids,
                order_labels=order_labels,
                overlap_targets=overlap_targets,
                press_ratios=press_ratios,
            )
            epoch_validation_domain = source_domains[
                _stable_seed(
                    model_seed, target, validation_domain, "source_epoch_validation"
                )
                % source_domains.size
            ]
            candidate = _fit_hierarchical_source_model(
                args,
                adapter_rank=adapter_rank,
                adapter_kind=adapter_kind,
                random_state=_stable_seed(
                    model_seed, target, "nested_adapter_screen", validation_domain
                ),
                window_store=window_store,
                source_indices=source_indices,
                subject_ids=subject_ids,
                trial_ids=trial_ids,
                source_labels=source_labels,
                pooled_mean=pooled_mean,
                pooled_std=pooled_std,
                moment_domains=moment_domains,
                subject_means=subject_means,
                subject_stds=subject_stds,
                source_epochs=int(getattr(args, "adapter_screen_source_epochs", 4)),
                source_refit_all=True,
                source_validation_domain=epoch_validation_domain,
            )
            pseudo_sequence = sequence_ids[pseudo_target_indices]
            pseudo_trials = trial_ids[pseudo_target_indices]
            trial_registry = pd.DataFrame(
                {"trial": pseudo_trials, "sequence": pseudo_sequence}
            ).drop_duplicates()
            minimum_trials = int(
                trial_registry.groupby("sequence")["trial"].nunique().min()
            )
            requested_screen_k = int(getattr(args, "adapter_screen_k", 10))
            screen_k = min(requested_screen_k, minimum_trials - 1)
            if screen_k < 1:
                raise ValueError(
                    f"Pseudo-target domain {validation_domain} has too few trials for screening"
                )
            pseudo_split = select_nested_trial_splits(
                pseudo_sequence,
                pseudo_trials,
                k_values=(screen_k,),
                seed=_stable_seed(model_seed, target, validation_domain, "adapter_screen_split"),
                context=f"outer={target}:pseudo={validation_domain}",
                split_mode="fixed_max_complement",
            )[screen_k]
            calibration_indices = pseudo_target_indices[pseudo_split.calibration_rows]
            evaluation_indices = pseudo_target_indices[pseudo_split.evaluation_rows]
            if np.intersect1d(calibration_indices, evaluation_indices).size:
                raise RuntimeError("Pseudo-target calibration and evaluation rows overlap")
            candidate.register_target_calibration_labels(
                calibration_indices,
                finger_labels=training_labels[calibration_indices],
                sequence_labels=sequence_ids[calibration_indices],
                order_labels=order_labels[calibration_indices],
                overlap_targets=overlap_targets[calibration_indices],
                press_ratios=press_ratios[calibration_indices],
            ).adapt_target_indices(
                calibration_indices,
                n_calibration_trials=int(pseudo_split.calibration_trials.size),
                mode="adapter_only",
            )
            probabilities = candidate.predict_proba_indices(evaluation_indices)
            pseudo_target_accuracy = float(
                np.mean(
                    np.argmax(probabilities, axis=1)
                    == raw_finger_labels[evaluation_indices]
                )
            )
            row = {
                "outer_test_subject": target,
                "outer_target_domain": target_index,
                "heldout_source_domain": int(validation_domain),
                "adapter_kind": adapter_kind,
                "adapter_rank": adapter_rank,
                "source_epoch_validation_domain": int(epoch_validation_domain),
                "pseudo_target_calibration_k_per_sequence": int(screen_k),
                "n_pseudo_target_calibration_trials": int(
                    pseudo_split.calibration_trials.size
                ),
                "n_pseudo_target_evaluation_trials": int(
                    pseudo_split.evaluation_trials.size
                ),
                "pseudo_target_finger_accuracy": pseudo_target_accuracy,
                "best_source_epoch": candidate.best_source_epoch_,
                "fit_seconds": float(time.monotonic() - started),
                "outer_target_rows_used": False,
                "outer_target_labels_used": False,
                "pseudo_target_labels_used_for_calibration": True,
                "pseudo_target_evaluation_labels_used_for_scoring_only": True,
                "pseudo_target_calibration_evaluation_disjoint": True,
            }
            _append_row(partial_path, row)
            completed.add(completion_key)
            _write_status(
                screen_status_path,
                state="fit_done",
                outer_test_subject=target,
                adapter_kind=adapter_kind,
                adapter_rank=int(adapter_rank),
                heldout_source_domain=int(validation_domain),
                n_completed=len(completed),
            )
            del candidate

    frame = pd.read_csv(partial_path).drop_duplicates(
        ["adapter_kind", "adapter_rank", "heldout_source_domain"], keep="last"
    )
    summary = (
        frame.groupby(["outer_test_subject", "adapter_kind", "adapter_rank"], as_index=False)
        .agg(
            mean_pseudo_target_finger_accuracy=("pseudo_target_finger_accuracy", "mean"),
            sem_pseudo_target_finger_accuracy=("pseudo_target_finger_accuracy", "sem"),
            n_source_validation_subjects=("heldout_source_domain", "nunique"),
            total_fit_seconds=("fit_seconds", "sum"),
        )
        .sort_values(
            ["mean_pseudo_target_finger_accuracy", "adapter_rank"],
            ascending=[False, True],
        )
    )
    best = summary.iloc[0]
    selected = {
        "adapter_kind": str(best["adapter_kind"]),
        "adapter_rank": int(best["adapter_rank"]),
        "selection": "nested_source_loso_calibrated_finger_accuracy",
        "outer_test_subject": target,
        "mean_pseudo_target_finger_accuracy": float(
            best["mean_pseudo_target_finger_accuracy"]
        ),
        "n_source_validation_subjects": int(best["n_source_validation_subjects"]),
        "outer_target_data_used": False,
        "outer_target_labels_used": False,
    }
    frame["selected"] = (
        (frame["adapter_kind"] == selected["adapter_kind"])
        & (frame["adapter_rank"] == selected["adapter_rank"])
    )
    frame.to_csv(screen_dir / "source_adapter_screen.csv", index=False)
    summary.to_csv(screen_dir / "source_adapter_screen_summary.csv", index=False)
    selected_path.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_status(
        screen_status_path,
        state="screen_complete",
        outer_test_subject=target,
        n_completed=int(frame.shape[0]),
        selected_adapter_kind=selected["adapter_kind"],
        selected_adapter_rank=selected["adapter_rank"],
    )
    return selected


def _aggregate_adapter_screens(output_dir: Path, targets: list[str]) -> dict[str, dict[str, Any]]:
    """Build top-level adapter-screen artifacts from target-specific nested screens."""

    frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    selected: dict[str, dict[str, Any]] = {}
    for target in targets:
        root = output_dir / "adapter_screens" / f"target={target}"
        frame_path = root / "source_adapter_screen.csv"
        summary_path = root / "source_adapter_screen_summary.csv"
        selected_path = root / "selected_adapter_config.json"
        if frame_path.exists():
            frames.append(pd.read_csv(frame_path))
        if summary_path.exists():
            summaries.append(pd.read_csv(summary_path))
        if selected_path.exists():
            selected[target] = json.loads(selected_path.read_text(encoding="utf-8"))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            output_dir / "source_adapter_screen.csv", index=False
        )
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            output_dir / "source_adapter_screen_summary.csv", index=False
        )
    if selected:
        (output_dir / "selected_adapter_configs.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return selected


def _composite_trial_ids(domain_ids: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [f"{domain}::{trial}" for domain, trial in zip(domain_ids.tolist(), trial_ids.tolist(), strict=True)],
        dtype=object,
    )


def validate_supplied_trial_row_order(composite_trial_ids: np.ndarray) -> dict[str, Any]:
    """Require each supplied trial to occupy one contiguous chronological row block."""

    trials = np.asarray(composite_trial_ids).reshape(-1)
    lengths: list[int] = []
    for trial in np.unique(trials):
        rows = np.flatnonzero(trials == trial)
        if rows.size == 0 or (rows.size > 1 and not np.all(np.diff(rows) == 1)):
            raise ValueError(
                f"Trial {trial!r} is not a contiguous cache row block; no explicit window index is available"
            )
        lengths.append(int(rows.size))
    return {
        "ordering_source": "row order within contiguous supplied-cache trial block",
        "n_trials": len(lengths),
        "minimum_windows_per_trial": min(lengths),
        "maximum_windows_per_trial": max(lengths),
        "all_trial_blocks_contiguous": True,
    }


def _context_template_labels(
    finger_labels: np.ndarray,
    order_labels: np.ndarray,
    domain_ids: np.ndarray,
    trial_ids: np.ndarray,
    fitting_indices: np.ndarray,
) -> np.ndarray:
    """Assign within-subject template 0/1 using fitting rows only."""

    output = np.zeros(finger_labels.size, dtype=np.int64)
    for domain in np.unique(domain_ids[fitting_indices]):
        domain_rows = fitting_indices[domain_ids[fitting_indices] == domain]
        observed: list[tuple[int, ...]] = []
        assignments: list[tuple[np.ndarray, tuple[int, ...]]] = []
        for trial in np.unique(trial_ids[domain_rows]):
            rows = domain_rows[trial_ids[domain_rows] == trial]
            template: list[int] = []
            for position in range(1, 6):
                values = finger_labels[rows][order_labels[rows] == position]
                values = values[values > 0]
                if values.size == 0:
                    template = []
                    break
                classes, counts = np.unique(values, return_counts=True)
                template.append(int(classes[np.argmax(counts)]))
            candidate = tuple(template)
            if len(candidate) == 5 and len(set(candidate)) == 5:
                if candidate not in observed:
                    observed.append(candidate)
                assignments.append((rows, candidate))
        if len(observed) > 2:
            raise ValueError(f"Subject {domain!r} has more than two observed finger templates")
        for rows, template in assignments:
            output[rows] = observed.index(template)
    return output


def split_calibration_inner_validation(
    calibration_indices: np.ndarray,
    sequence_ids: np.ndarray,
    trial_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Hold out one complete calibration trial per sequence for inner selection."""

    calibration = np.asarray(calibration_indices, dtype=np.int64).reshape(-1)
    if calibration.size == 0:
        raise ValueError("calibration_indices must not be empty")
    selected_validation_trials: list[Any] = []
    for sequence in np.unique(sequence_ids[calibration]):
        rows = calibration[sequence_ids[calibration] == sequence]
        trials = np.unique(trial_ids[rows])
        if trials.size < 2:
            return None
        rng = np.random.default_rng(_stable_seed(seed, "inner_calibration", sequence))
        selected_validation_trials.append(rng.permutation(trials)[0])
    validation_mask = np.zeros(calibration.size, dtype=bool)
    for position, row in enumerate(calibration.tolist()):
        sequence = sequence_ids[row]
        sequence_position = int(np.flatnonzero(np.unique(sequence_ids[calibration]) == sequence)[0])
        validation_mask[position] = trial_ids[row] == selected_validation_trials[sequence_position]
    training = calibration[~validation_mask]
    validation = calibration[validation_mask]
    if not training.size or not validation.size or np.intersect1d(training, validation).size:
        raise RuntimeError("Invalid calibration-only inner split")
    return training, validation


def select_window_adaptation_hyperparameters(
    source_model: TorchProgressiveTemporalWindowClassifier,
    *,
    calibration_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    sequence_ids: np.ndarray,
    trial_ids: np.ndarray,
    finger_labels: np.ndarray,
    raw_finger_labels: np.ndarray,
    order_labels: np.ndarray,
    overlap_targets: np.ndarray,
    press_ratios: np.ndarray,
    candidates: tuple[dict[str, float], ...],
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Select adaptation settings using calibration trials only."""

    calibration = np.asarray(calibration_indices, dtype=np.int64).reshape(-1)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64).reshape(-1)
    if np.intersect1d(calibration, evaluation).size:
        raise ValueError("Calibration-only tuning cannot overlap evaluation rows")
    if not candidates:
        raise ValueError("At least one adaptation candidate is required")
    split = split_calibration_inner_validation(
        calibration,
        sequence_ids,
        trial_ids,
        seed=seed,
    )
    if split is None:
        return dict(candidates[0]), [
            {
                **candidates[0],
                "candidate_index": 0,
                "selection_status": "predeclared_small_k",
                "n_inner_training_trials": int(np.unique(trial_ids[calibration]).size),
                "n_inner_validation_trials": 0,
                "inner_validation_accuracy": np.nan,
                "evaluation_rows_accessed": False,
            }
        ]
    inner_train, inner_validation = split
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        model = source_model.clone_source(
            random_state=_stable_seed(seed, "adaptation_candidate", candidate_index)
        )
        model.adaptation_learning_rate = float(candidate["learning_rate"])
        model.full_finetune_learning_rate = float(candidate["learning_rate"]) / 10.0
        model.source_replay_weight = float(candidate["source_replay_weight"])
        multiplier = float(candidate["step_multiplier"])
        model.adapter_steps = max(1, int(round(source_model.adapter_steps * multiplier)))
        model.last_block_steps = max(0, int(round(source_model.last_block_steps * multiplier)))
        model.full_finetune_steps = max(0, int(round(source_model.full_finetune_steps * multiplier)))
        model.register_target_calibration_labels(
            inner_train,
            finger_labels=finger_labels[inner_train],
            sequence_labels=sequence_ids[inner_train],
            order_labels=order_labels[inner_train],
            overlap_targets=overlap_targets[inner_train],
            press_ratios=press_ratios[inner_train],
        ).adapt_target_indices(
            inner_train,
            n_calibration_trials=int(np.unique(trial_ids[inner_train]).size),
            mode="progressive_full",
        )
        probabilities = model.predict_proba_indices(inner_validation)
        accuracy = float(
            np.mean(np.argmax(probabilities, axis=1) == raw_finger_labels[inner_validation])
        )
        rows.append(
            {
                **candidate,
                "candidate_index": candidate_index,
                "selection_status": "calibration_only_inner_validation",
                "n_inner_training_trials": int(np.unique(trial_ids[inner_train]).size),
                "n_inner_validation_trials": int(np.unique(trial_ids[inner_validation]).size),
                "inner_validation_accuracy": accuracy,
                "evaluation_rows_accessed": False,
            }
        )
    best = max(rows, key=lambda row: (row["inner_validation_accuracy"], -row["candidate_index"]))
    return {
        "learning_rate": float(best["learning_rate"]),
        "source_replay_weight": float(best["source_replay_weight"]),
        "step_multiplier": float(best["step_multiplier"]),
    }, rows


def _output_auxiliary(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = {
        "finger": "finger_logits",
        "press": "press_logits",
        "conditional_finger": "conditional_finger_logits",
        "sequence": "sequence_logits",
        "order": "order_logits",
        "overlap_logit": "overlap_logits",
        "overlap": "overlap_prediction",
        "template": "template_logits",
        "press_probabilities": "press_probabilities",
        "conditional_finger_probabilities": "conditional_finger_probabilities",
        "order_probabilities": "order_probabilities",
        "template_probabilities": "template_probabilities",
    }
    return {destination: outputs[source] for source, destination in names.items() if source in outputs}


def apply_probability_temperature(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply scalar temperature scaling to already normalized probabilities."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("probabilities must have shape [rows, 6]")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    return result


def fit_probability_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit one temperature from labeled calibration rows only."""

    from scipy.optimize import minimize_scalar

    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.shape != (targets.size, 6) or targets.size == 0:
        raise ValueError("probabilities and labels must be aligned six-class calibration rows")
    if np.any((targets < 0) | (targets > 5)):
        raise ValueError("labels must use classes 0..5")

    def objective(log_temperature: float) -> float:
        calibrated = apply_probability_temperature(values, np.exp(log_temperature))
        return float(-np.log(np.clip(calibrated[np.arange(targets.size), targets], 1e-12, 1.0)).mean())

    fitted = minimize_scalar(objective, bounds=(-2.3, 2.3), method="bounded")
    if not fitted.success or not np.isfinite(fitted.x):
        return 1.0
    return float(np.exp(fitted.x))


def _bundle_path(
    output_dir: Path,
    *,
    family: str,
    target: str,
    split_seed: int,
    k: int,
    model_seed: int,
) -> Path:
    return (
        output_dir
        / "predictions"
        / family
        / f"target={target}"
        / f"split_seed={split_seed}"
        / f"k={k}"
        / f"model_seed={model_seed}.npz"
    )


def _forced_trial_predictions(
    probabilities: np.ndarray,
    evaluation_global_rows: np.ndarray,
    trial_ids_global: np.ndarray,
    templates: tuple[tuple[int, ...], ...],
    stay_probabilities: np.ndarray,
    *,
    causal: bool,
    auxiliary: dict[str, Any] | None = None,
) -> np.ndarray:
    predicted = np.empty(evaluation_global_rows.size, dtype=np.int64)
    local_trials = trial_ids_global[evaluation_global_rows]
    auxiliary = auxiliary or {}
    order_probabilities = auxiliary.get("aux_order_probabilities")
    overlap_probabilities = auxiliary.get("aux_overlap_prediction")
    template_probabilities = auxiliary.get("aux_template_probabilities")
    for trial in np.unique(local_trials):
        local = np.flatnonzero(local_trials == trial)
        decode_arguments = {
            "stay_probabilities": stay_probabilities,
            "order_probabilities": (
                None if order_probabilities is None else order_probabilities[local]
            ),
            "overlap_probabilities": (
                None if overlap_probabilities is None else overlap_probabilities[local]
            ),
            "template_probabilities": (
                None if template_probabilities is None else template_probabilities[local]
            ),
        }
        if causal:
            predicted[local] = causal_trial_decode(
                probabilities[local], templates, **decode_arguments
            )
        else:
            predicted[local] = structured_trial_decode(
                probabilities[local], templates, **decode_arguments
            ).labels
    return predicted


def _metric_and_append(
    partial_path: Path,
    *,
    method: str,
    target: str,
    target_index: int,
    split_seed: int,
    model_seed: int,
    split,
    probabilities: np.ndarray,
    raw_labels_target: np.ndarray,
    training_labels_target: np.ndarray,
    target_trials: np.ndarray,
    n_source_windows: int,
    n_source_subjects: int,
    adaptation_stages: str,
    predicted_labels: np.ndarray | None = None,
    decoding_mode: str = "independent_windows",
) -> dict[str, Any]:
    row = _metric_row(
        method=method,
        target=target,
        target_index=target_index,
        seed=split_seed,
        split=split,
        probabilities=probabilities,
        raw_labels=raw_labels_target,
        training_labels=training_labels_target,
        target_trial_ids=target_trials,
        n_source_windows=n_source_windows,
        n_source_subjects=n_source_subjects,
        adaptation_stages=adaptation_stages,
        predicted_labels=predicted_labels,
    )
    row.update(
        {
            "split_seed": int(split_seed),
            "model_seed": int(model_seed),
            "split_mode": "fixed_max_complement",
            "evaluation_rows_fixed_across_k": True,
            "decoding_mode": decoding_mode,
            "offline_uses_future_windows": decoding_mode.startswith("offline_"),
            "evaluation_labels_available_to_fitting": False,
            "feature_kind": "raw_500ms_hierarchical_temporal",
        }
    )
    _append_row(partial_path, row)
    return row


def _paired_push_statistics(subject_rows: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import t as student_t
    from scipy.stats import ttest_1samp

    reference_methods = (
        "hierarchical_source_only",
        "hierarchical_tcn_single",
    )
    output: list[dict[str, Any]] = []
    for k in sorted(subject_rows["k_trials_per_sequence"].unique()):
        frame = subject_rows[subject_rows["k_trials_per_sequence"] == k]
        pivot = frame.pivot(index="target", columns="method", values="accuracy_raw_labels")
        for reference_method in reference_methods:
            if reference_method not in pivot:
                continue
            for method in pivot.columns:
                if method == reference_method:
                    continue
                paired = pivot[[method, reference_method]].dropna()
                if paired.empty:
                    continue
                differences = (
                    paired[method] - paired[reference_method]
                ).to_numpy(dtype=float)
                n_subjects = int(differences.size)
                sem = (
                    float(np.std(differences, ddof=1) / np.sqrt(n_subjects))
                    if n_subjects > 1
                    else np.nan
                )
                critical = (
                    float(student_t.ppf(0.975, n_subjects - 1))
                    if n_subjects > 1
                    else np.nan
                )
                output.append(
                    {
                        "method": method,
                        "reference_method": reference_method,
                        "k_trials_per_sequence": int(k),
                        "n_subjects": n_subjects,
                        "mean_paired_delta_accuracy": float(np.mean(differences)),
                        "sem_paired_delta_accuracy": sem,
                        "ci95_low_accuracy": float(
                            np.mean(differences) - critical * sem
                        ),
                        "ci95_high_accuracy": float(
                            np.mean(differences) + critical * sem
                        ),
                        "paired_t_p": float(ttest_1samp(differences, 0.0).pvalue)
                        if n_subjects > 1
                        else np.nan,
                    }
                )
    return pd.DataFrame(output)


def _common_curve_cohort_summaries(
    split_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Summarize every plotted method/k cell on one shared target cohort."""

    target_sets = [
        set(frame["target"].astype(str))
        for _, frame in split_rows.groupby(
            ["method", "k_trials_per_sequence"], sort=False
        )
    ]
    if not target_sets:
        raise ValueError("Cannot build a common curve cohort from empty results")
    common_targets = tuple(sorted(set.intersection(*target_sets)))
    if not common_targets:
        raise ValueError("No target participant is present in every method/k curve cell")
    common_rows = split_rows[
        split_rows["target"].astype(str).isin(common_targets)
    ].copy()
    subject, summary, julia = summarize_results(common_rows)
    if set(summary["n_subjects"].astype(int)) != {len(common_targets)}:
        raise RuntimeError("Common-cohort summary unexpectedly changed participant count")
    return subject, summary, julia, common_targets


def _plot_push_summary(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "hierarchical_source_only": "Source-only 5-model ensemble",
        "hierarchical_tcn_single": "Hierarchical TCN, single",
        "hierarchical_tcn_ensemble": "Hierarchical TCN, 5-model ensemble",
        "trial_transformer_causal_ensemble": "Causal trial Transformer ensemble",
        "trial_transformer_ensemble": "Offline trial Transformer ensemble",
        "trial_transformer_ensemble_structured": "Offline trial Transformer + structure",
        "hybrid_ensemble": "TCN + Transformer ensemble",
        "hybrid_ensemble_structured": "TCN + Transformer + structure (offline)",
    }
    colors = {
        "hierarchical_source_only": "#202020",
        "hierarchical_tcn_single": "#4c78a8",
        "hierarchical_tcn_ensemble": "#2a6fbb",
        "trial_transformer_causal_ensemble": "#59a14f",
        "trial_transformer_ensemble": "#8f63b8",
        "trial_transformer_ensemble_structured": "#9c2f8f",
        "hybrid_ensemble": "#e17c05",
        "hybrid_ensemble_structured": "#c43c39",
    }
    selected = summary[summary["method"].isin(labels)].copy()
    fig, axis = plt.subplots(figsize=(11.2, 5.4))
    axis.axhspan(
        62.5,
        64.5,
        color="#6b7c9e",
        alpha=0.13,
        label="Julia reported range (not fold-matched)",
    )
    if "mean_majority_class_accuracy" in selected:
        majority = (
            selected.sort_values(["k_trials_per_sequence", "method"])
            .drop_duplicates("k_trials_per_sequence")
            .sort_values("k_trials_per_sequence")
        )
        axis.plot(
            majority["k_trials_per_sequence"],
            100.0 * majority["mean_majority_class_accuracy"],
            color="#777777",
            linestyle="-.",
            linewidth=1.2,
            label="Empirical majority-class baseline",
        )
    for method, frame in selected.groupby("method", sort=False):
        frame = frame.sort_values("k_trials_per_sequence")
        source_only = method == "hierarchical_source_only"
        axis.errorbar(
            frame["k_trials_per_sequence"],
            100.0 * frame["mean_accuracy_raw_labels"],
            yerr=100.0 * frame["sem_accuracy_raw_labels"],
            color=colors[method],
            marker="D" if source_only else "o",
            linestyle="--" if source_only else "-",
            linewidth=1.8,
            capsize=3,
            label=labels[method],
        )
    axis.axhline(100.0 / 6.0, color="#777777", linestyle=":", linewidth=1.2, label="Uniform chance")
    axis.set_xscale("log")
    axis.set_xticks(sorted(selected["k_trials_per_sequence"].unique()))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("Labeled target trials per sequence (k)")
    axis.set_ylabel("Six-class window accuracy (%)")
    cohort_size = int(summary["n_subjects"].iloc[0])
    axis.set_title(
        "Katja sliding-window accuracy push\n"
        f"Common {cohort_size}-participant cohort; fixed evaluation trials\n"
        "Error bars: subject SEM",
        fontsize=12,
    )
    axis.grid(alpha=0.2)
    axis.legend(
        frameon=False,
        fontsize=8,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.72, 1.0))
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _plot_direct_julia_comparison(summary: pd.DataFrame, output: Path) -> None:
    """Plot only methods whose inference input is one supplied MEG window."""

    import matplotlib.pyplot as plt

    labels = {
        "hierarchical_source_only": "Source-only 5-model ensemble",
        "hierarchical_tcn_single": "Hierarchical TCN, single-model mean",
        "hierarchical_tcn_ensemble": "Hierarchical TCN, 5-model ensemble",
    }
    colors = {
        "hierarchical_source_only": "#202020",
        "hierarchical_tcn_single": "#4c78a8",
        "hierarchical_tcn_ensemble": "#2a6fbb",
    }
    selected = summary[summary["method"].isin(DIRECT_JULIA_COMPARISON_METHODS)].copy()
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.axhspan(
        62.5,
        64.5,
        color="#6b7c9e",
        alpha=0.13,
        label="Julia reported range (not fold-matched)",
    )
    if "mean_majority_class_accuracy" in selected:
        majority = (
            selected.sort_values(["k_trials_per_sequence", "method"])
            .drop_duplicates("k_trials_per_sequence")
            .sort_values("k_trials_per_sequence")
        )
        axis.plot(
            majority["k_trials_per_sequence"],
            100.0 * majority["mean_majority_class_accuracy"],
            color="#777777",
            linestyle="-.",
            linewidth=1.2,
            label="Empirical majority-class baseline",
        )
    for method, frame in selected.groupby("method", sort=False):
        frame = frame.sort_values("k_trials_per_sequence")
        source_only = method == "hierarchical_source_only"
        axis.errorbar(
            frame["k_trials_per_sequence"],
            100.0 * frame["mean_accuracy_raw_labels"],
            yerr=100.0 * frame["sem_accuracy_raw_labels"],
            color=colors[method],
            marker="D" if source_only else "o",
            linestyle="--" if source_only else "-",
            linewidth=1.8,
            capsize=3,
            label=labels[method],
        )
    axis.axhline(
        100.0 / 6.0,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
        label="Uniform chance",
    )
    axis.set_xscale("log")
    axis.set_xticks(sorted(selected["k_trials_per_sequence"].unique()))
    axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axis.set_xlabel("Labeled target trials per sequence (k)")
    axis.set_ylabel("Six-class window accuracy (%)")
    cohort_size = int(summary["n_subjects"].iloc[0])
    axis.set_title(
        "Katja direct independent-window comparison\n"
        f"common {cohort_size}-participant cohort; fixed evaluation trials; "
        "error bars: subject SEM"
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _method_comparison_scope(methods: Sequence[str]) -> pd.DataFrame:
    """Describe which inference inputs make each method directly comparable."""

    rows: list[dict[str, Any]] = []
    for method in sorted(set(str(value) for value in methods)):
        direct = method in DIRECT_JULIA_COMPARISON_METHODS
        structured = method.endswith("_structured") or method.endswith("_causal")
        context = "transformer" in method or method.startswith("hybrid_")
        if direct:
            scope = "direct_independent_window_comparison"
            inference_inputs = "one Julia-supplied MEG window"
            caveat = "same supplied window endpoint; unpublished architectures still differ"
        elif structured:
            scope = "supplementary_task_structure"
            inference_inputs = (
                "Julia-supplied MEG windows, trial grouping, chronological order, and "
                "calibration-learned task structure"
            )
            caveat = "Julia's unpublished inference code does not establish use of these priors"
        elif context:
            scope = "supplementary_trial_context"
            inference_inputs = "Julia-supplied MEG windows, trial grouping, and chronological order"
            caveat = "Julia's unpublished inference code does not establish use of trial context"
        else:
            scope = "supplementary"
            inference_inputs = "Julia-supplied cache inputs"
            caveat = "not designated as a primary direct-comparison method"
        rows.append(
            {
                "method": method,
                "comparison_scope": scope,
                "direct_julia_comparison": bool(direct),
                "inference_inputs": inference_inputs,
                "comparison_caveat": caveat,
            }
        )
    return pd.DataFrame(rows)


def _write_push_markdown(
    summary: pd.DataFrame,
    julia: pd.DataFrame,
    paired: pd.DataFrame,
    output: Path,
    *,
    curve_targets: Sequence[str],
) -> None:
    merged = summary.merge(
        julia,
        on=["method", "k_trials_per_sequence"],
        suffixes=("", "_fold"),
    )
    key_methods = [
        "hierarchical_source_only",
        "hierarchical_tcn_single",
        "hierarchical_tcn_ensemble",
        "trial_transformer_causal_ensemble",
        "trial_transformer_ensemble",
        "trial_transformer_ensemble_structured",
        "hybrid_ensemble",
        "hybrid_ensemble_structured",
    ]
    headline_rows = summary[
        summary["method"].isin(
            [
                "hierarchical_tcn_ensemble",
                "trial_transformer_ensemble",
                "trial_transformer_ensemble_structured",
            ]
        )
        & summary["k_trials_per_sequence"].isin([10, 20])
    ].set_index(["method", "k_trials_per_sequence"])

    def headline(method: str, k: int, label: str) -> str | None:
        key = (method, k)
        if key not in headline_rows.index:
            return None
        row = headline_rows.loc[key]
        return (
            f"- {label}, k={k}: {100 * row['mean_accuracy_raw_labels']:.2f}% "
            f"+/- {100 * row['sem_accuracy_raw_labels']:.2f} pp subject SEM "
            f"({int(row['n_subjects'])} participants)."
        )

    headline_lines = [
        headline("hierarchical_tcn_ensemble", 10, "Direct independent-window TCN ensemble"),
        headline("hierarchical_tcn_ensemble", 20, "Direct independent-window TCN ensemble"),
        headline("trial_transformer_ensemble", 20, "Supplementary offline trial-context ensemble"),
        headline(
            "trial_transformer_ensemble_structured",
            20,
            "Supplementary offline trial-context plus structure",
        ),
    ]
    headline_lines = [line for line in headline_lines if line is not None]
    lines = [
        "# Katja sliding-window accuracy push",
        "",
        "All target adaptation and model selection use labeled calibration trials only. "
        "The maximum calibration pool is reserved first, and every k is scored on the "
        "same remaining evaluation trials. Split seeds and model seeds are independent.",
        "",
        "Acceptance was gated on exact reproduction of the validated prior benchmark: "
        "53.36% at k=10 over ten targets and 56.03% at k=20 over the feasible nine.",
        "",
        "The bidirectional Transformer and Viterbi rows are offline structured results and "
        "may use future windows from the same evaluation trial. Independent-window and "
        "causal-prefix rows remain separate.",
        "",
        "At evaluation time, every method receives only Julia's cached MEG windows, target "
        "participant identity, trial membership, and chronological window order. Ground-truth "
        "target finger, sequence, press-order, overlap, press-ratio, and event-timing "
        "annotations are unavailable to classification. Structured decoding uses only "
        "calibration-observed templates, source-plus-calibration duration priors, and predicted "
        "auxiliary heads.",
        "",
        "This is an independent NeuRepTrace method comparison. Julia supplied the cache, "
        "schema, and documented tau=0.2 endpoint relabel rule; no collaborator model "
        "architecture, weights, training/adaptation code, or split function was supplied "
        "or copied. The hierarchical TCN, adapters, trial-context model, ensemble, and "
        "structured decoder were implemented independently in NeuRepTrace.",
        "",
        "The primary direct comparison to Julia is restricted to source-only and hierarchical "
        "TCN independent-window rows. Trial-context and structured rows are supplementary: "
        "they use trial grouping and chronological cache order, whose use in Julia's "
        "unpublished inference implementation is not established.",
        "",
        "Because the raw six-class endpoint is imbalanced, the empirical majority-class "
        "accuracy is reported alongside the 16.67% uniform-random reference.",
        "",
        f"Both figures use the same {len(curve_targets)}-participant cohort at every k "
        f"({', '.join(curve_targets)}). The headline table remains available-case so k=10 "
        "retains all ten participants, while k=20 uses the feasible common nine.",
        "",
        "## Headline results",
        "",
        *headline_lines,
        "",
        "Only the independent-window TCN rows are primary direct comparisons with Julia's "
        "reported 62.5-64.5% range. Trial-context and structured rows are supplementary; "
        "crossing 64.5% there is descriptive rather than a formal matched win.",
        "",
        "| Method | k | Subjects | Mean | Subject SEM | Fold SD | Majority | Press-only | Rest recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in merged[merged["method"].isin(key_methods)].sort_values(
        ["method", "k_trials_per_sequence"]
    ).itertuples(index=False):
        lines.append(
            f"| {row.method} | {int(row.k_trials_per_sequence)} | {int(row.n_subjects)} | "
            f"{100 * row.mean_accuracy_raw_labels:.2f}% | "
            f"{100 * row.sem_accuracy_raw_labels:.2f} pp | "
            f"{100 * row.sd_accuracy_raw_labels:.2f} pp | "
            f"{100 * row.mean_majority_class_accuracy:.2f}% | "
            f"{100 * row.mean_press_only_finger_accuracy:.2f}% | "
            f"{100 * row.mean_rest_recall:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Julia's 62.5-64.5% range is descriptive context, not a formal paired benchmark, "
            "because her unpublished model code and exact fold predictions are unavailable.",
            "",
            "Paired differences after averaging model/split repetitions within each target "
            "participant are available in `paired_method_improvements.csv`.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def _write_push_reports(rows: pd.DataFrame, output_dir: Path) -> None:
    key = ["target", "split_seed", "k_trials_per_sequence", "method", "model_seed"]
    if rows.duplicated(key).any():
        raise RuntimeError("Accuracy-push results contain duplicate split rows")
    metric_columns = [
        "accuracy_raw_labels",
        "balanced_accuracy_raw_labels",
        "accuracy_tau_labels",
        "press_only_finger_accuracy",
        "rest_recall",
        "press_detection_accuracy",
        "trial_macro_accuracy_raw_labels",
        "log_loss_raw_labels",
        "majority_class_accuracy",
    ]
    metric_columns = [column for column in metric_columns if column in rows]
    split_rows = (
        rows.groupby(
            ["method", "k_trials_per_sequence", "target", "split_seed"],
            as_index=False,
        )[metric_columns]
        .mean()
        .rename(columns={"split_seed": "seed"})
    )
    subject, summary, julia = summarize_results(split_rows)
    common_subject, common_summary, common_julia, common_targets = (
        _common_curve_cohort_summaries(split_rows)
    )
    rows.to_csv(output_dir / "fold_results.csv", index=False)
    subject.to_csv(output_dir / "subject_seed_averages.csv", index=False)
    summary.to_csv(output_dir / "summary_subject_sem.csv", index=False)
    julia.to_csv(output_dir / "summary_julia_fold_sd.csv", index=False)
    common_subject.to_csv(
        output_dir / "subject_seed_averages_common.csv", index=False
    )
    common_summary.to_csv(output_dir / "summary_common_subject_sem.csv", index=False)
    common_julia.to_csv(output_dir / "summary_common_julia_fold_sd.csv", index=False)
    paired = _paired_push_statistics(subject)
    paired.to_csv(output_dir / "paired_method_improvements.csv", index=False)
    scope = _method_comparison_scope(rows["method"].astype(str))
    scope.to_csv(output_dir / "method_comparison_scope.csv", index=False)
    summary[
        summary["k_trials_per_sequence"].isin([10, 20])
    ].merge(scope, on="method", how="left").to_csv(
        output_dir / "headline_results.csv", index=False
    )
    _plot_push_summary(
        common_summary, output_dir / "katja_window_accuracy_push.png"
    )
    _plot_direct_julia_comparison(
        common_summary,
        output_dir / "katja_window_accuracy_push_direct_comparison.png",
    )
    _write_push_markdown(
        summary,
        julia,
        paired,
        output_dir / "report.md",
        curve_targets=common_targets,
    )
    validation = {
        "all_required_checks_pass": bool(
            rows["calibration_evaluation_disjoint"].astype(bool).all()
            and not rows.duplicated(key).any()
            and rows["evaluation_rows_fixed_across_k"].astype(bool).all()
            and (~rows["evaluation_labels_available_to_fitting"].astype(bool)).all()
        ),
        "n_rows": int(rows.shape[0]),
        "n_methods": int(rows["method"].nunique()),
        "n_targets": int(rows["target"].nunique()),
        "checks": {
            "unique_model_split_rows": not rows.duplicated(key).any(),
            "calibration_evaluation_disjoint": bool(rows["calibration_evaluation_disjoint"].astype(bool).all()),
            "fixed_evaluation_rows_across_k": bool(rows["evaluation_rows_fixed_across_k"].astype(bool).all()),
            "evaluation_labels_not_used_for_fitting": bool((~rows["evaluation_labels_available_to_fitting"].astype(bool)).all()),
        },
        "created_at": _utc_timestamp(),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not validation["all_required_checks_pass"]:
        raise RuntimeError("Accuracy-push validation failed")


def _expected_full_design_keys(
    feasibility_by_target: dict[str, dict[str, Any]],
) -> set[tuple[str, int, int, str, int]]:
    """Build the exact result identity set for the predeclared full experiment."""

    expected: set[tuple[str, int, int, str, int]] = set()
    for target in JULIA_SUBJECTS:
        target_feasibility = feasibility_by_target.get(target)
        if target_feasibility is None:
            continue
        feasible = tuple(int(value) for value in target_feasibility["feasible_k_values"])
        for split_seed in DEFAULT_SEEDS:
            for k in feasible:
                expected.update(
                    (target, int(split_seed), int(k), method, -1)
                    for method in ENSEMBLE_METHODS
                )
                expected.update(
                    (target, int(split_seed), int(k), method, int(model_seed))
                    for method in MEMBER_METHODS
                    for model_seed in DEFAULT_SEEDS
                )
    return expected


def _validate_full_prediction_bundles(
    shard_by_target: dict[str, Path],
    feasibility_by_target: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Inspect persisted bundles and prove fixed, row-identical evaluation sets."""

    row_hashes: dict[str, str] = {}
    n_bundles = 0
    n_source_probability_comparisons = 0
    for target in JULIA_SUBJECTS:
        shard = shard_by_target[target]
        feasible = tuple(
            int(value) for value in feasibility_by_target[target]["feasible_k_values"]
        )
        for split_seed in DEFAULT_SEEDS:
            reference_rows: np.ndarray | None = None
            for model_seed in DEFAULT_SEEDS:
                source_reference: np.ndarray | None = None
                for k in feasible:
                    for family in PREDICTION_FAMILIES:
                        path = (
                            shard
                            / "predictions"
                            / family
                            / f"target={target}"
                            / f"split_seed={split_seed}"
                            / f"k={k}"
                            / f"model_seed={model_seed}.npz"
                        )
                        if not path.exists():
                            raise RuntimeError(f"Missing full-design prediction bundle: {path}")
                        with np.load(path, allow_pickle=False) as bundle:
                            rows = np.asarray(bundle["row_indices"], dtype=np.int64)
                            if rows.ndim != 1 or np.unique(rows).size != rows.size:
                                raise RuntimeError(
                                    f"Prediction bundle has invalid row identities: {path}"
                                )
                            if int(bundle["split_seed"]) != int(split_seed):
                                raise RuntimeError(f"Prediction bundle split mismatch: {path}")
                            if int(bundle["model_seed"]) != int(model_seed):
                                raise RuntimeError(f"Prediction bundle model mismatch: {path}")
                            if reference_rows is None:
                                reference_rows = rows.copy()
                            elif not np.array_equal(reference_rows, rows):
                                raise RuntimeError(
                                    "Evaluation rows differ across k, families, or model seeds: "
                                    f"target={target} split_seed={split_seed} bundle={path}"
                                )
                            if family == "hierarchical_source_only":
                                probabilities = np.asarray(
                                    bundle["probabilities"], dtype=np.float32
                                )
                                if probabilities.shape != (rows.size, 6):
                                    raise RuntimeError(
                                        f"Source-only bundle has invalid probabilities: {path}"
                                    )
                                if source_reference is None:
                                    source_reference = probabilities.copy()
                                elif not np.array_equal(source_reference, probabilities):
                                    raise RuntimeError(
                                        "Source-only probabilities changed across k despite the "
                                        f"fixed evaluation set: target={target} "
                                        f"split_seed={split_seed} model_seed={model_seed}"
                                    )
                                n_source_probability_comparisons += 1
                        n_bundles += 1
            assert reference_rows is not None
            row_hashes[f"{target}/split_seed={split_seed}"] = hashlib.sha256(
                reference_rows.astype("<i8", copy=False).tobytes()
            ).hexdigest()
    return {
        "all_prediction_bundles_present": True,
        "prediction_rows_identical_across_k_families_and_model_seeds": True,
        "source_only_probabilities_identical_across_k": True,
        "n_prediction_bundles_checked": int(n_bundles),
        "n_source_probability_bundles_checked": int(
            n_source_probability_comparisons
        ),
        "evaluation_row_sha256": row_hashes,
    }


def _validate_full_design(
    rows: pd.DataFrame,
    *,
    feasibility_by_target: dict[str, dict[str, Any]],
    shard_by_target: dict[str, Path],
) -> dict[str, Any]:
    """Require the exact ten-target, five-by-five-seed accuracy-push design."""

    expected_targets = set(JULIA_SUBJECTS)
    observed_targets = set(rows["target"].astype(str))
    expected_feasibility = {
        target: [
            int(k)
            for k in DEFAULT_K_VALUES
            if target != "s06" or int(k) <= 10
        ]
        for target in JULIA_SUBJECTS
    }
    observed_feasibility = {
        target: [int(value) for value in feasibility_by_target[target]["feasible_k_values"]]
        for target in sorted(feasibility_by_target)
    }
    key_columns = [
        "target",
        "split_seed",
        "k_trials_per_sequence",
        "method",
        "model_seed",
    ]
    observed_keys = set(
        zip(
            rows["target"].astype(str),
            rows["split_seed"].astype(int),
            rows["k_trials_per_sequence"].astype(int),
            rows["method"].astype(str),
            rows["model_seed"].astype(int),
            strict=True,
        )
    )
    expected_keys = _expected_full_design_keys(feasibility_by_target)
    checks = {
        "exact_ten_target_cohort": observed_targets == expected_targets,
        "exact_target_shard_map": set(shard_by_target) == expected_targets,
        "exact_feasible_k_pattern": observed_feasibility == expected_feasibility,
        "exact_twelve_methods": set(rows["method"].astype(str)) == set(PUSH_METHODS),
        "exact_result_identity_set": observed_keys == expected_keys,
        "exact_result_row_count_6960": int(rows.shape[0]) == 6960,
        "no_duplicate_result_identities": not rows.duplicated(key_columns).any(),
        "k10_has_all_ten_targets": set(
            rows.loc[rows["k_trials_per_sequence"].astype(int) == 10, "target"].astype(str)
        )
        == expected_targets,
        "k20_has_common_nine_targets": set(
            rows.loc[rows["k_trials_per_sequence"].astype(int) == 20, "target"].astype(str)
        )
        == expected_targets - {"s06"},
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"Full accuracy-push design validation failed: {failed}")
    bundle_validation = _validate_full_prediction_bundles(
        shard_by_target, feasibility_by_target
    )
    return {
        "all_required_checks_pass": True,
        "checks": checks,
        "n_expected_result_rows": len(expected_keys),
        "n_observed_result_rows": int(rows.shape[0]),
        "expected_feasible_k_values": expected_feasibility,
        **bundle_validation,
    }


def aggregate_accuracy_push_shards(
    shard_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    baseline_results: str | Path,
    require_full_design: bool = False,
) -> Path:
    """Combine validated, target-disjoint shards and rebuild all reports."""

    if not shard_dirs:
        raise ValueError("At least one shard directory is required")
    baseline = validate_baseline_reproduction(baseline_results)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    member_frames: list[pd.DataFrame] = []
    screen_frames: list[pd.DataFrame] = []
    screen_summaries: list[pd.DataFrame] = []
    adaptation_frames: list[pd.DataFrame] = []
    selected_adapters: dict[str, dict[str, Any]] = {}
    feasibility_by_target: dict[str, dict[str, Any]] = {}
    shard_by_target: dict[str, Path] = {}
    manifests: list[dict[str, Any]] = []
    observed_targets: set[str] = set()
    for raw_path in shard_dirs:
        shard = Path(raw_path).expanduser().resolve()
        validation_path = shard / "validation.json"
        rows_path = shard / "fold_results.csv"
        if not validation_path.exists() or not rows_path.exists():
            raise FileNotFoundError(
                f"Shard {shard} requires validation.json and fold_results.csv"
            )
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("all_required_checks_pass") is not True:
            raise RuntimeError(f"Shard validation failed: {shard}")
        if validation.get("checks", {}).get("all_expected_results_present") is not True:
            raise RuntimeError(f"Shard lacks exact expected-coverage validation: {shard}")
        frame = pd.read_csv(rows_path)
        targets = set(frame["target"].astype(str).unique().tolist())
        overlap = observed_targets & targets
        if overlap:
            raise RuntimeError(
                f"Shard targets overlap and could hide conflicting results: {sorted(overlap)}"
            )
        observed_targets.update(targets)
        for target in targets:
            shard_by_target[target] = shard
        frames.append(frame)
        member_path = shard / "member_results.csv"
        if member_path.exists():
            member_frames.append(pd.read_csv(member_path))
        for source, destination in (
            ("source_adapter_screen.csv", screen_frames),
            ("source_adapter_screen_summary.csv", screen_summaries),
            ("adaptation_selection.partial.csv", adaptation_frames),
        ):
            path = shard / source
            if path.exists():
                destination.append(pd.read_csv(path))
        selected_path = shard / "selected_adapter_configs.json"
        if selected_path.exists():
            for target, configuration in json.loads(
                selected_path.read_text(encoding="utf-8")
            ).items():
                if target in selected_adapters:
                    raise RuntimeError(f"Duplicate selected adapter for target {target}")
                selected_adapters[target] = configuration
        feasibility_path = shard / "feasibility.json"
        if not feasibility_path.exists():
            raise FileNotFoundError(f"Shard lacks feasibility.json: {shard}")
        for target, target_feasibility in json.loads(
            feasibility_path.read_text(encoding="utf-8")
        ).items():
            if target not in targets:
                raise RuntimeError(
                    f"Shard feasibility target {target} is absent from fold results: {shard}"
                )
            if target in feasibility_by_target:
                raise RuntimeError(f"Duplicate feasibility metadata for target {target}")
            feasibility_by_target[target] = target_feasibility
        manifests.append(
            {
                "shard": str(shard),
                "targets": sorted(targets),
                "n_rows": int(frame.shape[0]),
                "validation": str(validation_path),
                "prediction_root": str(shard / "predictions"),
            }
        )

    rows = pd.concat(frames, ignore_index=True)
    unique_key = [
        "target",
        "split_seed",
        "k_trials_per_sequence",
        "method",
        "model_seed",
    ]
    if rows.duplicated(unique_key).any():
        raise RuntimeError("Combined shards contain duplicate result identities")
    full_design_validation: dict[str, Any] | None = None
    if require_full_design:
        full_design_validation = _validate_full_design(
            rows,
            feasibility_by_target=feasibility_by_target,
            shard_by_target=shard_by_target,
        )
        (output / "full_design_validation.json").write_text(
            json.dumps(full_design_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _write_push_reports(rows, output)
    if member_frames:
        members = pd.concat(member_frames, ignore_index=True)
        if members.duplicated(unique_key).any():
            raise RuntimeError("Combined member shards contain duplicate identities")
        members.to_csv(output / "member_results.csv", index=False)
    if screen_frames:
        pd.concat(screen_frames, ignore_index=True).to_csv(
            output / "source_adapter_screen.csv", index=False
        )
    if screen_summaries:
        pd.concat(screen_summaries, ignore_index=True).to_csv(
            output / "source_adapter_screen_summary.csv", index=False
        )
    if adaptation_frames:
        pd.concat(adaptation_frames, ignore_index=True).to_csv(
            output / "adaptation_selection.csv", index=False
        )
    if selected_adapters:
        (output / "selected_adapter_configs.json").write_text(
            json.dumps(selected_adapters, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output / "feasibility.json").write_text(
        json.dumps(feasibility_by_target, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(manifests).to_csv(output / "shards.csv", index=False)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "aggregation": "validated_target_disjoint_shards",
                "baseline_reproduction": baseline,
                "shards": manifests,
                "targets": sorted(observed_targets),
                "feasibility_by_target": feasibility_by_target,
                "full_design_required": bool(require_full_design),
                "full_design_validation": (
                    str(output / "full_design_validation.json")
                    if full_design_validation is not None
                    else None
                ),
                "classification_information_boundary": CLASSIFICATION_INFORMATION_BOUNDARY,
                "independent_design_provenance": INDEPENDENT_DESIGN_PROVENANCE,
                "created_at": _utc_timestamp(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    combined_validation_path = output / "validation.json"
    combined_validation = json.loads(
        combined_validation_path.read_text(encoding="utf-8")
    )
    combined_validation["checks"].update(
        {
            "all_input_shards_validated": True,
            "input_shard_targets_disjoint": True,
        }
    )
    if full_design_validation is not None:
        combined_validation["checks"].update(full_design_validation["checks"])
        combined_validation["checks"].update(
            {
                "all_prediction_bundles_present": full_design_validation[
                    "all_prediction_bundles_present"
                ],
                "prediction_rows_identical_across_k_families_and_model_seeds": (
                    full_design_validation[
                        "prediction_rows_identical_across_k_families_and_model_seeds"
                    ]
                ),
                "source_only_probabilities_identical_across_k": (
                    full_design_validation[
                        "source_only_probabilities_identical_across_k"
                    ]
                ),
            }
        )
        combined_validation["all_required_checks_pass"] = bool(
            combined_validation["all_required_checks_pass"]
            and all(combined_validation["checks"].values())
        )
        combined_validation["n_expected_result_rows"] = int(
            full_design_validation["n_expected_result_rows"]
        )
    combined_validation["n_shards"] = len(manifests)
    combined_validation_path.write_text(
        json.dumps(combined_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_status(
        output / "status.json",
        state="complete",
        n_shards=len(manifests),
        n_targets=len(observed_targets),
        n_rows=int(rows.shape[0]),
    )
    return output


def run_accuracy_push(args: argparse.Namespace) -> Path:
    """Run the hierarchical, contextual, ensemble, and structured benchmark."""

    baseline = validate_baseline_reproduction(args.baseline_results)
    if not args.cache:
        raise ValueError("--cache is required unless --aggregate-shards is used")
    cache_path = Path(args.cache).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(args, output_dir)
    partial_path = output_dir / "fold_results.partial.csv"
    member_path = output_dir / "member_results.partial.csv"
    status_path = output_dir / "status.json"
    if not args.resume:
        for path in (partial_path, member_path):
            if path.exists():
                path.unlink()

    metadata = _load_npz_metadata(cache_path)
    required = {"press_order", "press_ratios"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Accuracy push requires cache arrays: {missing}")
    raw_labels = metadata["finger_ids"].astype(np.int64)
    training_labels = relabel_minimum_overlap(
        raw_labels, metadata["press_overlap_fraction"], args.minimum_overlap
    )
    order_labels = relabel_minimum_overlap(
        metadata["press_order"], metadata["press_overlap_fraction"], args.minimum_overlap
    ).astype(np.int64)
    subject_ids = metadata["subject_indices"]
    trial_ids = metadata["trial_id"]
    sequence_ids = metadata["sequence_id"]
    composite_trials = _composite_trial_ids(subject_ids, trial_ids)
    trial_row_order = validate_supplied_trial_row_order(composite_trials)

    raw_window_path = (
        Path(args.raw_window_cache)
        if args.raw_window_cache
        else cache_path.parent / "derived" / "meg_windows_raw.npy"
    )
    prepare_raw_window_memmap(cache_path, raw_window_path)
    window_store = np.load(raw_window_path, mmap_mode="r")
    moments = prepare_subject_sensor_moments(
        window_store,
        subject_ids,
        raw_window_path.with_suffix(raw_window_path.suffix + ".subject_moments.npz"),
        batch_size=args.feature_batch_size,
    )
    moment_domains, subject_means, subject_stds = _subject_moment_arrays(moments)

    targets = _parse_csv(args.targets or ",".join(JULIA_SUBJECTS))
    split_seeds = _parse_csv(args.split_seeds, cast=int)
    model_seeds = _parse_csv(args.model_seeds, cast=int)
    k_values = _parse_csv(args.k_values, cast=int)
    adaptation_candidates = tuple(
        {
            "learning_rate": float(learning_rate),
            "source_replay_weight": float(replay_weight),
            "step_multiplier": float(multiplier),
        }
        for learning_rate in _parse_csv(
            getattr(args, "adaptation_learning_rates", "0.001,0.002"), cast=float
        )
        for replay_weight in _parse_csv(
            getattr(args, "source_replay_weights", "0.05,0.10"), cast=float
        )
        for multiplier in _parse_csv(
            getattr(args, "adaptation_step_multipliers", "0.5,1.0"), cast=float
        )
    )
    context_modes = set(_parse_csv(args.context_modes))
    if not context_modes.issubset({"offline", "causal", "none"}):
        raise ValueError("context_modes must contain offline, causal, or none")
    predeclared_adapter = {
        "adapter_kind": str(args.adapter_kind),
        "adapter_rank": int(args.adapter_rank),
        "selection": "predeclared_cli",
    }
    frozen_adapter: dict[str, Any] | None = None
    selected_config_path = getattr(args, "selected_adapter_config", None)
    if selected_config_path:
        frozen_path = Path(selected_config_path).expanduser().resolve()
        frozen_adapter = dict(predeclared_adapter)
        frozen_adapter.update(
            json.loads(frozen_path.read_text(encoding="utf-8"))
        )
        frozen_adapter["selection"] = "frozen_file"
        frozen_adapter["selection_artifact"] = str(frozen_path)
        for name in ("source_adapter_screen.csv", "source_adapter_screen_summary.csv"):
            source = frozen_path.parent / name
            if source.exists():
                pd.read_csv(source).to_csv(output_dir / name, index=False)
    elif not bool(getattr(args, "screen_adapters", True)):
        frozen_adapter = dict(predeclared_adapter)

    selected_adapters: dict[str, dict[str, Any]] = {}
    feasibility_by_target: dict[str, dict[str, Any]] = {}
    if bool(getattr(args, "screen_only", False)):
        if frozen_adapter is not None:
            raise ValueError("--screen-only requires adapter screening, not a frozen configuration")
        for target in targets:
            selected_adapters[target] = _screen_adapter_for_outer_target(
                args,
                target=target,
                output_dir=output_dir,
                model_seed=model_seeds[0],
                window_store=window_store,
                subject_ids=subject_ids,
                trial_ids=composite_trials,
                training_labels=training_labels,
                raw_finger_labels=raw_labels,
                sequence_ids=sequence_ids,
                order_labels=order_labels,
                overlap_targets=metadata["press_overlap_fraction"],
                press_ratios=metadata["press_ratios"],
                moments=moments,
                moment_domains=moment_domains,
                subject_means=subject_means,
                subject_stds=subject_stds,
            )
        _aggregate_adapter_screens(output_dir, targets)
        _write_status(
            status_path,
            state="screen_complete",
            n_outer_targets=len(selected_adapters),
            nested_outer_target_exclusion=True,
        )
        return output_dir

    completed: set[tuple[str, int, int, str]] = set()
    completed_members: set[tuple[str, int, int, str, int]] = set()
    expected_result_keys: set[tuple[str, int, int, str, int]] = set()
    if args.resume and partial_path.exists():
        prior = pd.read_csv(partial_path).drop_duplicates(
            ["target", "split_seed", "k_trials_per_sequence", "method", "model_seed"],
            keep="last",
        )
        prior.to_csv(partial_path, index=False)
        completed = set(
            zip(
                prior["target"].astype(str),
                prior["split_seed"].astype(int),
                prior["k_trials_per_sequence"].astype(int),
                prior["method"].astype(str),
                strict=True,
            )
        )
    if args.resume and member_path.exists():
        prior_members = pd.read_csv(member_path).drop_duplicates(
            ["target", "split_seed", "k_trials_per_sequence", "method", "model_seed"],
            keep="last",
        )
        prior_members.to_csv(member_path, index=False)
        completed_members = set(
            zip(
                prior_members["target"].astype(str),
                prior_members["split_seed"].astype(int),
                prior_members["k_trials_per_sequence"].astype(int),
                prior_members["method"].astype(str),
                prior_members["model_seed"].astype(int),
                strict=True,
            )
        )
    fold_counter = 0
    for target in targets:
        target_index = JULIA_SUBJECTS.index(target)
        if frozen_adapter is None:
            selected_adapter = _screen_adapter_for_outer_target(
                args,
                target=target,
                output_dir=output_dir,
                model_seed=model_seeds[0],
                window_store=window_store,
                subject_ids=subject_ids,
                trial_ids=composite_trials,
                training_labels=training_labels,
                raw_finger_labels=raw_labels,
                sequence_ids=sequence_ids,
                order_labels=order_labels,
                overlap_targets=metadata["press_overlap_fraction"],
                press_ratios=metadata["press_ratios"],
                moments=moments,
                moment_domains=moment_domains,
                subject_means=subject_means,
                subject_stds=subject_stds,
            )
        else:
            frozen_target = frozen_adapter.get("outer_test_subject")
            if frozen_target is not None and str(frozen_target) != target:
                raise ValueError(
                    f"Frozen adapter belongs to {frozen_target}, not outer target {target}"
                )
            selected_adapter = {
                **frozen_adapter,
                "outer_test_subject": target,
                "outer_target_data_used": False,
                "outer_target_labels_used": False,
            }
        selected_adapters[target] = selected_adapter
        target_global = np.flatnonzero(subject_ids == target_index)
        source_domains = np.asarray([index for index in range(len(JULIA_SUBJECTS)) if index != target_index])
        source_global = np.flatnonzero(np.isin(subject_ids, source_domains))
        pooled_mean, pooled_std = _source_pooled_moments(moments, source_domains)
        target_sequence = sequence_ids[target_global]
        target_trials = trial_ids[target_global]
        target_raw = raw_labels[target_global]
        target_training = training_labels[target_global]
        registry = pd.DataFrame({"trial": target_trials, "sequence": target_sequence}).drop_duplicates()
        per_sequence = registry.groupby("sequence")["trial"].nunique().to_numpy()
        feasible = tuple(k for k in k_values if np.all(per_sequence > k))
        if not feasible:
            raise ValueError(
                f"Target {target} cannot reserve the requested maximum k={max(k_values)} calibration pool"
            )
        feasibility_by_target[target] = {
            "trials_per_sequence": {
                str(sequence): int(count)
                for sequence, count in registry.groupby("sequence")["trial"].nunique().items()
            },
            "requested_k_values": [int(k) for k in k_values],
            "feasible_k_values": [int(k) for k in feasible],
            "infeasible_k_values": [int(k) for k in k_values if k not in feasible],
            "fixed_evaluation_pool_k": int(max(feasible)),
        }

        source_template_labels = _context_template_labels(
            training_labels,
            order_labels,
            subject_ids,
            trial_ids,
            source_global,
        )
        source_label_views = _masked_source_labels(
            source_global,
            training_labels=training_labels,
            raw_finger_labels=raw_labels,
            sequence_ids=sequence_ids,
            order_labels=order_labels,
            overlap_targets=metadata["press_overlap_fraction"],
            press_ratios=metadata["press_ratios"],
        )
        source_model_cache: dict[int, TorchProgressiveTemporalWindowClassifier] = {}
        source_context_embedding_cache: dict[int, np.ndarray | None] = {}
        context_source_cache: dict[int, dict[str, TorchKatjaTrialContextRefiner]] = {}
        adaptation_selection_cache: dict[tuple[int, int], dict[str, float]] = {}
        for split_seed in split_seeds:
            fold_counter += 1
            if args.max_folds is not None and fold_counter > args.max_folds:
                break
            splits = select_nested_trial_splits(
                target_sequence,
                target_trials,
                k_values=feasible,
                seed=split_seed,
                context=target,
                split_mode="fixed_max_complement",
            )
            for k in feasible:
                ensemble_methods = {
                    "hierarchical_source_only",
                    "hierarchical_tcn_ensemble",
                    "hierarchical_tcn_ensemble_structured",
                    "hierarchical_tcn_ensemble_causal",
                }
                if "offline" in context_modes:
                    ensemble_methods.update(
                        {
                            "trial_transformer_ensemble",
                            "trial_transformer_ensemble_structured",
                            "hybrid_ensemble",
                            "hybrid_ensemble_structured",
                        }
                    )
                if "causal" in context_modes:
                    ensemble_methods.add("trial_transformer_causal_ensemble")
                expected_result_keys.update(
                    (target, int(split_seed), int(k), method, -1)
                    for method in ensemble_methods
                )
                member_methods = {"hierarchical_tcn_single"}
                if "offline" in context_modes:
                    member_methods.add("trial_transformer_single")
                if "causal" in context_modes:
                    member_methods.add("trial_transformer_causal_single")
                expected_result_keys.update(
                    (target, int(split_seed), int(k), method, int(model_seed))
                    for method in member_methods
                    for model_seed in model_seeds
                )
            model_family_paths: dict[tuple[str, int], list[Path]] = {}
            for model_seed in model_seeds:
                expected_families = ["hierarchical_source_only", "hierarchical_tcn"]
                expected_families.extend(
                    f"trial_transformer_{mode}" for mode in sorted(context_modes - {"none"})
                )
                expected_paths = {
                    (family, k): _bundle_path(
                        output_dir,
                        family=family,
                        target=target,
                        split_seed=split_seed,
                        k=k,
                        model_seed=model_seed,
                    )
                    for family in expected_families
                    for k in feasible
                }
                expected_member_keys = {
                    (
                        target,
                        int(split_seed),
                        int(k),
                        (
                            "hierarchical_tcn_single"
                            if family == "hierarchical_tcn"
                            else (
                                "trial_transformer_causal_single"
                                if family == "trial_transformer_causal"
                                else "trial_transformer_single"
                            )
                        ),
                        int(model_seed),
                    )
                    for family in expected_families
                    if family != "hierarchical_source_only"
                    for k in feasible
                }
                if (
                    args.resume
                    and all(path.exists() for path in expected_paths.values())
                    and expected_member_keys.issubset(completed_members)
                ):
                    for family_k, path in expected_paths.items():
                        model_family_paths.setdefault(family_k, []).append(path)
                    continue
                _write_status(
                    status_path,
                    state="source_fit",
                    target=target,
                    split_seed=split_seed,
                    model_seed=model_seed,
                )
                if model_seed in source_model_cache:
                    source_model = source_model_cache[model_seed]
                    source_context_embeddings = source_context_embedding_cache[model_seed]
                    context_sources = context_source_cache[model_seed]
                else:
                    source_model = _fit_hierarchical_source_model(
                        args,
                        adapter_rank=int(selected_adapter["adapter_rank"]),
                        adapter_kind=str(selected_adapter["adapter_kind"]),
                        random_state=_stable_seed(model_seed, target, "window_source"),
                        window_store=window_store,
                        source_indices=source_global,
                        subject_ids=subject_ids,
                        trial_ids=composite_trials,
                        source_labels=source_label_views,
                        pooled_mean=pooled_mean,
                        pooled_std=pooled_std,
                        moment_domains=moment_domains,
                        subject_means=subject_means,
                        subject_stds=subject_stds,
                    )
                    source_context_embeddings = None
                    context_sources: dict[str, TorchKatjaTrialContextRefiner] = {}
                    if context_modes - {"none"}:
                        source_outputs = source_model.predict_outputs_indices(
                            source_global, source_domain_mode=True
                        )
                        source_context_embeddings = np.zeros(
                            (raw_labels.size, source_outputs["embedding"].shape[1]), dtype=np.float32
                        )
                        source_context_embeddings[source_global] = source_outputs["embedding"]
                        for mode in sorted(context_modes - {"none"}):
                            context_sources[mode] = TorchKatjaTrialContextRefiner(
                                hidden_units=args.context_hidden_units,
                                num_layers=2,
                                num_heads=args.context_heads,
                                source_epochs=args.context_source_epochs,
                                adaptation_steps=args.context_adaptation_steps,
                                batch_trials=args.context_batch_trials,
                                causal=mode == "causal",
                                random_state=_stable_seed(model_seed, target, mode, "context_source"),
                                device=args.device,
                            ).fit_source(
                                source_context_embeddings,
                                source_indices=source_global,
                                domain_ids=subject_ids,
                                trial_ids=trial_ids,
                                finger_labels=source_label_views["finger"],
                                press_ratios=source_label_views["press_ratios"],
                                order_labels=source_label_views["order"],
                                overlap_targets=source_label_views["overlap"],
                                template_labels=source_template_labels,
                                validation_finger_labels=source_label_views[
                                    "validation_finger"
                                ],
                            )
                    source_model_cache[model_seed] = source_model
                    source_context_embedding_cache[model_seed] = source_context_embeddings
                    context_source_cache[model_seed] = context_sources

                for k in feasible:
                    split = splits[k]
                    _write_status(
                        status_path,
                        state="adaptation_start",
                        target=target,
                        split_seed=int(split_seed),
                        model_seed=int(model_seed),
                        k_trials_per_sequence=int(k),
                    )
                    partition = TargetCalibrationPartition(
                        calibration_indices=target_global[split.calibration_rows],
                        evaluation_indices=target_global[split.evaluation_rows],
                        reserved_indices=target_global[split.reserved_rows],
                        split_seed=split_seed,
                    )
                    write_partition_audit(
                        output_dir
                        / "split_audits"
                        / f"target={target}"
                        / f"split_seed={split_seed}"
                        / f"k={k}.json",
                        partition,
                        k_trials_per_sequence=int(k),
                        fixed_evaluation_pool_k=int(max(feasible)),
                    )
                    source_only_outputs = source_model.predict_outputs_indices(
                        partition.evaluation_indices
                    )
                    source_only_path = _bundle_path(
                        output_dir,
                        family="hierarchical_source_only",
                        target=target,
                        split_seed=split_seed,
                        k=k,
                        model_seed=model_seed,
                    )
                    write_prediction_bundle(
                        source_only_path,
                        row_indices=partition.evaluation_indices,
                        probabilities=source_only_outputs["probabilities"],
                        split_seed=split_seed,
                        model_seed=model_seed,
                        method="hierarchical_source_only",
                        auxiliary=_output_auxiliary(source_only_outputs),
                    )
                    model_family_paths.setdefault(("hierarchical_source_only", k), []).append(
                        source_only_path
                    )
                    model = source_model.clone_source(
                        random_state=_stable_seed(model_seed, target, split_seed, k, "target")
                    )
                    selection_key = (int(split_seed), int(k))
                    selection_path = (
                        output_dir
                        / "adaptation_selection"
                        / f"target={target}"
                        / f"split_seed={split_seed}"
                        / f"k={k}.json"
                    )
                    if selection_key in adaptation_selection_cache:
                        selected_adaptation = adaptation_selection_cache[selection_key]
                    elif args.resume and selection_path.exists():
                        selected_adaptation = json.loads(
                            selection_path.read_text(encoding="utf-8")
                        )["selected_adaptation"]
                        adaptation_selection_cache[selection_key] = selected_adaptation
                    elif bool(getattr(args, "tune_adaptation", True)):
                        selected_adaptation, tuning_rows = select_window_adaptation_hyperparameters(
                            source_model,
                            calibration_indices=partition.calibration_indices,
                            evaluation_indices=partition.evaluation_indices,
                            sequence_ids=sequence_ids,
                            trial_ids=composite_trials,
                            finger_labels=training_labels,
                            raw_finger_labels=raw_labels,
                            order_labels=order_labels,
                            overlap_targets=metadata["press_overlap_fraction"],
                            press_ratios=metadata["press_ratios"],
                            candidates=adaptation_candidates,
                            seed=_stable_seed(model_seed, target, split_seed, k, "adaptation_tuning"),
                        )
                        adaptation_selection_cache[selection_key] = selected_adaptation
                        for tuning_row in tuning_rows:
                            _append_row(
                                output_dir / "adaptation_selection.partial.csv",
                                {
                                    "target": target,
                                    "split_seed": int(split_seed),
                                    "model_seed": int(model_seed),
                                    "k_trials_per_sequence": int(k),
                                    **tuning_row,
                                    "selected": bool(
                                        float(tuning_row["learning_rate"])
                                        == selected_adaptation["learning_rate"]
                                        and float(tuning_row["source_replay_weight"])
                                        == selected_adaptation["source_replay_weight"]
                                        and float(tuning_row["step_multiplier"])
                                        == selected_adaptation["step_multiplier"]
                                    ),
                                },
                            )
                        selection_path.parent.mkdir(parents=True, exist_ok=True)
                        selection_path.write_text(
                            json.dumps(
                                {
                                    "target": target,
                                    "split_seed": int(split_seed),
                                    "k_trials_per_sequence": int(k),
                                    "selected_adaptation": selected_adaptation,
                                    "selected_using_evaluation_labels": False,
                                    "selected_with_model_seed": int(model_seed),
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    else:
                        selected_adaptation = dict(adaptation_candidates[0])
                        adaptation_selection_cache[selection_key] = selected_adaptation
                    model.adaptation_learning_rate = selected_adaptation["learning_rate"]
                    model.full_finetune_learning_rate = selected_adaptation["learning_rate"] / 10.0
                    model.source_replay_weight = selected_adaptation["source_replay_weight"]
                    model.adapter_steps = max(
                        1, int(round(source_model.adapter_steps * selected_adaptation["step_multiplier"]))
                    )
                    model.last_block_steps = max(
                        0, int(round(source_model.last_block_steps * selected_adaptation["step_multiplier"]))
                    )
                    model.full_finetune_steps = max(
                        0, int(round(source_model.full_finetune_steps * selected_adaptation["step_multiplier"]))
                    )
                    model.register_target_calibration_labels(
                        partition.calibration_indices,
                        finger_labels=training_labels[partition.calibration_indices],
                        sequence_labels=sequence_ids[partition.calibration_indices],
                        order_labels=order_labels[partition.calibration_indices],
                        overlap_targets=metadata["press_overlap_fraction"][partition.calibration_indices],
                        press_ratios=metadata["press_ratios"][partition.calibration_indices],
                    )
                    model.adapt_target_indices(
                        partition.calibration_indices,
                        n_calibration_trials=int(split.calibration_trials.size),
                        mode="progressive_full",
                    )
                    evaluation_outputs = model.predict_outputs_indices(partition.evaluation_indices)
                    calibration_outputs = model.predict_outputs_indices(
                        partition.calibration_indices
                    )
                    temperature = fit_probability_temperature(
                        calibration_outputs["probabilities"],
                        raw_labels[partition.calibration_indices],
                    )
                    evaluation_outputs["probabilities"] = apply_probability_temperature(
                        evaluation_outputs["probabilities"], temperature
                    )
                    tcn_auxiliary = _output_auxiliary(evaluation_outputs)
                    tcn_auxiliary["calibration_temperature"] = np.full(
                        partition.evaluation_indices.size, temperature, dtype=np.float32
                    )
                    window_path = _bundle_path(
                        output_dir,
                        family="hierarchical_tcn",
                        target=target,
                        split_seed=split_seed,
                        k=k,
                        model_seed=model_seed,
                    )
                    write_prediction_bundle(
                        window_path,
                        row_indices=partition.evaluation_indices,
                        probabilities=evaluation_outputs["probabilities"],
                        split_seed=split_seed,
                        model_seed=model_seed,
                        method="hierarchical_tcn",
                        auxiliary=tcn_auxiliary,
                    )
                    model_family_paths.setdefault(("hierarchical_tcn", k), []).append(window_path)
                    member_key = (
                        target,
                        int(split_seed),
                        int(k),
                        "hierarchical_tcn_single",
                        int(model_seed),
                    )
                    if member_key not in completed_members:
                        member_row = _metric_and_append(
                            member_path,
                            method="hierarchical_tcn_single",
                            target=target,
                            target_index=target_index,
                            split_seed=split_seed,
                            model_seed=model_seed,
                            split=split,
                            probabilities=evaluation_outputs["probabilities"],
                            raw_labels_target=target_raw,
                            training_labels_target=target_training,
                            target_trials=target_trials,
                            n_source_windows=source_global.size,
                            n_source_subjects=source_domains.size,
                            adaptation_stages=",".join(
                                item["stage"] for item in model.adaptation_history_
                            ),
                        )
                        completed_members.add(member_key)
                        print(
                            f"{target} split={split_seed} model={model_seed} k={k} "
                            f"hierarchical_tcn={member_row['accuracy_raw_labels']:.4f}",
                            flush=True,
                        )

                    if context_sources:
                        _write_status(
                            status_path,
                            state="context_start",
                            target=target,
                            split_seed=int(split_seed),
                            model_seed=int(model_seed),
                            k_trials_per_sequence=int(k),
                        )
                        context_rows = np.concatenate(
                            (partition.calibration_indices, partition.evaluation_indices)
                        )
                        target_outputs = model.predict_outputs_indices(context_rows)
                        for mode, context_source in context_sources.items():
                            context = context_source.clone_source(
                                random_state=_stable_seed(
                                    model_seed, target, split_seed, k, mode, "target_context"
                                )
                            )
                            assert source_context_embeddings is not None
                            context.embeddings_ = source_context_embeddings.copy()
                            context.embeddings_[context_rows] = target_outputs["embedding"]
                            target_template_rows = _context_template_labels(
                                training_labels,
                                order_labels,
                                subject_ids,
                                trial_ids,
                                partition.calibration_indices,
                            )
                            context.register_target_calibration_labels(
                                partition.calibration_indices,
                                finger_labels=training_labels[partition.calibration_indices],
                                press_ratios=metadata["press_ratios"][partition.calibration_indices],
                                order_labels=order_labels[partition.calibration_indices],
                                overlap_targets=metadata["press_overlap_fraction"][partition.calibration_indices],
                                template_labels=target_template_rows[partition.calibration_indices],
                            )
                            context.adapt_target_indices(partition.calibration_indices)
                            context_outputs = context.predict_outputs_indices(
                                partition.evaluation_indices
                            )
                            context_calibration_outputs = context.predict_outputs_indices(
                                partition.calibration_indices
                            )
                            context_temperature = fit_probability_temperature(
                                context_calibration_outputs["probabilities"],
                                raw_labels[partition.calibration_indices],
                            )
                            context_outputs["probabilities"] = apply_probability_temperature(
                                context_outputs["probabilities"], context_temperature
                            )
                            context_auxiliary = _output_auxiliary(context_outputs)
                            context_auxiliary["calibration_temperature"] = np.full(
                                partition.evaluation_indices.size,
                                context_temperature,
                                dtype=np.float32,
                            )
                            family = f"trial_transformer_{mode}"
                            context_path = _bundle_path(
                                output_dir,
                                family=family,
                                target=target,
                                split_seed=split_seed,
                                k=k,
                                model_seed=model_seed,
                            )
                            write_prediction_bundle(
                                context_path,
                                row_indices=partition.evaluation_indices,
                                probabilities=context_outputs["probabilities"],
                                split_seed=split_seed,
                                model_seed=model_seed,
                                method=family,
                                auxiliary=context_auxiliary,
                            )
                            model_family_paths.setdefault((family, k), []).append(context_path)
                            member_method = (
                                "trial_transformer_causal_single"
                                if mode == "causal"
                                else "trial_transformer_single"
                            )
                            member_key = (
                                target,
                                int(split_seed),
                                int(k),
                                member_method,
                                int(model_seed),
                            )
                            if member_key not in completed_members:
                                _metric_and_append(
                                    member_path,
                                    method=member_method,
                                    target=target,
                                    target_index=target_index,
                                    split_seed=split_seed,
                                    model_seed=model_seed,
                                    split=split,
                                    probabilities=context_outputs["probabilities"],
                                    raw_labels_target=target_raw,
                                    training_labels_target=target_training,
                                    target_trials=target_trials,
                                    n_source_windows=source_global.size,
                                    n_source_subjects=source_domains.size,
                                    adaptation_stages=f"hierarchical_progressive_full,{family}",
                                    decoding_mode=(
                                        "causal_context"
                                        if mode == "causal"
                                        else "offline_context"
                                    ),
                                )
                                completed_members.add(member_key)

                del source_model, source_context_embeddings, context_sources

            for k in feasible:
                split = splits[k]
                _write_status(
                    status_path,
                    state="ensemble_start",
                    target=target,
                    split_seed=int(split_seed),
                    model_seed=-1,
                    k_trials_per_sequence=int(k),
                )
                evaluation_global = target_global[split.evaluation_rows]
                calibration_global = target_global[split.calibration_rows]
                reserved_global = target_global[split.reserved_rows]
                partition = TargetCalibrationPartition(
                    calibration_global, evaluation_global, reserved_global, split_seed
                )
                templates = learn_finger_templates(
                    training_labels,
                    order_labels,
                    composite_trials,
                    calibration_indices=calibration_global,
                    evaluation_indices=evaluation_global,
                )
                fitting_rows = np.concatenate((source_global, calibration_global))
                stay = estimate_state_stay_probabilities(
                    training_labels,
                    order_labels,
                    composite_trials,
                    fitting_indices=fitting_rows,
                    evaluation_indices=evaluation_global,
                )

                ensembles: dict[str, dict[str, Any]] = {}
                for family in (
                    "hierarchical_source_only",
                    "hierarchical_tcn",
                    "trial_transformer_offline",
                    "trial_transformer_causal",
                ):
                    paths = model_family_paths.get((family, k), [])
                    if len(paths) != len(model_seeds):
                        continue
                    ensembles[family] = ensemble_prediction_bundles(paths)

                if "hierarchical_source_only" in ensembles:
                    method = "hierarchical_source_only"
                    key = (target, split_seed, k, method)
                    if key not in completed:
                        _metric_and_append(
                            partial_path,
                            method=method,
                            target=target,
                            target_index=target_index,
                            split_seed=split_seed,
                            model_seed=-1,
                            split=split,
                            probabilities=ensembles["hierarchical_source_only"]["probabilities"],
                            raw_labels_target=target_raw,
                            training_labels_target=target_training,
                            target_trials=target_trials,
                            n_source_windows=source_global.size,
                            n_source_subjects=source_domains.size,
                            adaptation_stages="none",
                            decoding_mode="independent_windows_source_only",
                        )
                        completed.add(key)

                if "hierarchical_tcn" in ensembles:
                    tcn = ensembles["hierarchical_tcn"]["probabilities"]
                    for method, probabilities, decoding in (
                        ("hierarchical_tcn_ensemble", tcn, "independent_windows"),
                    ):
                        key = (target, split_seed, k, method)
                        if key not in completed:
                            _metric_and_append(
                                partial_path,
                                method=method,
                                target=target,
                                target_index=target_index,
                                split_seed=split_seed,
                                model_seed=-1,
                                split=split,
                                probabilities=probabilities,
                                raw_labels_target=target_raw,
                                training_labels_target=target_training,
                                target_trials=target_trials,
                                n_source_windows=source_global.size,
                                n_source_subjects=source_domains.size,
                                adaptation_stages="hierarchical_progressive_full",
                                decoding_mode=decoding,
                            )
                            completed.add(key)
                    structured = _forced_trial_predictions(
                        tcn,
                        evaluation_global,
                        composite_trials,
                        templates,
                        stay,
                        causal=False,
                        auxiliary=ensembles["hierarchical_tcn"],
                    )
                    causal = _forced_trial_predictions(
                        tcn,
                        evaluation_global,
                        composite_trials,
                        templates,
                        stay,
                        causal=True,
                        auxiliary=ensembles["hierarchical_tcn"],
                    )
                    for method, predicted, decoding in (
                        ("hierarchical_tcn_ensemble_structured", structured, "offline_structured"),
                        ("hierarchical_tcn_ensemble_causal", causal, "causal_prefix"),
                    ):
                        key = (target, split_seed, k, method)
                        if key not in completed:
                            _metric_and_append(
                                partial_path,
                                method=method,
                                target=target,
                                target_index=target_index,
                                split_seed=split_seed,
                                model_seed=-1,
                                split=split,
                                probabilities=tcn,
                                predicted_labels=predicted,
                                raw_labels_target=target_raw,
                                training_labels_target=target_training,
                                target_trials=target_trials,
                                n_source_windows=source_global.size,
                                n_source_subjects=source_domains.size,
                                adaptation_stages="hierarchical_progressive_full,task_structure",
                                decoding_mode=decoding,
                            )
                            completed.add(key)

                for mode in ("offline", "causal"):
                    family = f"trial_transformer_{mode}"
                    if family not in ensembles:
                        continue
                    probabilities = ensembles[family]["probabilities"]
                    ensemble_name = (
                        "trial_transformer_ensemble"
                        if mode == "offline"
                        else "trial_transformer_causal_ensemble"
                    )
                    for method, values, model_seed in (
                        (ensemble_name, probabilities, -1),
                    ):
                        key = (target, split_seed, k, method)
                        if key not in completed:
                            _metric_and_append(
                                partial_path,
                                method=method,
                                target=target,
                                target_index=target_index,
                                split_seed=split_seed,
                                model_seed=model_seed,
                                split=split,
                                probabilities=values,
                                raw_labels_target=target_raw,
                                training_labels_target=target_training,
                                target_trials=target_trials,
                                n_source_windows=source_global.size,
                                n_source_subjects=source_domains.size,
                                adaptation_stages=f"hierarchical_progressive_full,{family}",
                                decoding_mode="causal_context" if mode == "causal" else "offline_context",
                            )
                            completed.add(key)
                    if mode == "offline":
                        predicted = _forced_trial_predictions(
                            probabilities,
                            evaluation_global,
                            composite_trials,
                            templates,
                            stay,
                            causal=False,
                            auxiliary=ensembles[family],
                        )
                        method = "trial_transformer_ensemble_structured"
                        key = (target, split_seed, k, method)
                        if key not in completed:
                            _metric_and_append(
                                partial_path,
                                method=method,
                                target=target,
                                target_index=target_index,
                                split_seed=split_seed,
                                model_seed=-1,
                                split=split,
                                probabilities=probabilities,
                                predicted_labels=predicted,
                                raw_labels_target=target_raw,
                                training_labels_target=target_training,
                                target_trials=target_trials,
                                n_source_windows=source_global.size,
                                n_source_subjects=source_domains.size,
                                adaptation_stages="hierarchical_progressive_full,offline_context,task_structure",
                                decoding_mode="offline_structured",
                            )
                            completed.add(key)

                if "hierarchical_tcn" in ensembles and "trial_transformer_offline" in ensembles:
                    hybrid = 0.5 * (
                        ensembles["hierarchical_tcn"]["probabilities"]
                        + ensembles["trial_transformer_offline"]["probabilities"]
                    )
                    hybrid_auxiliary: dict[str, np.ndarray] = {}
                    for name in ("aux_order_probabilities", "aux_overlap_prediction"):
                        if (
                            name in ensembles["hierarchical_tcn"]
                            and name in ensembles["trial_transformer_offline"]
                        ):
                            hybrid_auxiliary[name] = 0.5 * (
                                ensembles["hierarchical_tcn"][name]
                                + ensembles["trial_transformer_offline"][name]
                            )
                    if "aux_template_probabilities" in ensembles["trial_transformer_offline"]:
                        hybrid_auxiliary["aux_template_probabilities"] = ensembles[
                            "trial_transformer_offline"
                        ]["aux_template_probabilities"]
                    for method, structured_mode in (
                        ("hybrid_ensemble", False),
                        ("hybrid_ensemble_structured", True),
                    ):
                        key = (target, split_seed, k, method)
                        if key in completed:
                            continue
                        predicted = None
                        decoding = "offline_context_ensemble"
                        if structured_mode:
                            predicted = _forced_trial_predictions(
                                hybrid,
                                evaluation_global,
                                composite_trials,
                                templates,
                                stay,
                                causal=False,
                                auxiliary=hybrid_auxiliary,
                            )
                            decoding = "offline_structured"
                        _metric_and_append(
                            partial_path,
                            method=method,
                            target=target,
                            target_index=target_index,
                            split_seed=split_seed,
                            model_seed=-1,
                            split=split,
                            probabilities=hybrid,
                            predicted_labels=predicted,
                            raw_labels_target=target_raw,
                            training_labels_target=target_training,
                            target_trials=target_trials,
                            n_source_windows=source_global.size,
                            n_source_subjects=source_domains.size,
                            adaptation_stages="five_seed_tcn_context_ensemble",
                            decoding_mode=decoding,
                        )
                        completed.add(key)

            _write_status(
                status_path,
                state="fold_done",
                target=target,
                split_seed=split_seed,
                n_completed_rows=len(completed),
            )
        source_model_cache.clear()
        source_context_embedding_cache.clear()
        context_source_cache.clear()
        if args.max_folds is not None and fold_counter >= args.max_folds:
            break

    if not partial_path.exists():
        raise RuntimeError("No accuracy-push rows were produced")
    rows = pd.read_csv(partial_path).drop_duplicates(
        ["target", "split_seed", "k_trials_per_sequence", "method", "model_seed"],
        keep="last",
    )
    if member_path.exists():
        member_rows = pd.read_csv(member_path).drop_duplicates(
            ["target", "split_seed", "k_trials_per_sequence", "method", "model_seed"],
            keep="last",
        )
        member_rows.to_csv(output_dir / "member_results.csv", index=False)
        rows = pd.concat((rows, member_rows), ignore_index=True)
    rows = rows.sort_values(
        ["method", "k_trials_per_sequence", "target", "split_seed", "model_seed"]
    ).reset_index(drop=True)
    actual_result_keys = set(
        zip(
            rows["target"].astype(str),
            rows["split_seed"].astype(int),
            rows["k_trials_per_sequence"].astype(int),
            rows["method"].astype(str),
            rows["model_seed"].astype(int),
            strict=True,
        )
    )
    missing_result_keys = sorted(expected_result_keys - actual_result_keys)
    coverage = {
        "all_expected_results_present": not missing_result_keys,
        "n_expected_result_rows": len(expected_result_keys),
        "n_actual_result_rows": len(actual_result_keys),
        "missing_result_identities": [list(key) for key in missing_result_keys[:100]],
        "missing_result_identity_count": len(missing_result_keys),
    }
    (output_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if missing_result_keys:
        _write_status(
            status_path,
            state="failed_incomplete_coverage",
            n_missing_result_rows=len(missing_result_keys),
        )
        raise RuntimeError(
            f"Accuracy-push sweep is missing {len(missing_result_keys)} expected rows"
        )
    _aggregate_adapter_screens(output_dir, list(selected_adapters))
    (output_dir / "selected_adapter_configs.json").write_text(
        json.dumps(selected_adapters, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "feasibility.json").write_text(
        json.dumps(feasibility_by_target, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_push_reports(rows, output_dir)
    validation_path = output_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["checks"]["all_expected_results_present"] = True
    validation["n_expected_result_rows"] = len(expected_result_keys)
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "cache": str(cache_path),
        "baseline_reproduction": baseline,
        "targets": list(targets),
        "split_seeds": list(split_seeds),
        "model_seeds": list(model_seeds),
        "selected_adapters_by_outer_target": selected_adapters,
        "feasibility_by_target": feasibility_by_target,
        "adapter_selection_uses_outer_target": False,
        "k_values": list(k_values),
        "split_mode": "fixed_max_complement",
        "evaluation_rows_fixed_across_k": True,
        "target_label_boundary": "source plus explicit calibration rows only",
        "classification_information_boundary": CLASSIFICATION_INFORMATION_BOUNDARY,
        "independent_design_provenance": INDEPENDENT_DESIGN_PROVENANCE,
        "hierarchical_target": "P(press) times P(finger_given_press) from press_ratios",
        "structured_decode": (
            "calibration-observed two-template monotonic Viterbi with fixed auxiliary "
            "order=0.35, overlap=0.15, template=0.15 log-emission weights"
        ),
        "offline_structured_uses_future_windows": True,
        "causal_variant_reported_separately": True,
        "trial_window_order": trial_row_order,
        "created_at": _utc_timestamp(),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_status(status_path, state="complete", n_rows=int(rows.shape[0]))
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument(
        "--aggregate-shards",
        help="Comma-separated completed shard directories; aggregate instead of fitting.",
    )
    parser.add_argument(
        "--require-full-design",
        action="store_true",
        help=(
            "During shard aggregation, require the exact ten-target, five-split, "
            "five-model-seed design and inspect all prediction bundle row identities."
        ),
    )
    parser.add_argument("--raw-window-cache")
    parser.add_argument("--targets", default=",".join(JULIA_SUBJECTS))
    parser.add_argument("--k-values", default=",".join(str(value) for value in DEFAULT_K_VALUES))
    parser.add_argument("--split-seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--model-seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--context-modes", default="offline,causal")
    parser.add_argument("--minimum-overlap", type=float, default=0.2)
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--adapter-rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument(
        "--adapter-kind",
        choices=("low_rank", "channel_affine_residual"),
        default="low_rank",
    )
    parser.add_argument(
        "--screen-adapters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Screen adapter configurations with held-out-source finger accuracy, then freeze one.",
    )
    parser.add_argument(
        "--adapter-configurations",
        default="low_rank:8,low_rank:16,low_rank:32,channel_affine_residual:16",
    )
    parser.add_argument("--adapter-screen-source-epochs", type=int, default=4)
    parser.add_argument(
        "--adapter-screen-k",
        type=int,
        default=10,
        help="Complete pseudo-target calibration trials per sequence used to screen adapters.",
    )
    parser.add_argument(
        "--adapter-screen-fold-limit",
        type=int,
        help="Limit source-LOSO subjects for engineering smoke tests only.",
    )
    parser.add_argument("--selected-adapter-config")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--source-epochs", type=int, default=12)
    parser.add_argument("--source-validation-patience", type=int, default=4)
    parser.add_argument("--adapter-steps", type=int, default=100)
    parser.add_argument("--last-block-steps", type=int, default=80)
    parser.add_argument("--full-finetune-steps", type=int, default=80)
    parser.add_argument(
        "--tune-adaptation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Choose adaptation settings by complete-trial calibration-only inner validation.",
    )
    parser.add_argument("--adaptation-learning-rates", default="0.001,0.002")
    parser.add_argument("--source-replay-weights", default="0.05,0.10")
    parser.add_argument("--adaptation-step-multipliers", default="0.5,1.0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sequence-loss-weight", type=float, default=0.15)
    parser.add_argument("--order-loss-weight", type=float, default=0.30)
    parser.add_argument("--overlap-loss-weight", type=float, default=0.30)
    parser.add_argument("--context-hidden-units", type=int, default=128)
    parser.add_argument("--context-heads", type=int, default=4)
    parser.add_argument("--context-source-epochs", type=int, default=6)
    parser.add_argument("--context-adaptation-steps", type=int, default=80)
    parser.add_argument("--context-batch-trials", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.aggregate_shards:
        aggregate_accuracy_push_shards(
            _parse_csv(args.aggregate_shards),
            output_dir=args.out_dir,
            baseline_results=args.baseline_results,
            require_full_design=bool(args.require_full_design),
        )
    else:
        run_accuracy_push(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PUSH_METHODS",
    "aggregate_accuracy_push_shards",
    "build_arg_parser",
    "main",
    "run_accuracy_push",
    "validate_baseline_reproduction",
]
