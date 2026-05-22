"""Strict source-only nested top-k ensemble decoding for BUSH-MEG.

The runner reuses ``bushmeg_source_loso`` data loading and candidate features,
keeps cue files out of the workflow, scores candidates by inner source-subject
LOSO only, and refits a weighted probability ensemble on all source subjects for
the held-out participant.  It is intended to reduce brittle single-candidate
selection among near-tied BUSH-MEG source-only decoders.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neureptrace.bushmeg_source_loso import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SELECTION_METRIC,
    MINIMIZE_SELECTION_METRICS,
    SUPPORTED_SELECTION_METRICS,
    CandidateSpec,
    FeatureCache,
    SubjectEpochs,
    _candidate_grid,
    _candidate_metrics,
    _candidate_rowspec,
    _inner_loso_scores,
    _load_subjects_from_config,
    _predict_candidate,
    _resolve_output,
    _section,
)
from neureptrace.dataset_config import apply_overrides, load_config

DEFAULT_ENSEMBLE_TOP_K = 5
DEFAULT_ENSEMBLE_WEIGHTING = "softmax"
DEFAULT_ENSEMBLE_TEMPERATURE = 0.01
ENSEMBLE_WEIGHTING_MODES = {"uniform", "rank", "softmax"}


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    candidate: CandidateSpec
    mean_score: float
    std_score: float
    comparable_score: float
    n_folds: int


@dataclass(frozen=True, slots=True)
class EnsembleMember:
    candidate: CandidateSpec
    rank: int
    weight: float
    mean_score: float
    std_score: float
    comparable_score: float


@dataclass(frozen=True, slots=True)
class EnsembleSelection:
    members: tuple[EnsembleMember, ...]
    selection_metric: str
    weighting: str
    temperature: float | None

    @property
    def name(self) -> str:
        return f"ensemble_top{len(self.members)}__{self.weighting}"


def _larger_is_better(score: float, metric: str) -> float:
    return -float(score) if metric in MINIMIZE_SELECTION_METRICS else float(score)


def _normalize_weighting(value: Any) -> str:
    mode = DEFAULT_ENSEMBLE_WEIGHTING if value is None else str(value).strip().lower().replace("-", "_")
    mode = {"average": "uniform", "mean": "uniform", "equal": "uniform", "inverse_rank": "rank", "rank_weighted": "rank", "exp": "softmax"}.get(mode, mode)
    if mode not in ENSEMBLE_WEIGHTING_MODES:
        raise ValueError(f"Unknown ensemble weighting {value!r}; choose one of {sorted(ENSEMBLE_WEIGHTING_MODES)}.")
    return mode


def _normalize_temperature(value: Any, weighting: str) -> float | None:
    if weighting != "softmax":
        return None
    temperature = DEFAULT_ENSEMBLE_TEMPERATURE if value is None else float(value)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("source_loso.ensemble_temperature must be positive and finite.")
    return temperature


def _weights_from_scores(scores: Sequence[float], *, weighting: str, temperature: float | None) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Ensemble scores must be a non-empty finite vector.")
    weighting = _normalize_weighting(weighting)
    if weighting == "uniform" or scores.size == 1:
        weights = np.ones(scores.size, dtype=float)
    elif weighting == "rank":
        weights = 1.0 / np.arange(1, scores.size + 1, dtype=float)
    else:
        temperature = _normalize_temperature(temperature, weighting)
        weights = np.exp(np.clip((scores - float(np.max(scores))) / float(temperature), -60.0, 0.0))
    return weights / float(np.sum(weights))


def _candidate_summary(candidate: CandidateSpec, rows: Sequence[Mapping[str, Any]], metric: str) -> CandidateSummary:
    frame = pd.DataFrame(rows)
    mean_score = float(frame[metric].mean())
    std_score = float(frame[metric].std(ddof=0))
    return CandidateSummary(candidate, mean_score, std_score, _larger_is_better(mean_score, metric), len(frame))


def _select_ensemble(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    candidates: Sequence[CandidateSpec],
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    metric: str,
    top_k: int,
    weighting: str,
    temperature: float | None,
) -> tuple[EnsembleSelection, list[dict[str, Any]], list[dict[str, Any]]]:
    if metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unknown selection metric {metric!r}; choose one of {sorted(SUPPORTED_SELECTION_METRICS)}.")
    if top_k < 1:
        raise ValueError("source_loso.ensemble_top_k must be at least 1.")
    inner_rows: list[dict[str, Any]] = []
    summaries: list[CandidateSummary] = []
    for candidate in candidates:
        rows = _inner_loso_scores(subjects=subjects, cache=cache, candidate=candidate, outer_test_subject=outer_test_subject, n_classes=n_classes, max_iter=max_iter)
        inner_rows.extend(rows)
        summaries.append(_candidate_summary(candidate, rows, metric))
    ranked = sorted(summaries, key=lambda item: (-item.comparable_score, item.candidate.name))
    selected = ranked[: min(top_k, len(ranked))]
    weights = _weights_from_scores([item.comparable_score for item in selected], weighting=weighting, temperature=temperature)
    members = tuple(EnsembleMember(item.candidate, rank, float(weight), item.mean_score, item.std_score, item.comparable_score) for rank, (item, weight) in enumerate(zip(selected, weights, strict=True), start=1))
    rank_rows = [
        {
            "outer_test_subject": outer_test_subject,
            "rank": rank,
            "selected": rank <= len(members),
            **_candidate_rowspec(item.candidate),
            "inner_selection_metric": metric,
            "inner_mean_score": item.mean_score,
            "inner_std_score": item.std_score,
            "inner_comparable_score": item.comparable_score,
            "inner_n_folds": item.n_folds,
            "ensemble_weight": float(weights[rank - 1]) if rank <= len(weights) else 0.0,
        }
        for rank, item in enumerate(ranked, start=1)
    ]
    return EnsembleSelection(members, metric, weighting, temperature), inner_rows, rank_rows


def _renormalize(probabilities: np.ndarray) -> np.ndarray:
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise ValueError("Ensemble probabilities must have positive finite row sums.")
    return probabilities / row_sums


def _predict_ensemble(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    selection: EnsembleSelection,
    train_subjects: Sequence[str],
    test_subject: str,
    n_classes: int,
    max_iter: int,
) -> np.ndarray:
    probability_sum = np.zeros((len(subjects[test_subject].labels), n_classes), dtype=float)
    for member in selection.members:
        probability_sum += member.weight * _predict_candidate(subjects=subjects, cache=cache, candidate=member.candidate, train_subjects=train_subjects, test_subject=test_subject, n_classes=n_classes, max_iter=max_iter)
    return _renormalize(probability_sum)


def _selection_rowspec(selection: EnsembleSelection) -> dict[str, Any]:
    primary = selection.members[0]
    return {
        **_candidate_rowspec(primary.candidate),
        "candidate": selection.name,
        "primary_candidate": primary.candidate.name,
        "ensemble_size": len(selection.members),
        "ensemble_weighting": selection.weighting,
        "ensemble_temperature": "" if selection.temperature is None else selection.temperature,
        "ensemble_candidates": "|".join(member.candidate.name for member in selection.members),
        "ensemble_weights": "|".join(f"{member.weight:.8g}" for member in selection.members),
        "ensemble_inner_scores": "|".join(f"{member.mean_score:.8g}" for member in selection.members),
        "ensemble_decoders": "|".join(dict.fromkeys(str(member.candidate.decoder) for member in selection.members)),
    }


def _write_json_sidecar(path: Path, payload: Mapping[str, Any]) -> None:
    Path(str(path) + ".provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_bushmeg_source_loso_ensemble(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    inner_cv_out_path: str | Path | None = None,
    predictions_out_path: str | Path | None = None,
    candidate_summary_out_path: str | Path | None = None,
) -> pd.DataFrame:
    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    source_loso = _section(config, "source_loso")
    metric = str(source_loso.get("selection_metric", DEFAULT_SELECTION_METRIC))
    top_k = int(source_loso.get("ensemble_top_k", DEFAULT_ENSEMBLE_TOP_K))
    weighting = _normalize_weighting(source_loso.get("ensemble_weighting", DEFAULT_ENSEMBLE_WEIGHTING))
    temperature = _normalize_temperature(source_loso.get("ensemble_temperature", DEFAULT_ENSEMBLE_TEMPERATURE), weighting)
    max_iter = int((_section(config, "decoding") or {}).get("max_iter", 1000))

    subjects, encoder = _load_subjects_from_config(config, config_dir=config_path.parent)
    candidates = _candidate_grid(config)
    cache = FeatureCache(subjects)
    n_classes = len(encoder.classes_)

    out = Path(out_path) if out_path is not None else _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_summary_csv", default="source_loso_ensemble_summary.csv")
    inner_out = Path(inner_cv_out_path) if inner_cv_out_path is not None else _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_inner_cv_csv", default="source_loso_ensemble_inner_cv.csv")
    pred_out = Path(predictions_out_path) if predictions_out_path is not None else _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_predictions_csv", default="source_loso_ensemble_predictions.csv")
    rank_out = Path(candidate_summary_out_path) if candidate_summary_out_path is not None else _resolve_output(config, config_dir=config_path.parent, key="source_loso_ensemble_candidate_summary_csv", default="source_loso_ensemble_candidate_summary.csv")

    summary_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outer_test_subject in sorted(subjects):
        selection, fold_inner_rows, fold_rank_rows = _select_ensemble(subjects=subjects, cache=cache, candidates=candidates, outer_test_subject=outer_test_subject, n_classes=n_classes, max_iter=max_iter, metric=metric, top_k=top_k, weighting=weighting, temperature=temperature)
        inner_rows.extend(fold_inner_rows)
        rank_rows.extend(fold_rank_rows)
        train_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
        probabilities = _predict_ensemble(subjects=subjects, cache=cache, selection=selection, train_subjects=train_subjects, test_subject=outer_test_subject, n_classes=n_classes, max_iter=max_iter)
        labels = subjects[outer_test_subject].labels
        predictions = probabilities.argmax(axis=1)
        member_scores = np.asarray([member.mean_score for member in selection.members], dtype=float)
        member_weights = np.asarray([member.weight for member in selection.members], dtype=float)
        summary_rows.append({
            "outer_test_subject": outer_test_subject,
            **_selection_rowspec(selection),
            "inner_selection_metric": metric,
            "inner_mean_score": float(np.average(member_scores, weights=member_weights)),
            "inner_best_score": float(np.min(member_scores) if metric in MINIMIZE_SELECTION_METRICS else np.max(member_scores)),
            "inner_std_score": float(np.std(member_scores, ddof=0)),
            "inner_n_folds": len(train_subjects),
            **_candidate_metrics(probabilities, labels, n_classes=n_classes),
            "n_train_subjects": len(train_subjects),
            "n_test_trials": len(labels),
            "n_classes": n_classes,
            "class_names": "|".join(map(str, encoder.classes_)),
        })
        metadata = subjects[outer_test_subject].metadata.reset_index(drop=True)
        for row_idx, (true_label, predicted_label) in enumerate(zip(labels, predictions, strict=True)):
            row: dict[str, Any] = {
                "outer_test_subject": outer_test_subject,
                "trial_index": int(row_idx),
                "candidate": selection.name,
                "primary_candidate": selection.members[0].candidate.name,
                "ensemble_size": len(selection.members),
                "ensemble_candidates": "|".join(member.candidate.name for member in selection.members),
                "ensemble_weights": "|".join(f"{member.weight:.8g}" for member in selection.members),
                "true_label": int(true_label),
                "true_class": str(encoder.classes_[true_label]),
                "predicted_label": int(predicted_label),
                "predicted_class": str(encoder.classes_[predicted_label]),
                "probability_true_class": float(probabilities[row_idx, true_label]),
                "confidence": float(np.max(probabilities[row_idx])),
                "is_correct": bool(predicted_label == true_label),
            }
            for column in ("participant", "condition", "stimulus_class"):
                if column in metadata.columns:
                    row[column] = metadata.loc[row_idx, column]
            for class_idx, class_name in enumerate(encoder.classes_):
                row[f"class_{class_idx}"] = str(class_name)
                row[f"prob_class_{class_idx}"] = float(probabilities[row_idx, class_idx])
            prediction_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    for path, frame in ((out, summary), (inner_out, pd.DataFrame(inner_rows)), (rank_out, pd.DataFrame(rank_rows)), (pred_out, pd.DataFrame(prediction_rows))):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    _write_json_sidecar(out, {"config_path": str(config_path), "selection_metric": metric, "ensemble_top_k": top_k, "ensemble_weighting": weighting, "ensemble_temperature": temperature, "n_subjects": len(subjects), "n_candidates": len(candidates), "cue_files_used": False, "target_labels_used_for_selection": False, "random_seed": DEFAULT_RANDOM_SEED})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cue-free BUSH-MEG source-only nested top-k ensemble LOSO decoding.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inner-cv-out", type=Path)
    parser.add_argument("--candidate-summary-out", type=Path)
    parser.add_argument("--predictions-out", type=Path)
    args = parser.parse_args(argv)
    summary = run_bushmeg_source_loso_ensemble(args.config, overrides=args.overrides, out_path=args.out, inner_cv_out_path=args.inner_cv_out, candidate_summary_out_path=args.candidate_summary_out, predictions_out_path=args.predictions_out)
    print(f"Wrote {len(summary)} LOSO rows")
    print(f"Mean balanced accuracy: {float(summary['balanced_accuracy'].mean()):.6f}")
    print(f"Mean top-2/top-3 accuracy: {float(summary['top2_accuracy'].mean()):.6f} / {float(summary['top3_accuracy'].mean()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
