"""Strict BUSH-MEG source-only LOSO decoder with supervised low-rank temporal features.

This workflow is intended as a stronger cue-free baseline for BUSH-MEG.  It
keeps the existing leave-one-subject-out discipline, but replaces the short
PCA-window representation by wider binned epoch features followed by a
fold-local supervised PLS projection.  The PLS projection is fitted only on the
training subjects of each inner/outer fold, so held-out subjects and cue trials
cannot influence the low-rank subspace.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from neureptrace.bushmeg_source_loso import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SELECTION_METRIC,
    MINIMIZE_SELECTION_METRICS,
    SUPPORTED_SELECTION_METRICS,
    SubjectEpochs,
    _candidate_metrics,
    _load_subjects_from_config,
    _resolve_output,
    _section,
    _write_json_sidecar,
)
from neureptrace.dataset_config import apply_overrides, load_config
from neureptrace.decoding import normalize_decoder_name, parse_c_grid, score_to_probabilities
from neureptrace.mne_time_decode import _align_probability_columns


DEFAULT_PLS_COMPONENTS = 32


@dataclass(frozen=True, slots=True)
class EpochWindow:
    """One absolute-time epoch window used for binned temporal features."""

    name: str
    start: float
    stop: float


@dataclass(frozen=True, slots=True)
class LowRankCandidateSpec:
    """One supervised low-rank source-only decoder candidate."""

    name: str
    decoder: str
    classifier_param: float | None
    window: EpochWindow
    temporal_bins: int
    pls_components: int
    include_deltas: bool = False


class SupervisedPLSTransformer(TransformerMixin, BaseEstimator):
    """Fold-local supervised PLS projection for multiclass epoch features.

    The transformer converts labels to a one-hot matrix and fits
    :class:`~sklearn.cross_decomposition.PLSRegression` on ``X`` and that
    one-hot target.  This is the usual PLS-DA bottleneck, but exposed as a
    scikit-learn transformer so it can sit inside a leakage-safe pipeline.

    ``n_components`` is clipped at fit time to the feasible rank implied by the
    current training fold.  This makes LOSO folds robust when a requested grid
    value is larger than the number of training trials or input features.
    """

    def __init__(self, n_components: int = DEFAULT_PLS_COMPONENTS, *, scale: bool = False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, features: np.ndarray, labels: Sequence[Any]):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError("SupervisedPLSTransformer expects a two-dimensional feature matrix.")
        labels = np.asarray(labels)
        classes, encoded = np.unique(labels, return_inverse=True)
        if classes.size < 2:
            raise ValueError("SupervisedPLSTransformer needs at least two classes.")
        requested = _normalize_pls_components(self.n_components)
        feasible = min(requested, features.shape[0] - 1, features.shape[1])
        if feasible < 1:
            raise ValueError(
                "Supervised PLS needs at least two training examples and one input feature; "
                f"got shape {features.shape}."
            )
        targets = np.eye(classes.size, dtype=float)[encoded]
        self.classes_ = classes
        self.n_components_ = int(feasible)
        self.pls_ = PLSRegression(n_components=self.n_components_, scale=bool(self.scale))
        self.pls_.fit(features, targets)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self, "pls_"):
            raise RuntimeError("SupervisedPLSTransformer must be fitted before transform().")
        projected = self.pls_.transform(np.asarray(features, dtype=float))
        return np.asarray(projected, dtype=np.float32)


class LowRankFeatureCache:
    """Per-subject cache for binned wide-epoch features."""

    def __init__(self, subjects: Mapping[str, SubjectEpochs]):
        self._subjects = dict(subjects)
        self._cache: dict[tuple[str, EpochWindow, int, bool], np.ndarray] = {}

    def get(self, subject: str, candidate: LowRankCandidateSpec) -> np.ndarray:
        key = (subject, candidate.window, int(candidate.temporal_bins), bool(candidate.include_deltas))
        if key not in self._cache:
            subject_epochs = self._subjects[subject]
            self._cache[key] = _epoch_bin_mean_features(
                subject_epochs.data,
                subject_epochs.times,
                candidate.window,
                temporal_bins=candidate.temporal_bins,
                include_deltas=candidate.include_deltas,
            )
        return self._cache[key]


def _normalize_pls_components(value: Any) -> int:
    if value is None:
        return DEFAULT_PLS_COMPONENTS
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"auto", "default"}:
            return DEFAULT_PLS_COMPONENTS
        value = float(stripped) if any(marker in stripped for marker in (".", "e", "E")) else int(stripped)
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError("pls_components must be a positive integer.")
    if not float(value).is_integer() or int(value) < 1:
        raise ValueError("pls_components must be a positive integer.")
    return int(value)


def _as_list(value: Any, default: Sequence[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _epoch_window_from_mapping(item: Mapping[str, Any], *, index: int) -> EpochWindow:
    if not isinstance(item, Mapping):
        raise ValueError("epoch_windows entries must be mappings.")
    if "range" in item:
        raw_range = item["range"]
        if not isinstance(raw_range, Sequence) or isinstance(raw_range, (str, bytes)) or len(raw_range) != 2:
            raise ValueError("epoch window range must contain exactly two values.")
        start, stop = map(float, raw_range)
    else:
        start = float(item.get("start", item.get("tmin")))
        stop = float(item.get("stop", item.get("tmax")))
    if stop <= start:
        raise ValueError("epoch window stop must be greater than start.")
    name = str(item.get("name", f"window_{index:02d}_{start:g}_{stop:g}"))
    return EpochWindow(name=name, start=start, stop=stop)


def _epoch_bin_mean_features(
    data: np.ndarray,
    times: np.ndarray,
    window: EpochWindow,
    *,
    temporal_bins: int,
    include_deltas: bool,
) -> np.ndarray:
    if temporal_bins < 1:
        raise ValueError("temporal_bins must be at least one.")
    tolerance = 1e-12
    indices = np.flatnonzero((times >= window.start - tolerance) & (times <= window.stop + tolerance))
    if indices.size == 0:
        raise ValueError(
            f"Epoch window '{window.name}' [{window.start:.6g}, {window.stop:.6g}] "
            f"does not overlap available times [{times[0]:.6g}, {times[-1]:.6g}]."
        )
    if indices.size < temporal_bins:
        raise ValueError(
            f"Epoch window '{window.name}' has only {indices.size} samples, "
            f"not enough for {temporal_bins} bins."
        )
    bins = np.array_split(indices, int(temporal_bins))
    means = [data[:, :, bin_indices].mean(axis=2) for bin_indices in bins]
    features = [*means]
    if include_deltas and len(means) > 1:
        features.extend(means[index + 1] - means[index] for index in range(len(means) - 1))
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _stack_features(cache: LowRankFeatureCache, subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str], candidate: LowRankCandidateSpec) -> np.ndarray:
    return np.concatenate([cache.get(subject_id, candidate) for subject_id in subject_ids], axis=0)


def _stack_labels(subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([subjects[subject_id].labels for subject_id in subject_ids], axis=0)


def _make_model(candidate: LowRankCandidateSpec, *, max_iter: int):
    decoder = normalize_decoder_name(candidate.decoder)
    classifier_param = 1.0 if candidate.classifier_param is None else float(candidate.classifier_param)
    steps: list[Any] = [
        StandardScaler(),
        SupervisedPLSTransformer(n_components=candidate.pls_components),
        StandardScaler(),
    ]
    if decoder in {"logistic", "multinomial-logistic"}:
        classifier = LogisticRegression(
            class_weight="balanced",
            C=classifier_param,
            max_iter=max_iter,
            random_state=DEFAULT_RANDOM_SEED,
            solver="lbfgs",
        )
    elif decoder == "linear_svm":
        classifier = LinearSVC(
            class_weight="balanced",
            C=classifier_param,
            max_iter=max_iter,
            random_state=DEFAULT_RANDOM_SEED,
        )
    elif decoder == "ridge":
        classifier = RidgeClassifier(class_weight="balanced", alpha=classifier_param, max_iter=max_iter)
    elif decoder == "lda":
        classifier = LinearDiscriminantAnalysis(solver="svd")
    elif decoder == "shrinkage_lda":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(
            "supervised-lowrank-loso supports logistic/multinomial-logistic, "
            "linear_svm, ridge, lda, and shrinkage_lda decoders."
        )
    return make_pipeline(*steps, classifier)


def _model_probabilities(model, features: np.ndarray, *, n_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
    elif hasattr(model, "decision_function"):
        probabilities = score_to_probabilities(model.decision_function(features))
    else:
        raise ValueError("Low-rank decoder does not provide predict_proba or decision_function.")
    return _align_probability_columns(probabilities, model=model, classes=np.arange(n_classes))


def _predict_candidate(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: LowRankFeatureCache,
    candidate: LowRankCandidateSpec,
    train_subjects: Sequence[str],
    test_subject: str,
    n_classes: int,
    max_iter: int,
) -> np.ndarray:
    train_features = _stack_features(cache, subjects, train_subjects, candidate)
    train_labels = _stack_labels(subjects, train_subjects)
    test_features = cache.get(test_subject, candidate)
    model = _make_model(candidate, max_iter=max_iter)
    model.fit(train_features, train_labels)
    return _model_probabilities(model, test_features, n_classes=n_classes)


def _combine_probabilities(probability_list: Sequence[np.ndarray], *, mode: str, min_probability: float = 1e-12) -> np.ndarray:
    if not probability_list:
        raise ValueError("At least one candidate probability matrix is required.")
    stack = np.stack(probability_list, axis=0)
    normalized_mode = mode.lower().replace("-", "_")
    if normalized_mode in {"log", "log_mean", "geometric", "geometric_mean"}:
        log_probabilities = np.log(np.clip(stack, float(min_probability), 1.0)).mean(axis=0)
        log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
        combined = np.exp(np.clip(log_probabilities, -745.0, 0.0))
    elif normalized_mode in {"mean", "arithmetic", "arithmetic_mean"}:
        combined = stack.mean(axis=0)
    else:
        raise ValueError("ensemble_aggregation must be 'mean' or 'log_mean'.")
    row_sums = combined.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Combined probabilities must have positive row sums.")
    return combined / row_sums


def _candidate_rowspec(candidate: LowRankCandidateSpec) -> dict[str, Any]:
    return {
        "candidate": candidate.name,
        "decoder": normalize_decoder_name(candidate.decoder),
        "classifier_param": "" if candidate.classifier_param is None else candidate.classifier_param,
        "epoch_window": candidate.window.name,
        "window_start": candidate.window.start,
        "window_stop": candidate.window.stop,
        "temporal_bins": candidate.temporal_bins,
        "pls_components": candidate.pls_components,
        "include_deltas": bool(candidate.include_deltas),
        "feature_preprocessor": "supervised_pls",
    }


def _inner_loso_scores(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: LowRankFeatureCache,
    candidate: LowRankCandidateSpec,
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
) -> list[dict[str, Any]]:
    source_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
    rows: list[dict[str, Any]] = []
    for inner_test_subject in source_subjects:
        train_subjects = [subject for subject in source_subjects if subject != inner_test_subject]
        probabilities = _predict_candidate(
            subjects=subjects,
            cache=cache,
            candidate=candidate,
            train_subjects=train_subjects,
            test_subject=inner_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
        )
        labels = subjects[inner_test_subject].labels
        rows.append(
            {
                "outer_test_subject": outer_test_subject,
                "inner_test_subject": inner_test_subject,
                **_candidate_rowspec(candidate),
                **_candidate_metrics(probabilities, labels, n_classes=n_classes),
                "n_train_subjects": len(train_subjects),
                "n_test_trials": len(labels),
            }
        )
    return rows


def _select_candidates(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: LowRankFeatureCache,
    candidates: Sequence[LowRankCandidateSpec],
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    selection_metric: str,
    ensemble_size: int,
) -> tuple[list[LowRankCandidateSpec], list[dict[str, Any]], dict[str, Any]]:
    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unknown selection metric '{selection_metric}'. Available metrics: {sorted(SUPPORTED_SELECTION_METRICS)}.")
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be at least one.")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = _inner_loso_scores(
            subjects=subjects,
            cache=cache,
            candidate=candidate,
            outer_test_subject=outer_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
        )
        all_rows.extend(rows)
        frame = pd.DataFrame(rows)
        summaries.append(
            {
                "candidate": candidate,
                "score": float(frame[selection_metric].mean()),
                "std": float(frame[selection_metric].std(ddof=0)),
                "n_folds": len(frame),
            }
        )

    minimize = selection_metric in MINIMIZE_SELECTION_METRICS
    ordered = sorted(summaries, key=lambda item: (item["score"], item["candidate"].name), reverse=not minimize)
    selected_items = ordered[: min(int(ensemble_size), len(ordered))]
    selected = [item["candidate"] for item in selected_items]
    summary = {
        "inner_selection_metric": selection_metric,
        "inner_best_score": float(ordered[0]["score"]),
        "inner_worst_selected_score": float(selected_items[-1]["score"]),
        "inner_selected_mean_score": float(np.mean([item["score"] for item in selected_items])),
        "inner_selected_std_score": float(np.std([item["score"] for item in selected_items], ddof=0)),
        "inner_n_folds": int(selected_items[0]["n_folds"]),
        "ensemble_size": len(selected),
        "selected_candidates": "|".join(candidate.name for candidate in selected),
        "selected_candidate_scores": "|".join(f"{item['score']:.8g}" for item in selected_items),
    }
    return selected, all_rows, summary


def _candidate_grid(config: Mapping[str, Any]) -> list[LowRankCandidateSpec]:
    lowrank = _section(config, "supervised_lowrank_loso")
    grid = lowrank.get("candidate_grid", {}) or {}
    if not isinstance(grid, Mapping):
        raise ValueError("supervised_lowrank_loso.candidate_grid must be a mapping.")

    raw_windows = grid.get("epoch_windows") or [
        {"name": "post_000_250ms", "start": 0.000, "stop": 0.250},
        {"name": "post_m050_250ms", "start": -0.050, "stop": 0.250},
        {"name": "post_000_350ms", "start": 0.000, "stop": 0.350},
    ]
    windows = [_epoch_window_from_mapping(item, index=index) for index, item in enumerate(raw_windows)]

    decoders = [normalize_decoder_name(str(value)) for value in _as_list(grid.get("decoders"), ["multinomial-logistic"])]
    bins_values = [int(value) for value in _as_list(grid.get("temporal_bins"), [12, 20])]
    component_values = [_normalize_pls_components(value) for value in _as_list(grid.get("pls_components", grid.get("pca_components")), [16, 32])]
    c_grid = [float(value) for value in parse_c_grid(grid.get("c_grid", [0.1, 1.0, 10.0]))]
    delta_values = [_as_bool(value) for value in _as_list(grid.get("include_deltas"), [False])]

    candidates: list[LowRankCandidateSpec] = []
    for window in windows:
        for temporal_bins in bins_values:
            for pls_components in component_values:
                for decoder in decoders:
                    for classifier_param in c_grid:
                        for include_deltas in delta_values:
                            name = "__".join(
                                [
                                    window.name,
                                    f"bins{temporal_bins}",
                                    f"pls{pls_components}",
                                    normalize_decoder_name(decoder),
                                    f"c{classifier_param:g}",
                                    "delta" if include_deltas else "level",
                                ]
                            )
                            candidates.append(
                                LowRankCandidateSpec(
                                    name=name,
                                    decoder=decoder,
                                    classifier_param=classifier_param,
                                    window=window,
                                    temporal_bins=temporal_bins,
                                    pls_components=pls_components,
                                    include_deltas=include_deltas,
                                )
                            )
    if not candidates:
        raise ValueError("No supervised low-rank candidates were configured.")
    return candidates


def run_supervised_lowrank_loso_subjects(
    subjects: Mapping[str, SubjectEpochs],
    *,
    candidates: Sequence[LowRankCandidateSpec],
    class_names: Sequence[Any],
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    ensemble_size: int = 3,
    max_iter: int = 3000,
    ensemble_aggregation: str = "log_mean",
    min_probability: float = 1e-12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run nested source-only LOSO on already-loaded subject epochs."""

    if len(subjects) < 3:
        raise ValueError("Need at least three subjects for nested source-only LOSO.")
    cache = LowRankFeatureCache(subjects)
    n_classes = len(class_names)
    class_names = list(class_names)

    summary_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outer_test_subject in sorted(subjects):
        selected, candidate_inner_rows, selected_summary = _select_candidates(
            subjects=subjects,
            cache=cache,
            candidates=candidates,
            outer_test_subject=outer_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
            selection_metric=selection_metric,
            ensemble_size=ensemble_size,
        )
        inner_rows.extend(candidate_inner_rows)
        train_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
        candidate_probabilities = [
            _predict_candidate(
                subjects=subjects,
                cache=cache,
                candidate=candidate,
                train_subjects=train_subjects,
                test_subject=outer_test_subject,
                n_classes=n_classes,
                max_iter=max_iter,
            )
            for candidate in selected
        ]
        probabilities = _combine_probabilities(candidate_probabilities, mode=ensemble_aggregation, min_probability=min_probability)
        labels = subjects[outer_test_subject].labels
        predictions = probabilities.argmax(axis=1)
        summary_rows.append(
            {
                "outer_test_subject": outer_test_subject,
                **selected_summary,
                **_candidate_metrics(probabilities, labels, n_classes=n_classes),
                "n_train_subjects": len(train_subjects),
                "n_test_trials": len(labels),
                "n_classes": n_classes,
                "class_names": "|".join(map(str, class_names)),
                "feature_preprocessor": "supervised_pls",
                "ensemble_aggregation": ensemble_aggregation,
            }
        )

        metadata = subjects[outer_test_subject].metadata.reset_index(drop=True)
        for row_idx, (true_label, predicted_label) in enumerate(zip(labels, predictions, strict=True)):
            row: dict[str, Any] = {
                "outer_test_subject": outer_test_subject,
                "trial_index": int(row_idx),
                "selected_candidates": selected_summary["selected_candidates"],
                "true_label": int(true_label),
                "true_class": str(class_names[true_label]),
                "predicted_label": int(predicted_label),
                "predicted_class": str(class_names[predicted_label]),
                "probability_true_class": float(probabilities[row_idx, true_label]),
                "confidence": float(np.max(probabilities[row_idx])),
                "is_correct": bool(predicted_label == true_label),
            }
            for column in ("participant", "condition", "stimulus_class"):
                if column in metadata.columns:
                    row[column] = metadata.loc[row_idx, column]
            for class_idx, class_name in enumerate(class_names):
                row[f"class_{class_idx}"] = str(class_name)
                row[f"prob_class_{class_idx}"] = float(probabilities[row_idx, class_idx])
            prediction_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(inner_rows), pd.DataFrame(prediction_rows)


def run_supervised_lowrank_loso(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    inner_cv_out_path: str | Path | None = None,
    predictions_out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run strict cue-free BUSH-MEG supervised-lowrank nested LOSO decoding."""

    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    lowrank = _section(config, "supervised_lowrank_loso")
    selection_metric = str(lowrank.get("selection_metric", DEFAULT_SELECTION_METRIC))
    ensemble_size = int(lowrank.get("ensemble_size", 3))
    max_iter = int((_section(config, "decoding") or {}).get("max_iter", lowrank.get("max_iter", 3000)))
    ensemble_aggregation = str(lowrank.get("ensemble_aggregation", "log_mean"))
    min_probability = float(lowrank.get("min_probability", 1e-12))

    subjects, encoder = _load_subjects_from_config(config, config_dir=config_path.parent)
    candidates = _candidate_grid(config)

    summary, inner, predictions = run_supervised_lowrank_loso_subjects(
        subjects,
        candidates=candidates,
        class_names=encoder.classes_,
        selection_metric=selection_metric,
        ensemble_size=ensemble_size,
        max_iter=max_iter,
        ensemble_aggregation=ensemble_aggregation,
        min_probability=min_probability,
    )

    out = Path(out_path) if out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="supervised_lowrank_loso_summary_csv",
        default="supervised_lowrank_loso_summary.csv",
    )
    inner_out = Path(inner_cv_out_path) if inner_cv_out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="supervised_lowrank_loso_inner_cv_csv",
        default="supervised_lowrank_loso_inner_cv.csv",
    )
    predictions_out = Path(predictions_out_path) if predictions_out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="supervised_lowrank_loso_predictions_csv",
        default="supervised_lowrank_loso_predictions.csv",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    inner_out.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    inner.to_csv(inner_out, index=False)
    predictions.to_csv(predictions_out, index=False)
    _write_json_sidecar(
        out,
        {
            "config_path": str(config_path),
            "selection_metric": selection_metric,
            "n_subjects": len(subjects),
            "n_candidates": len(candidates),
            "ensemble_size": ensemble_size,
            "ensemble_aggregation": ensemble_aggregation,
            "feature_preprocessor": "supervised_pls",
            "normalization_scope": "subject_unlabeled_baseline",
            "cue_files_used": False,
            "random_seed": DEFAULT_RANDOM_SEED,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run strict cue-free supervised-lowrank nested LOSO decoding for BUSH-MEG main-task FieldTrip MAT files."
    )
    parser.add_argument("config", type=Path, help="Dataset/workflow config, for example configs/bush_meg/supervised_lowrank_loso.yml.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a dotted config key.")
    parser.add_argument("--out", type=Path, help="Summary CSV path.")
    parser.add_argument("--inner-cv-out", type=Path, help="Inner LOSO candidate-score CSV path.")
    parser.add_argument("--predictions-out", type=Path, help="Held-out trial probability CSV path.")
    args = parser.parse_args(argv)

    summary = run_supervised_lowrank_loso(
        args.config,
        overrides=args.overrides,
        out_path=args.out,
        inner_cv_out_path=args.inner_cv_out,
        predictions_out_path=args.predictions_out,
    )
    print(f"Wrote {len(summary)} supervised-lowrank LOSO rows")
    print(f"Mean balanced accuracy: {float(summary['balanced_accuracy'].mean()):.6f}")
    print(f"Mean top-2/top-3 accuracy: {float(summary['top2_accuracy'].mean()):.6f} / {float(summary['top3_accuracy'].mean()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
