"""Category-2 autoencoder latent-space LOSO decoding for BUSH-MEG.

This workflow implements an unlabeled target-adaptive protocol:

* source epochs and labels are used to train the downstream classifier;
* held-out target epochs are used only as unlabeled examples for fitting a shared
  encoder/decoder representation and feature scaling;
* held-out target labels are used only after prediction for reporting metrics.

In the four-protocol taxonomy, this uses ``X_s``, ``y_s`` and ``X_t`` but not
``y_t`` for fitting/adaptation/model selection, i.e. Protocol 2.  By default the
workflow is transductive because the target test split supplies the unlabeled
``X_t`` reconstruction pool.  For a deployment-style Protocol-2 run, replace
that pool by a separate unlabeled target calibration block before evaluation.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from neureptrace.bushmeg_source_loso import (
    DEFAULT_COVARIANCE_MAX_CHANNELS,
    DEFAULT_RANDOM_SEED,
    FeatureCache,
    SubjectEpochs,
    WindowSpec,
    _config_bool,
    _load_subjects_from_config,
    _resolve_output,
    _section,
    _window_size_seconds,
    normalize_source_feature_kind,
)
from neureptrace.dataset_config import apply_overrides, load_config

SUPPORTED_AUTOENCODERS = {"linear", "linear_pca", "pca", "mlp"}
SUPPORTED_SCALERS = {"standard", "none"}
SUPPORTED_MLP_ACTIVATIONS = {"identity", "logistic", "tanh", "relu"}


@dataclass(frozen=True, slots=True)
class Category2AutoencoderConfig:
    """Hyperparameters for the unlabeled target-adaptive latent representation."""

    windows: tuple[WindowSpec, ...]
    temporal_bins: int
    feature_kind: str
    covariance_max_channels: int
    autoencoder: str
    latent_dim: int
    feature_scaling: str
    classifier_c: float
    classifier_class_weight: str | None
    classifier_max_iter: int
    random_seed: int
    mlp_activation: str
    mlp_alpha: float
    mlp_learning_rate_init: float
    mlp_max_iter: int
    mlp_batch_size: int | str
    mlp_early_stopping: bool
    mlp_validation_fraction: float
    mlp_tol: float


@dataclass(frozen=True, slots=True)
class AutoencoderFoldResult:
    """Latent source/target matrices and reconstruction diagnostics for one fold."""

    z_source: np.ndarray
    z_target: np.ndarray
    reconstruction_mse_all: float
    reconstruction_mse_source: float
    reconstruction_mse_target: float
    effective_latent_dim: int
    n_autoencoder_iterations: int | None


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return normalized


def _positive_float(value: Any, *, name: str, minimum: float = 0.0, inclusive: bool = False) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite floating-point value.") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if inclusive:
        ok = normalized >= minimum
        comparator = ">="
    else:
        ok = normalized > minimum
        comparator = ">"
    if not ok:
        raise ValueError(f"{name} must be {comparator} {minimum}.")
    return normalized


def _list_value(value: Any, default: Sequence[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _float_list(value: Any, default: Sequence[float], *, name: str) -> list[float]:
    values = [_positive_float(item, name=name, minimum=-np.inf, inclusive=True) for item in _list_value(value, default)]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def _normalize_autoencoder(value: Any) -> str:
    normalized = "linear_pca" if value is None else str(value).strip().lower().replace("-", "_")
    if normalized == "pca":
        normalized = "linear_pca"
    if normalized not in SUPPORTED_AUTOENCODERS:
        raise ValueError(f"Unknown autoencoder '{value}'. Supported values: {sorted(SUPPORTED_AUTOENCODERS)}.")
    return normalized


def _normalize_scaler(value: Any) -> str:
    normalized = "standard" if value is None else str(value).strip().lower().replace("-", "_")
    if normalized in {"", "false", "off", "identity"}:
        normalized = "none"
    if normalized not in SUPPORTED_SCALERS:
        raise ValueError(f"Unknown feature_scaling '{value}'. Supported values: {sorted(SUPPORTED_SCALERS)}.")
    return normalized


def _normalize_class_weight(value: Any) -> str | None:
    if value is None:
        return "balanced"
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"", "none", "false", "off", "no", "unweighted"}:
        return None
    if normalized in {"balanced", "class_balanced"}:
        return "balanced"
    raise ValueError("classifier_class_weight must be 'balanced' or 'none'.")


def _normalize_mlp_batch_size(value: Any) -> int | str:
    if value is None:
        return "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return _positive_int(value, name="mlp_batch_size", minimum=1)


def _category2_config(config: Mapping[str, Any]) -> Category2AutoencoderConfig:
    section = _section(config, "category2_autoencoder_loso")
    preprocessing = _section(config, "preprocessing")

    window_width = _positive_float(
        section.get("window_size", section.get("window_width", preprocessing.get("window_size", _window_size_seconds(preprocessing)))),
        name="window_size",
    )
    centers = _float_list(section.get("window_centers", section.get("centers")), [0.184], name="window_centers")
    windows = tuple(WindowSpec(center=float(center), width=window_width) for center in centers)
    if not windows:
        raise ValueError("category2_autoencoder_loso.window_centers must define at least one window.")

    autoencoder = _normalize_autoencoder(section.get("autoencoder", section.get("autoencoder_kind", "linear_pca")))
    mlp_activation = str(section.get("mlp_activation", "relu")).strip().lower()
    if mlp_activation not in SUPPORTED_MLP_ACTIVATIONS:
        raise ValueError(f"Unknown mlp_activation '{mlp_activation}'. Supported values: {sorted(SUPPORTED_MLP_ACTIVATIONS)}.")

    validation_fraction = _positive_float(section.get("mlp_validation_fraction", 0.10), name="mlp_validation_fraction", minimum=0.0)
    if validation_fraction >= 1.0:
        raise ValueError("mlp_validation_fraction must be < 1.")

    return Category2AutoencoderConfig(
        windows=windows,
        temporal_bins=_positive_int(section.get("temporal_bins", 4), name="temporal_bins"),
        feature_kind=normalize_source_feature_kind(section.get("feature_kind", "evoked_dct")),
        covariance_max_channels=_positive_int(section.get("covariance_max_channels", DEFAULT_COVARIANCE_MAX_CHANNELS), name="covariance_max_channels"),
        autoencoder=autoencoder,
        latent_dim=_positive_int(section.get("latent_dim", 64), name="latent_dim"),
        feature_scaling=_normalize_scaler(section.get("feature_scaling", "standard")),
        classifier_c=_positive_float(section.get("classifier_c", 1.0), name="classifier_c"),
        classifier_class_weight=_normalize_class_weight(section.get("classifier_class_weight", "balanced")),
        classifier_max_iter=_positive_int(section.get("classifier_max_iter", _section(config, "decoding").get("max_iter", 2000)), name="classifier_max_iter"),
        random_seed=_positive_int(section.get("random_seed", DEFAULT_RANDOM_SEED), name="random_seed", minimum=0),
        mlp_activation=mlp_activation,
        mlp_alpha=_positive_float(section.get("mlp_alpha", 1e-4), name="mlp_alpha", minimum=0.0, inclusive=True),
        mlp_learning_rate_init=_positive_float(section.get("mlp_learning_rate_init", 1e-3), name="mlp_learning_rate_init"),
        mlp_max_iter=_positive_int(section.get("mlp_max_iter", 200), name="mlp_max_iter"),
        mlp_batch_size=_normalize_mlp_batch_size(section.get("mlp_batch_size", "auto")),
        mlp_early_stopping=_config_bool(section.get("mlp_early_stopping"), default=False),
        mlp_validation_fraction=validation_fraction,
        mlp_tol=_positive_float(section.get("mlp_tol", 1e-4), name="mlp_tol"),
    )


def _subject_feature_matrix(cache: FeatureCache, subject_id: str, cfg: Category2AutoencoderConfig) -> np.ndarray:
    features = [
        cache.get(
            subject_id,
            window,
            cfg.temporal_bins,
            feature_kind=cfg.feature_kind,
            covariance_max_channels=cfg.covariance_max_channels,
        )
        for window in cfg.windows
    ]
    if len(features) == 1:
        return features[0]
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _stack_subject_features(cache: FeatureCache, subject_ids: Sequence[str], cfg: Category2AutoencoderConfig) -> np.ndarray:
    return np.concatenate([_subject_feature_matrix(cache, subject_id, cfg) for subject_id in subject_ids], axis=0)


def _stack_subject_labels(subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([subjects[subject_id].labels for subject_id in subject_ids], axis=0).astype(int, copy=False)


def _scale_features(x_source: np.ndarray, x_target: np.ndarray, cfg: Category2AutoencoderConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pool = np.concatenate([x_source, x_target], axis=0).astype(np.float64, copy=False)
    if cfg.feature_scaling == "none":
        return x_source.astype(np.float64, copy=False), x_target.astype(np.float64, copy=False), pool
    scaler = StandardScaler()
    scaler.fit(pool)
    return scaler.transform(x_source), scaler.transform(x_target), scaler.transform(pool)


def _activation(values: np.ndarray, activation: str) -> np.ndarray:
    if activation == "identity":
        return values
    if activation == "logistic":
        return 1.0 / (1.0 + np.exp(-values))
    if activation == "tanh":
        return np.tanh(values)
    if activation == "relu":
        return np.maximum(values, 0.0)
    raise AssertionError(f"Unhandled activation: {activation}")


def _mlp_latent(model: MLPRegressor, values: np.ndarray) -> np.ndarray:
    hidden = values @ model.coefs_[0] + model.intercepts_[0]
    return _activation(hidden, model.activation).astype(np.float64, copy=False)


def _mean_squared_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _fit_autoencoder_latents(x_source: np.ndarray, x_target: np.ndarray, cfg: Category2AutoencoderConfig) -> AutoencoderFoldResult:
    x_source_scaled, x_target_scaled, x_pool_scaled = _scale_features(x_source, x_target, cfg)
    n_source = x_source_scaled.shape[0]
    effective_latent_dim = min(cfg.latent_dim, x_pool_scaled.shape[0], x_pool_scaled.shape[1])
    if effective_latent_dim < 1:
        raise ValueError("The autoencoder pool must contain at least one sample and one feature.")

    if cfg.autoencoder in {"linear", "linear_pca"}:
        model = PCA(n_components=effective_latent_dim, svd_solver="auto", random_state=cfg.random_seed)
        z_pool = model.fit_transform(x_pool_scaled)
        reconstructed = model.inverse_transform(z_pool)
        n_iter = None
    elif cfg.autoencoder == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(cfg.latent_dim,),
            activation=cfg.mlp_activation,
            solver="adam",
            alpha=cfg.mlp_alpha,
            batch_size=cfg.mlp_batch_size,
            learning_rate_init=cfg.mlp_learning_rate_init,
            max_iter=cfg.mlp_max_iter,
            shuffle=True,
            random_state=cfg.random_seed,
            tol=cfg.mlp_tol,
            early_stopping=cfg.mlp_early_stopping,
            validation_fraction=cfg.mlp_validation_fraction,
        )
        model.fit(x_pool_scaled, x_pool_scaled)
        z_pool = _mlp_latent(model, x_pool_scaled)
        reconstructed = model.predict(x_pool_scaled)
        effective_latent_dim = int(z_pool.shape[1])
        n_iter = int(model.n_iter_)
    else:  # pragma: no cover - guarded by normalization
        raise AssertionError(f"Unhandled autoencoder: {cfg.autoencoder}")

    return AutoencoderFoldResult(
        z_source=z_pool[:n_source].astype(np.float64, copy=False),
        z_target=z_pool[n_source:].astype(np.float64, copy=False),
        reconstruction_mse_all=_mean_squared_error(x_pool_scaled, reconstructed),
        reconstruction_mse_source=_mean_squared_error(x_pool_scaled[:n_source], reconstructed[:n_source]),
        reconstruction_mse_target=_mean_squared_error(x_pool_scaled[n_source:], reconstructed[n_source:]),
        effective_latent_dim=effective_latent_dim,
        n_autoencoder_iterations=n_iter,
    )


def _predict_source_classifier(z_source: np.ndarray, y_source: np.ndarray, z_target: np.ndarray, cfg: Category2AutoencoderConfig, *, n_classes: int) -> np.ndarray:
    classifier = LogisticRegression(C=cfg.classifier_c, class_weight=cfg.classifier_class_weight, max_iter=cfg.classifier_max_iter, solver="lbfgs")
    classifier.fit(z_source, y_source)
    raw_probabilities = classifier.predict_proba(z_target)
    probabilities = np.zeros((z_target.shape[0], n_classes), dtype=np.float64)
    probabilities[:, classifier.classes_.astype(int)] = raw_probabilities
    row_sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / np.maximum(row_sums, 1e-12)


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    if probabilities.size == 0:
        return float("nan")
    k = min(int(k), probabilities.shape[1])
    top = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == labels.reshape(-1, 1), axis=1)))


def _classification_metrics(probabilities: np.ndarray, labels: np.ndarray, *, n_classes: int) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "top2_accuracy": _top_k_accuracy(probabilities, labels, k=2),
        "top3_accuracy": _top_k_accuracy(probabilities, labels, k=3),
    }
    try:
        metrics["log_loss"] = float(log_loss(labels, probabilities, labels=np.arange(n_classes)))
    except ValueError:
        metrics["log_loss"] = float("nan")
    return metrics


def _write_json_sidecar(path: Path, payload: Mapping[str, Any]) -> None:
    sidecar = Path(str(path) + ".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_bushmeg_category2_autoencoder_loso(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    predictions_out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run Protocol-2 unlabeled target-adaptive autoencoder LOSO decoding."""

    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    cfg = _category2_config(config)
    subjects, encoder = _load_subjects_from_config(config, config_dir=config_path.parent)
    cache = FeatureCache(subjects)
    n_classes = len(encoder.classes_)

    out = Path(out_path) if out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="category2_autoencoder_loso_summary_csv",
        default="category2_autoencoder_loso_summary.csv",
    )
    predictions_out = Path(predictions_out_path) if predictions_out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="category2_autoencoder_loso_predictions_csv",
        default="category2_autoencoder_loso_predictions.csv",
    )

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outer_test_subject in sorted(subjects):
        source_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
        x_source = _stack_subject_features(cache, source_subjects, cfg)
        y_source = _stack_subject_labels(subjects, source_subjects)
        x_target = _subject_feature_matrix(cache, outer_test_subject, cfg)

        # Protocol-2 hygiene boundary: the target labels are intentionally not
        # read until after all fitting/adaptation and classifier inference.
        latent = _fit_autoencoder_latents(x_source, x_target, cfg)
        probabilities = _predict_source_classifier(latent.z_source, y_source, latent.z_target, cfg, n_classes=n_classes)

        target_labels_for_metrics = subjects[outer_test_subject].labels.astype(int, copy=False)
        predictions = probabilities.argmax(axis=1)
        metrics = _classification_metrics(probabilities, target_labels_for_metrics, n_classes=n_classes)
        window_centers = tuple(float(window.center) for window in cfg.windows)
        window_widths = tuple(float(window.width) for window in cfg.windows)
        summary_rows.append(
            {
                "outer_test_subject": outer_test_subject,
                "protocol": "category2_unlabeled_target_adaptive",
                "transductive_target_test_x_used_for_autoencoder": True,
                "source_data_used_for_autoencoder": True,
                "source_labels_used_for_classifier": True,
                "target_data_used_for_autoencoder": True,
                "target_labels_used_for_autoencoder": False,
                "target_labels_used_for_classifier": False,
                "target_labels_used_for_model_selection": False,
                "autoencoder": cfg.autoencoder,
                "latent_dim": cfg.latent_dim,
                "effective_latent_dim": latent.effective_latent_dim,
                "feature_scaling": cfg.feature_scaling,
                "feature_kind": cfg.feature_kind,
                "temporal_bins": cfg.temporal_bins,
                "window_centers": "|".join(f"{value:.6g}" for value in window_centers),
                "window_widths": "|".join(f"{value:.6g}" for value in window_widths),
                "covariance_max_channels": cfg.covariance_max_channels,
                "classifier": "logistic_regression",
                "classifier_c": cfg.classifier_c,
                "classifier_class_weight": "" if cfg.classifier_class_weight is None else cfg.classifier_class_weight,
                "n_autoencoder_iterations": "" if latent.n_autoencoder_iterations is None else latent.n_autoencoder_iterations,
                "reconstruction_mse_all": latent.reconstruction_mse_all,
                "reconstruction_mse_source": latent.reconstruction_mse_source,
                "reconstruction_mse_target": latent.reconstruction_mse_target,
                "n_train_subjects": len(source_subjects),
                "n_source_trials": int(x_source.shape[0]),
                "n_target_trials": int(x_target.shape[0]),
                "n_features": int(x_source.shape[1]),
                "n_classes": n_classes,
                "class_names": "|".join(map(str, encoder.classes_)),
                **metrics,
            }
        )

        metadata = subjects[outer_test_subject].metadata.reset_index(drop=True)
        for row_idx, (true_label, predicted_label) in enumerate(zip(target_labels_for_metrics, predictions, strict=True)):
            row: dict[str, Any] = {
                "outer_test_subject": outer_test_subject,
                "trial_index": int(row_idx),
                "protocol": "category2_unlabeled_target_adaptive",
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
    predictions = pd.DataFrame(prediction_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    predictions.to_csv(predictions_out, index=False)
    _write_json_sidecar(
        out,
        {
            "config_path": str(config_path),
            "protocol": "category2_unlabeled_target_adaptive",
            "uses_source_data": True,
            "uses_source_labels": True,
            "uses_target_data": True,
            "uses_target_labels_for_fitting_or_selection": False,
            "transductive_target_test_x_used_for_autoencoder": True,
            "note": "Target labels are used only after prediction for metric computation.",
            "n_subjects": len(subjects),
            "n_classes": n_classes,
            "class_names": list(map(str, encoder.classes_)),
            "autoencoder_config": {
                "autoencoder": cfg.autoencoder,
                "latent_dim": cfg.latent_dim,
                "feature_scaling": cfg.feature_scaling,
                "feature_kind": cfg.feature_kind,
                "temporal_bins": cfg.temporal_bins,
                "window_centers": [float(window.center) for window in cfg.windows],
                "window_widths": [float(window.width) for window in cfg.windows],
                "classifier_c": cfg.classifier_c,
                "classifier_class_weight": cfg.classifier_class_weight,
                "random_seed": cfg.random_seed,
            },
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Category-2 unlabeled target-adaptive autoencoder latent-space LOSO decoding for BUSH-MEG.")
    parser.add_argument("config", type=Path, help="Dataset/workflow config, for example configs/bush_meg/category2_autoencoder_loso.yml.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a dotted config key, e.g. --set category2_autoencoder_loso.latent_dim=32.")
    parser.add_argument("--out", type=Path, help="Summary CSV path. Defaults to outputs.category2_autoencoder_loso_summary_csv.")
    parser.add_argument("--predictions-out", type=Path, help="Held-out trial probability CSV path.")
    args = parser.parse_args(argv)

    summary = run_bushmeg_category2_autoencoder_loso(
        args.config,
        overrides=args.overrides,
        out_path=args.out,
        predictions_out_path=args.predictions_out,
    )
    mean_balanced = float(summary["balanced_accuracy"].mean())
    mean_top2 = float(summary["top2_accuracy"].mean())
    mean_top3 = float(summary["top3_accuracy"].mean())
    print(f"Wrote {len(summary)} Category-2 LOSO rows")
    print(f"Mean balanced accuracy: {mean_balanced:.6f}")
    print(f"Mean top-2/top-3 accuracy: {mean_top2:.6f} / {mean_top3:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
