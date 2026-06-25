"""Honor Category-2 autoencoder fold limits in direct and all-protocol runs."""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd

_RUNNER_MODULE = "neureptrace.bushmeg_category2_autoencoder_loso"
_ALL_PROTOCOLS_PATCH_MODULE = "neureptrace._category2_autoencoder_all_protocols_patch"
_RUNNER_MARKER = "_neureptrace_category2_autoencoder_max_folds_patch_installed"
_ALL_PROTOCOLS_MARKER = "_neureptrace_category2_autoencoder_max_folds_all_protocols_patch_installed"


def _resolve_max_folds(module: Any, config: Mapping[str, Any], explicit_max_folds: int | None) -> int | None:
    raw_value: Any = explicit_max_folds
    if raw_value is None:
        section = config.get("category2_autoencoder_loso", {}) or {}
        if isinstance(section, Mapping):
            raw_value = section.get("max_folds", section.get("fold_limit"))
    if raw_value is None:
        return None
    return module._positive_int(raw_value, name="max_folds")


def _resolve_summary_path(module: Any, config_path: Path, config: Mapping[str, Any], out_path: str | Path | None) -> Path:
    if out_path is not None:
        return Path(out_path)
    return module._resolve_output(
        config,
        config_dir=config_path.parent,
        key="category2_autoencoder_loso_summary_csv",
        default="category2_autoencoder_loso_summary.csv",
    )


def _resolve_predictions_path(
    module: Any,
    config_path: Path,
    config: Mapping[str, Any],
    predictions_out_path: str | Path | None,
) -> Path:
    if predictions_out_path is not None:
        return Path(predictions_out_path)
    return module._resolve_output(
        config,
        config_dir=config_path.parent,
        key="category2_autoencoder_loso_predictions_csv",
        default="category2_autoencoder_loso_predictions.csv",
    )


def _patch_runner_module(module: Any) -> None:
    if getattr(module, _RUNNER_MARKER, False):
        return

    original_run = module.run_bushmeg_category2_autoencoder_loso

    @wraps(original_run)
    def run_bushmeg_category2_autoencoder_loso(
        config_path: str | Path,
        *,
        overrides: Sequence[str] | None = None,
        out_path: str | Path | None = None,
        predictions_out_path: str | Path | None = None,
        max_folds: int | None = None,
    ) -> pd.DataFrame:
        config_path = Path(config_path)
        config = module.apply_overrides(module.load_config(config_path), overrides)
        cfg = module._category2_config(config)
        subjects, encoder = module._load_subjects_from_config(config, config_dir=config_path.parent)
        cache = module.FeatureCache(subjects)
        n_classes = len(encoder.classes_)

        out = _resolve_summary_path(module, config_path, config, out_path)
        predictions_out = _resolve_predictions_path(module, config_path, config, predictions_out_path)
        resolved_max_folds = _resolve_max_folds(module, config, max_folds)
        all_subjects = sorted(subjects)
        outer_subjects = all_subjects if resolved_max_folds is None else all_subjects[:resolved_max_folds]

        summary_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        for outer_test_subject in outer_subjects:
            source_subjects = [subject for subject in all_subjects if subject != outer_test_subject]
            x_source = module._stack_subject_features(cache, source_subjects, cfg)
            y_source = module._stack_subject_labels(subjects, source_subjects)
            x_target = module._subject_feature_matrix(cache, outer_test_subject, cfg)

            # Protocol-2 hygiene boundary: the target labels are intentionally not
            # read until after all fitting/adaptation and classifier inference.
            latent = module._fit_autoencoder_latents(x_source, x_target, cfg)
            probabilities = module._predict_source_classifier(latent.z_source, y_source, latent.z_target, cfg, n_classes=n_classes)

            target_labels_for_metrics = subjects[outer_test_subject].labels.astype(int, copy=False)
            predictions = probabilities.argmax(axis=1)
            metrics = module._classification_metrics(probabilities, target_labels_for_metrics, n_classes=n_classes)
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
                    "max_folds": "" if resolved_max_folds is None else int(resolved_max_folds),
                    "n_outer_folds": len(outer_subjects),
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
                    "confidence": float(module.np.max(probabilities[row_idx])),
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
        module._write_json_sidecar(
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
                "n_outer_folds": len(outer_subjects),
                "max_folds": resolved_max_folds,
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

    module.run_bushmeg_category2_autoencoder_loso = run_bushmeg_category2_autoencoder_loso
    setattr(module, _RUNNER_MARKER, True)


def _with_category2_max_folds(config: Mapping[str, Any], max_folds: int | None) -> Mapping[str, Any]:
    if max_folds is None:
        return config
    updated = copy.deepcopy(dict(config))
    section = updated.setdefault("category2_autoencoder_loso", {})
    if not isinstance(section, dict):
        raise ValueError("Config section 'category2_autoencoder_loso' must be a mapping.")
    section["max_folds"] = int(max_folds)
    return updated


def _patch_all_protocols_patch_module(module: Any) -> None:
    if getattr(module, _ALL_PROTOCOLS_MARKER, False):
        return

    original_run_category2_loso = module._run_category2_loso

    @wraps(original_run_category2_loso)
    def _run_category2_loso(
        allp: Any,
        spec: Any,
        *,
        config: Mapping[str, Any],
        all_protocols_config: Mapping[str, Any],
        method_dir: str | Path,
        data_dir: Any,
        participants: Any,
        max_folds: int | None,
        resume: bool,
        include_heavy: bool,
        aggregate_callback: Any = None,
        method_timeout_seconds: float | None = None,
        fold_timeout_seconds: float | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        _patch_runner_module(importlib.import_module(_RUNNER_MODULE))
        config = _with_category2_max_folds(config, max_folds)
        return original_run_category2_loso(
            allp,
            spec,
            config=config,
            all_protocols_config=all_protocols_config,
            method_dir=method_dir,
            data_dir=data_dir,
            participants=participants,
            max_folds=max_folds,
            resume=resume,
            include_heavy=include_heavy,
            aggregate_callback=aggregate_callback,
            method_timeout_seconds=method_timeout_seconds,
            fold_timeout_seconds=fold_timeout_seconds,
        )

    module._run_category2_loso = _run_category2_loso
    setattr(module, _ALL_PROTOCOLS_MARKER, True)


def install() -> None:
    """Install Category-2 autoencoder fold-limit support."""

    _patch_all_protocols_patch_module(importlib.import_module(_ALL_PROTOCOLS_PATCH_MODULE))
    _patch_runner_module(importlib.import_module(_RUNNER_MODULE))
