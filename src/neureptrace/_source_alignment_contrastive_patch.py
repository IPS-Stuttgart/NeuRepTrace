"""Runtime integration for contrastive source-subject alignment.

The core source-alignment module is intentionally stable and guarded by tests.
This patch registers ``method="contrastive"`` as a class/stimulus-anchored
alignment method and routes it through the existing LOSO metadata and guardrail
surface.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Hashable, Mapping, Sequence
from types import ModuleType
from typing import Any

import numpy as np

from neureptrace.decoding.contrastive_alignment import (
    CONTRASTIVE_ALIGNMENT_METHOD,
    fit_contrastive_alignment,
    fit_projection_to_contrastive_template,
    transform_with_contrastive_projection,
)

_TARGET_MODULE = "neureptrace.decoding.source_alignment"
_PATCH_MARKER = "_neureptrace_source_alignment_contrastive_patch_installed"
_FINDER_MARKER = "_neureptrace_source_alignment_contrastive_finder"
_CONTRASTIVE_ALIASES = {
    "contrastive",
    "contrastive_alignment",
    "contrastive_subject_alignment",
    "subject_contrastive_alignment",
    "supervised_contrastive",
    "supervised_contrastive_alignment",
}


def _apply_existing_source_alignment_patches(source_alignment: ModuleType) -> None:
    """Apply older source-alignment patches before wrapping the final functions."""

    for module_name in (
        "neureptrace._source_alignment_anchor_patch",
        "neureptrace._source_alignment_pseudo_calibration_patch",
        "neureptrace._mne_alignment_calibration_anchor_patch",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        patch = getattr(module, "_patch_source_alignment", None)
        if callable(patch):
            patch(source_alignment)


def _patch_source_alignment(source_alignment: ModuleType) -> None:
    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    _apply_existing_source_alignment_patches(source_alignment)
    _register_contrastive_method(source_alignment)

    original_normalize_method = source_alignment.normalize_source_alignment_method
    original_fit_source_alignment_model = source_alignment._fit_source_alignment_model
    original_align_train_test_features = source_alignment.align_train_test_features
    original_transform_inner_heldout_subject = source_alignment._transform_inner_heldout_subject

    def normalize_source_alignment_method(method: str | None) -> str:
        normalized = "none" if method is None else str(method).strip().lower().replace("-", "_")
        if normalized in _CONTRASTIVE_ALIASES:
            return CONTRASTIVE_ALIGNMENT_METHOD
        return original_normalize_method(method)

    def _fit_source_alignment_model(
        features_by_subject: Mapping[Hashable, np.ndarray],
        anchors_by_subject: Mapping[Hashable, np.ndarray],
        *,
        config,
        sample_mode: str,
        external_anchor_mode: bool,
    ):
        if config.method != CONTRASTIVE_ALIGNMENT_METHOD:
            return original_fit_source_alignment_model(
                features_by_subject,
                anchors_by_subject,
                config=config,
                sample_mode=sample_mode,
                external_anchor_mode=external_anchor_mode,
            )

        subject_ids = tuple(features_by_subject)
        fit_features_by_subject, fit_anchors_by_subject, anchor_filter_metadata = source_alignment._source_alignment_fit_inputs(
            features_by_subject,
            anchors_by_subject,
            external_anchor_mode=external_anchor_mode,
        )
        n_repetitions = source_alignment._effective_repetitions_per_class(fit_anchors_by_subject, sample_mode, config)
        repetition_selection = source_alignment._source_alignment_repetition_selection(config, sample_mode)
        model, alignment = fit_contrastive_alignment(
            fit_features_by_subject,
            fit_anchors_by_subject,
            sample_mode=sample_mode,
            n_repetitions_per_class=n_repetitions,
            repetition_selection=repetition_selection,
            n_components=config.components,
            regularization=config.mcca_regularization,
        )
        transformed_by_subject = {
            subject_id: model.transform(subject_id, features_by_subject[subject_id])
            for subject_id in subject_ids
        }
        anchor_before = alignment.aligned_by_subject
        anchor_after = {
            subject_id: model.transform(subject_id, anchor_before[subject_id])
            for subject_id in subject_ids
        }
        return source_alignment._SourceAlignmentFit(
            model=model,
            alignment=alignment,
            transformed_by_subject=transformed_by_subject,
            anchor_filter_metadata=anchor_filter_metadata,
            anchor_before=anchor_before,
            anchor_after=anchor_after,
            n_components=int(model.n_components),
        )

    def _transform_inner_heldout_subject(
        *,
        test_features: np.ndarray,
        test_anchors: np.ndarray,
        fit,
        config,
    ) -> tuple[np.ndarray, np.ndarray]:
        if config.method != CONTRASTIVE_ALIGNMENT_METHOD:
            return original_transform_inner_heldout_subject(
                test_features=test_features,
                test_anchors=test_anchors,
                fit=fit,
                config=config,
            )

        if not config.fits_target_projection:
            target_feature_mean = np.mean(test_features, axis=0) if config.target_centered_group_projection else None
            return fit.model.transform_group(test_features, feature_mean=target_feature_mean), np.ones(
                test_features.shape[0],
                dtype=bool,
            )

        evaluation_mask = np.ones(test_features.shape[0], dtype=bool)
        projection_features = test_features
        projection_anchors = test_anchors
        selected_offsets_by_class = fit.alignment.selected_offsets_by_class
        if config.target_calibrated or getattr(config, "pseudo_label_target_calibrated", False):
            calibration_mask = source_alignment._inner_target_calibration_mask(
                test_anchors,
                classes=fit.alignment.classes,
                per_anchor=config.target_calibration_per_anchor,
                seed=config.target_calibration_seed,
            )
            evaluation_mask = ~calibration_mask
            if not np.any(evaluation_mask):
                raise ValueError("source-inner contrastive target-calibrated diagnostic left no held-out rows for scoring.")
            projection_features = test_features[calibration_mask]
            projection_anchors = test_anchors[calibration_mask]
            if fit.alignment.n_repetitions_per_class is not None:
                selected_offsets_by_class = {
                    class_position: np.arange(int(fit.alignment.n_repetitions_per_class), dtype=int)
                    for class_position, _class_label in enumerate(fit.alignment.classes)
                }

        target_anchors = source_alignment._target_alignment_matrix(
            projection_features,
            projection_anchors,
            classes=fit.alignment.classes,
            config=config,
            n_repetitions_per_class=fit.alignment.n_repetitions_per_class,
            selected_offsets_by_class=selected_offsets_by_class,
        )
        target_projection = fit_projection_to_contrastive_template(
            target_anchors,
            template=fit.model,
            regularization=config.mcca_regularization,
        )
        scoring_features = test_features[evaluation_mask]
        return transform_with_contrastive_projection(scoring_features, target_projection), evaluation_mask

    def align_train_test_features(
        *,
        train_features: Sequence[Sequence[float]] | np.ndarray,
        train_labels: Sequence[Any] | np.ndarray,
        train_subject_ids: Sequence[Hashable] | np.ndarray,
        test_features: Sequence[Sequence[float]] | np.ndarray,
        config,
        target_labels: Sequence[Any] | np.ndarray | None = None,
        train_anchor_values: Sequence[Any] | np.ndarray | None = None,
        target_anchor_values: Sequence[Any] | np.ndarray | None = None,
        target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
        target_calibration_anchor_values: Sequence[Any] | np.ndarray | None = None,
        compute_source_inner_diagnostics: bool = True,
    ):
        if config.method != CONTRASTIVE_ALIGNMENT_METHOD:
            return original_align_train_test_features(
                train_features=train_features,
                train_labels=train_labels,
                train_subject_ids=train_subject_ids,
                test_features=test_features,
                config=config,
                target_labels=target_labels,
                train_anchor_values=train_anchor_values,
                target_anchor_values=target_anchor_values,
                target_calibration_features=target_calibration_features,
                target_calibration_labels=target_calibration_labels,
                target_calibration_anchor_values=target_calibration_anchor_values,
                compute_source_inner_diagnostics=compute_source_inner_diagnostics,
            )
        return _align_contrastive_train_test_features(
            source_alignment,
            train_features=train_features,
            train_labels=train_labels,
            train_subject_ids=train_subject_ids,
            test_features=test_features,
            config=config,
            target_labels=target_labels,
            train_anchor_values=train_anchor_values,
            target_anchor_values=target_anchor_values,
            target_calibration_features=target_calibration_features,
            target_calibration_labels=target_calibration_labels,
            target_calibration_anchor_values=target_calibration_anchor_values,
            compute_source_inner_diagnostics=compute_source_inner_diagnostics,
        )

    source_alignment.normalize_source_alignment_method = normalize_source_alignment_method
    source_alignment._fit_source_alignment_model = _fit_source_alignment_model
    source_alignment._transform_inner_heldout_subject = _transform_inner_heldout_subject
    source_alignment.align_train_test_features = align_train_test_features
    setattr(source_alignment, _PATCH_MARKER, True)


def _register_contrastive_method(source_alignment: ModuleType) -> None:
    if CONTRASTIVE_ALIGNMENT_METHOD not in source_alignment.SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS:
        source_alignment.SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS = (
            *source_alignment.SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS,
            CONTRASTIVE_ALIGNMENT_METHOD,
        )
    source_alignment.SOURCE_ALIGNMENT_METHODS = (
        "none",
        *source_alignment.SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS,
        *source_alignment.SOURCE_ALIGNMENT_UNSUPERVISED_METHODS,
    )


def _align_contrastive_train_test_features(
    source_alignment: ModuleType,
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    train_subject_ids: Sequence[Hashable] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config,
    target_labels: Sequence[Any] | np.ndarray | None,
    train_anchor_values: Sequence[Any] | np.ndarray | None,
    target_anchor_values: Sequence[Any] | np.ndarray | None,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None,
    target_calibration_labels: Sequence[Any] | np.ndarray | None,
    target_calibration_anchor_values: Sequence[Any] | np.ndarray | None,
    compute_source_inner_diagnostics: bool,
):
    target_calibration_args = (
        target_calibration_features,
        target_calibration_labels,
        target_calibration_anchor_values,
    )
    if target_labels is not None and not config.oracle_target_calibrated:
        raise ValueError("Strict source-only alignment does not accept target labels.")
    if target_anchor_values is not None and not config.oracle_target_calibrated:
        raise ValueError("Strict source-only alignment does not accept target anchor values.")
    if any(value is not None for value in target_calibration_args) and not (
        config.target_calibrated or config.pseudo_label_target_calibrated
    ):
        raise ValueError(
            "Target calibration rows are accepted only with target_calibrated_alignment "
            "or pseudo_label_target_calibrated_alignment."
        )

    class_anchor_mode = source_alignment._uses_decoder_label_anchors(config.anchor_mode)
    if class_anchor_mode:
        train_anchor_source = "decoder_labels"
        if train_anchor_values is None:
            train_anchor_values = train_labels
        if config.oracle_target_calibrated and target_anchor_values is None:
            target_anchor_values = target_labels
        if (config.target_calibrated or config.pseudo_label_target_calibrated) and target_calibration_anchor_values is None:
            target_calibration_anchor_values = target_calibration_labels
    else:
        train_anchor_source = "metadata"
        if train_anchor_values is None:
            raise ValueError(f"{config.anchor_mode} alignment requires train_anchor_values derived from metadata.")
    if config.oracle_target_calibrated and target_anchor_values is None:
        raise ValueError(
            "oracle_target_calibrated_alignment requires held-out target labels or anchor values and is not valid for benchmark reporting."
        )
    if (config.target_calibrated or config.pseudo_label_target_calibrated) and target_calibration_anchor_values is None:
        projection_name = (
            "pseudo_label_target_calibrated_alignment"
            if config.pseudo_label_target_calibrated
            else "target_calibrated_alignment"
        )
        raise ValueError(f"{projection_name} requires target calibration labels or anchor values.")

    train_matrix = source_alignment._feature_matrix(train_features, name="train_features")
    test_matrix = source_alignment._feature_matrix(test_features, name="test_features")
    target_calibration_matrix = (
        None
        if target_calibration_features is None
        else source_alignment._feature_matrix(target_calibration_features, name="target_calibration_features")
    )
    train_vector = source_alignment._anchor_value_vector(train_labels)
    subject_vector = np.asarray(train_subject_ids, dtype=object).reshape(-1)
    target_vector = None if target_labels is None else source_alignment._anchor_value_vector(target_labels)
    train_anchor_vector = source_alignment._anchor_vector(
        train_anchor_values,
        expected_length=train_matrix.shape[0],
        name="train_anchor_values",
    )
    target_anchor_vector = (
        None
        if target_anchor_values is None
        else source_alignment._anchor_vector(
            target_anchor_values,
            expected_length=test_matrix.shape[0],
            name="target_anchor_values",
        )
    )
    target_calibration_label_vector = (
        None if target_calibration_labels is None else source_alignment._anchor_value_vector(target_calibration_labels)
    )
    target_calibration_anchor_vector = (
        None
        if target_calibration_anchor_values is None
        else source_alignment._anchor_vector(
            target_calibration_anchor_values,
            expected_length=0 if target_calibration_matrix is None else target_calibration_matrix.shape[0],
            name="target_calibration_anchor_values",
        )
    )

    if train_matrix.shape[0] != train_vector.shape[0]:
        raise ValueError("train_features and train_labels must have the same row count.")
    if train_matrix.shape[0] != subject_vector.shape[0]:
        raise ValueError("train_features and train_subject_ids must have the same row count.")
    if target_vector is not None and test_matrix.shape[0] != target_vector.shape[0]:
        raise ValueError("test_features and target_labels must have the same row count.")
    if (config.target_calibrated or config.pseudo_label_target_calibrated) and target_calibration_matrix is None:
        projection_name = (
            "pseudo_label_target_calibrated_alignment"
            if config.pseudo_label_target_calibrated
            else "target_calibrated_alignment"
        )
        raise ValueError(f"{projection_name} requires target_calibration_features.")
    if target_calibration_matrix is not None and target_calibration_label_vector is not None:
        if target_calibration_matrix.shape[0] != target_calibration_label_vector.shape[0]:
            raise ValueError("target_calibration_features and target_calibration_labels must have the same row count.")
    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width before alignment: "
            f"{train_matrix.shape[1]} != {test_matrix.shape[1]}."
        )
    if target_calibration_matrix is not None and train_matrix.shape[1] != target_calibration_matrix.shape[1]:
        raise ValueError(
            "train_features and target_calibration_features must have the same feature width before alignment: "
            f"{train_matrix.shape[1]} != {target_calibration_matrix.shape[1]}."
        )

    metadata = config.static_metadata()
    subject_ids = tuple(dict.fromkeys(subject_vector.tolist()))
    if len(subject_ids) < 2:
        raise ValueError("Strict source-only alignment requires at least two source subjects.")

    features_by_subject = {subject_id: train_matrix[subject_vector == subject_id] for subject_id in subject_ids}
    labels_by_subject = {subject_id: train_vector[subject_vector == subject_id] for subject_id in subject_ids}
    anchors_by_subject = {subject_id: train_anchor_vector[subject_vector == subject_id] for subject_id in subject_ids}
    sample_mode = source_alignment._alignment_sample_mode(config.anchor_mode)
    fit = source_alignment._fit_source_alignment_model(
        features_by_subject,
        anchors_by_subject,
        config=config,
        sample_mode=sample_mode,
        external_anchor_mode=not class_anchor_mode,
    )

    transformed_train = np.empty((train_matrix.shape[0], fit.n_components), dtype=float)
    for subject_id in subject_ids:
        transformed_train[subject_vector == subject_id] = fit.transformed_by_subject[subject_id]

    source_inner_raw_ba = float("nan")
    source_inner_aligned_ba = float("nan")
    if compute_source_inner_diagnostics:
        source_inner_raw_ba, source_inner_aligned_ba = source_alignment._source_inner_strict_loso_scores(
            features_by_subject=features_by_subject,
            labels_by_subject=labels_by_subject,
            anchors_by_subject=anchors_by_subject,
            config=config,
            sample_mode=sample_mode,
            external_anchor_mode=not class_anchor_mode,
        )

    if config.fits_target_projection:
        projection_features = (
            target_calibration_matrix
            if config.target_calibrated or config.pseudo_label_target_calibrated
            else test_matrix
        )
        projection_anchors = (
            target_calibration_anchor_vector
            if config.target_calibrated or config.pseudo_label_target_calibrated
            else target_anchor_vector
        )
        target_anchors = source_alignment._target_alignment_matrix(
            projection_features,
            projection_anchors,
            classes=fit.alignment.classes,
            config=config,
            n_repetitions_per_class=fit.alignment.n_repetitions_per_class,
            selected_offsets_by_class=fit.alignment.selected_offsets_by_class,
        )
        target_projection = fit_projection_to_contrastive_template(
            target_anchors,
            template=fit.model,
            regularization=config.mcca_regularization,
        )
        transformed_test = transform_with_contrastive_projection(test_matrix, target_projection)
        target_alignment_rows = int(target_projection.n_alignment_rows)
        target_projection_fit = (
            "pseudo_label_contrastive_ridge_projection"
            if config.pseudo_label_target_calibrated
            else "target_calibrated_contrastive_ridge_projection"
            if config.target_calibrated
            else "oracle_contrastive_ridge_projection"
        )
    else:
        target_feature_mean = np.mean(test_matrix, axis=0) if config.target_centered_group_projection else None
        transformed_test = fit.model.transform_group(test_matrix, feature_mean=target_feature_mean)
        target_alignment_rows = ""
        target_projection_fit = (
            "source_group_contrastive_projection_target_centered"
            if config.target_centered_group_projection
            else "source_group_contrastive_projection"
        )

    n_alignment_rows = int(next(iter(fit.anchor_before.values())).shape[0])
    source_inner_gain = (
        source_inner_aligned_ba - source_inner_raw_ba
        if np.isfinite(source_inner_aligned_ba) and np.isfinite(source_inner_raw_ba)
        else float("nan")
    )
    low_rank_warning = source_alignment._alignment_low_rank_warning(fit.alignment)
    diagnostics = {
        "alignment_method": config.method,
        "sample_mode": sample_mode,
        "n_source_subjects": len(subject_ids),
        "n_classes": len(fit.alignment.classes),
        "n_alignment_rows": n_alignment_rows,
        "n_repetitions_per_class": "" if fit.alignment.n_repetitions_per_class is None else int(fit.alignment.n_repetitions_per_class),
        "requested_components": source_alignment._component_label(config.components),
        "actual_components": int(fit.n_components),
        "feature_dim": int(train_matrix.shape[1]),
        "decode_feature_dim": int(transformed_test.shape[1]),
        "uses_channel_projection_collapse": False,
        "alignment_dimensionality_reduction": bool(transformed_test.shape[1] < train_matrix.shape[1]),
        "alignment_low_rank_warning": low_rank_warning,
        "anchor_row_correlation_before": source_alignment._finite_or_blank(
            source_alignment._mean_pairwise_anchor_row_correlation(fit.anchor_before)
        ),
        "anchor_row_correlation_after": source_alignment._finite_or_blank(
            source_alignment._mean_pairwise_anchor_row_correlation(fit.anchor_after)
        ),
        "source_inner_decoding_before_alignment": source_alignment._finite_or_blank(source_inner_raw_ba),
        "source_inner_decoding_after_alignment": source_alignment._finite_or_blank(source_inner_aligned_ba),
        "source_inner_raw_balanced_accuracy": source_alignment._finite_or_blank(source_inner_raw_ba),
        "source_inner_aligned_balanced_accuracy": source_alignment._finite_or_blank(source_inner_aligned_ba),
        "source_inner_aligned_minus_raw": source_alignment._finite_or_blank(source_inner_gain),
        "source_inner_validation_type": source_alignment._source_inner_validation_type(
            config,
            compute_source_inner_diagnostics=compute_source_inner_diagnostics,
        ),
        "uses_unlabeled_target_data": bool(
            config.pseudo_label_target_calibrated or config.target_centered_group_projection
        ),
        "covariance_alignment_estimator": "",
        "target_transform_type": target_projection_fit,
    }

    return source_alignment.SourceAlignmentResult(
        train_features=transformed_train,
        test_features=transformed_test,
        metadata={
            **metadata,
            "alignment_n_components": int(fit.n_components),
            "alignment_n_source_subjects": len(subject_ids),
            "alignment_n_classes": len(fit.alignment.classes),
            "alignment_n_anchor_values": int(len(fit.alignment.classes)),
            "alignment_anchor_value_source": train_anchor_source,
            "alignment_common_anchor_count": int(len(fit.alignment.classes)),
            **fit.anchor_filter_metadata,
            "alignment_repetitions_per_class": "" if fit.alignment.n_repetitions_per_class is None else int(fit.alignment.n_repetitions_per_class),
            "alignment_target_alignment_rows": target_alignment_rows,
            "alignment_target_projection_fit": target_projection_fit,
            "alignment_low_rank_warning": low_rank_warning,
            "alignment_target_labels_used": bool(
                (config.oracle_target_calibrated and target_labels is not None)
                or (config.target_calibrated and target_calibration_labels is not None)
            ),
            "alignment_target_pseudo_labels_used": bool(
                config.pseudo_label_target_calibrated and target_calibration_labels is not None
            ),
            "alignment_target_anchor_values_used": bool(
                (config.oracle_target_calibrated and target_anchor_vector is not None)
                or (
                    (config.target_calibrated or config.pseudo_label_target_calibrated)
                    and target_calibration_anchor_vector is not None
                )
            ),
        },
        diagnostics=diagnostics,
    )


class _SourceAlignmentContrastivePatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):  # type: ignore[override]
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped_loader.exec_module(module)
        _patch_source_alignment(module)


class _SourceAlignmentContrastivePatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _SourceAlignmentContrastivePatchLoader):
            return spec
        spec.loader = _SourceAlignmentContrastivePatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install contrastive source-alignment support."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_source_alignment(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _SourceAlignmentContrastivePatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
