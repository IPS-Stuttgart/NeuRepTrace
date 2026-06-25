# ruff: noqa
from __future__ import annotations

import numpy as np

from neureptrace._object_label_utils import label_accuracy
from neureptrace._windowed_composite_labels_patch import _balanced_accuracy, _label_vector

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from neureptrace.decoding import windowed

    def score_windowed_decoding(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        *,
        fit_model,
        components_pca=float("inf"),
        train_window=None,
        n_permutations=0,
        permutation_rng=None,
    ):
        n_permutations = windowed._validate_permutation_count(n_permutations)
        train_features = windowed._feature_matrix(train_features, name="train_features")
        train_labels = _label_vector(train_labels, expected_length=train_features.shape[0], name="train_labels")
        validation_features = windowed._feature_matrix(validation_features, name="validation_features")
        validation_labels = _label_vector(validation_labels, expected_length=validation_features.shape[0], name="validation_labels")
        model_bundle = windowed.fit_window_model(train_features, train_labels, fit_model=fit_model, components_pca=components_pca, train_window=train_window)
        predictions, scores = windowed.predict_window_model(model_bundle, validation_features)
        accuracy = label_accuracy(validation_labels, predictions)
        balanced_accuracy = _balanced_accuracy(predictions, validation_labels)
        permutation_accuracy = np.array([], dtype=float)
        permutation_balanced_accuracy = np.array([], dtype=float)
        permutation_p_value = np.nan
        balanced_accuracy_p_value = np.nan
        if n_permutations > 0:
            transformed_train = windowed.transform_window_features(model_bundle, train_features)
            transformed_validation = windowed.transform_window_features(model_bundle, validation_features)
            permutation_accuracy, permutation_balanced_accuracy = windowed.permutation_score_curves(
                transformed_train,
                validation_features=transformed_validation,
                validation_labels=validation_labels,
                train_labels=train_labels,
                fit_model=fit_model,
                n_permutations=n_permutations,
                permutation_rng=permutation_rng,
            )
            permutation_p_value = windowed.permutation_p_from_accuracy(accuracy, permutation_accuracy)
            balanced_accuracy_p_value = windowed.permutation_p_from_accuracy(balanced_accuracy, permutation_balanced_accuracy)
        return windowed.WindowedDecodingResult(
            model_bundle=model_bundle,
            predictions=predictions,
            scores=scores,
            accuracy=accuracy,
            permutation_accuracy=permutation_accuracy,
            permutation_p_value=permutation_p_value,
            balanced_accuracy=balanced_accuracy,
            permutation_balanced_accuracy=permutation_balanced_accuracy,
            balanced_accuracy_p_value=balanced_accuracy_p_value,
        )

    windowed.score_windowed_decoding = score_windowed_decoding
    _INSTALLED = True
