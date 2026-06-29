import numpy as np
import pytest

from neureptrace.decoding.windowed import (
    fit_window_model,
    permutation_p_from_accuracy,
    permutation_score_curves,
    predict_window_model,
    score_windowed_decoding,
)


class _SignClassifier:
    def __init__(self):
        self.negative_label = None
        self.positive_label = None

    def fit(self, features, labels):
        labels = np.asarray(labels)
        features = np.asarray(features)
        self.negative_label = _majority_label(labels[features[:, 0] < 0])
        self.positive_label = _majority_label(labels[features[:, 0] >= 0])
        return self

    def predict(self, features):
        features = np.asarray(features)
        return np.where(features[:, 0] < 0, self.negative_label, self.positive_label)

    def decision_function(self, features):
        return np.asarray(features)[:, 0]


def _majority_label(labels):
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def _fit_sign_classifier(features, labels):
    return _SignClassifier().fit(features, labels)


def test_fit_window_model_applies_pca_and_predicts_validation_features():
    train_features = np.array(
        [
            [-2.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )
    train_labels = np.array([0, 0, 1, 1])
    validation_features = np.array([[-1.5, -1.5], [1.5, 1.5]])

    model_bundle = fit_window_model(
        train_features,
        train_labels,
        fit_model=_fit_sign_classifier,
        components_pca=1,
        train_window=(0.125, 0.225),
    )
    predictions, scores = predict_window_model(model_bundle, validation_features)

    assert predictions.tolist() == [0, 1]
    assert scores.shape == (2,)
    assert model_bundle.train_window == (0.125, 0.225)
    assert model_bundle.actual_components_pca == 1
    assert model_bundle.pca_coeff.shape == (2, 1)
    assert model_bundle.explained_variance_percent == pytest.approx(100.0)


def test_fit_window_model_accepts_fractional_pca_variance_ratio():
    train_features = np.array(
        [
            [-3.0, -3.0, 0.2],
            [-1.0, -1.0, -0.1],
            [1.0, 1.0, 0.1],
            [3.0, 3.0, -0.2],
        ]
    )
    train_labels = np.array([0, 0, 1, 1])
    validation_features = np.array([[-2.0, -2.0, 0.0], [2.0, 2.0, 0.0]])

    model_bundle = fit_window_model(
        train_features,
        train_labels,
        fit_model=_fit_sign_classifier,
        components_pca=0.90,
    )
    predictions, scores = predict_window_model(model_bundle, validation_features)

    assert predictions.tolist() == [0, 1]
    assert scores.shape == (2,)
    assert 1 <= model_bundle.actual_components_pca <= train_features.shape[1]
    assert model_bundle.pca_coeff.shape == (train_features.shape[1], model_bundle.actual_components_pca)
    assert model_bundle.explained_variance_percent >= 90.0


@pytest.mark.parametrize(
    "components_pca",
    [
        0,
        0.0,
        1.5,
        -1,
        np.nan,
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.asarray(True),
        np.array([1]),
        "not-a-pca-value",
    ],
)
def test_fit_window_model_rejects_invalid_pca_components(components_pca):
    with pytest.raises(ValueError, match="components_pca must be"):
        fit_window_model(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            fit_model=_fit_sign_classifier,
            components_pca=components_pca,
        )


def test_fit_window_model_rejects_feature_matrix_without_columns():
    with pytest.raises(ValueError, match="train_features must contain at least one column"):
        fit_window_model(
            np.empty((2, 0)),
            np.array([0, 1]),
            fit_model=_fit_sign_classifier,
            components_pca=float("inf"),
        )


def test_fit_window_model_rejects_multidimensional_train_labels():
    with pytest.raises(ValueError, match="train_labels must be one-dimensional"):
        fit_window_model(
            np.array([[-1.0], [1.0]]),
            np.array([[0], [1]]),
            fit_model=_fit_sign_classifier,
            components_pca=float("inf"),
        )


def test_score_windowed_decoding_returns_accuracy_predictions_and_permutation_p():
    result = score_windowed_decoding(
        train_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        train_labels=np.array([0, 0, 1, 1]),
        validation_features=np.array([[-1.5], [1.5]]),
        validation_labels=np.array([0, 1]),
        fit_model=_fit_sign_classifier,
        components_pca=float("inf"),
        n_permutations=4,
        permutation_rng=np.random.default_rng(13),
    )

    assert result.accuracy == 1.0
    assert result.predictions.tolist() == [0, 1]
    assert result.permutation_accuracy.shape == (4,)
    assert result.permutation_p_value == permutation_p_from_accuracy(1.0, result.permutation_accuracy)
    assert result.permutation_balanced_accuracy.shape == (4,)
    assert result.balanced_accuracy_p_value == permutation_p_from_accuracy(
        result.balanced_accuracy,
        result.permutation_balanced_accuracy,
    )


def test_score_windowed_decoding_rejects_negative_permutation_counts():
    with pytest.raises(ValueError, match="n_permutations must be a non-negative integer"):
        score_windowed_decoding(
            train_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
            train_labels=np.array([0, 0, 1, 1]),
            validation_features=np.array([[-1.5], [1.5]]),
            validation_labels=np.array([0, 1]),
            fit_model=_fit_sign_classifier,
            n_permutations=-1,
        )


def test_permutation_score_curves_returns_accuracy_and_balanced_accuracy():
    accuracy, balanced_accuracy = permutation_score_curves(
        train_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        train_labels=np.array([0, 0, 1, 1]),
        validation_features=np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
        validation_labels=np.array([0, 0, 0, 0, 1]),
        fit_model=_fit_sign_classifier,
        n_permutations=3,
        permutation_rng=np.random.default_rng(13),
    )

    assert accuracy.shape == (3,)
    assert balanced_accuracy.shape == (3,)
    assert np.all((0.0 <= balanced_accuracy) & (balanced_accuracy <= 1.0))


def test_permutation_score_curves_rejects_fractional_permutation_counts():
    kwargs = {
        "train_features": np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        "train_labels": np.array([0, 0, 1, 1]),
        "validation_features": np.array([[-2.0], [2.0]]),
        "validation_labels": np.array([0, 1]),
        "fit_model": _fit_sign_classifier,
    }

    with pytest.raises(ValueError, match="n_permutations must be a non-negative integer"):
        permutation_score_curves(**kwargs, n_permutations=1.5)

    with pytest.raises(ValueError, match="n_permutations must be a non-negative integer"):
        permutation_score_curves(**kwargs, n_permutations=True)


@pytest.mark.parametrize("n_permutations", [np.asarray(True), np.array([2])])
def test_permutation_score_curves_rejects_array_permutation_counts(n_permutations):
    kwargs = {
        "train_features": np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        "train_labels": np.array([0, 0, 1, 1]),
        "validation_features": np.array([[-2.0], [2.0]]),
        "validation_labels": np.array([0, 1]),
        "fit_model": _fit_sign_classifier,
    }

    with pytest.raises(ValueError, match="n_permutations must be a non-negative integer"):
        permutation_score_curves(**kwargs, n_permutations=n_permutations)


def test_score_windowed_decoding_rejects_mismatched_validation_labels():
    with pytest.raises(ValueError, match="validation_labels length must match feature rows"):
        score_windowed_decoding(
            train_features=np.array([[-1.0], [1.0]]),
            train_labels=np.array([0, 1]),
            validation_features=np.array([[-1.0], [1.0]]),
            validation_labels=np.array([0]),
            fit_model=_fit_sign_classifier,
        )


def test_score_windowed_decoding_rejects_multidimensional_validation_labels():
    with pytest.raises(ValueError, match="validation_labels must be one-dimensional"):
        score_windowed_decoding(
            train_features=np.array([[-1.0], [1.0]]),
            train_labels=np.array([0, 1]),
            validation_features=np.array([[-1.0], [1.0]]),
            validation_labels=np.array([[0], [1]]),
            fit_model=_fit_sign_classifier,
        )


def test_permutation_p_from_accuracy_uses_plus_one_correction():
    assert permutation_p_from_accuracy(0.75, np.array([0.1, 0.5, 0.8])) == pytest.approx(0.5)
