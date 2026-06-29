import numpy as np
import pytest

from neureptrace.decoding.windowed import fit_window_model, permutation_score_curves


class _SignClassifier:
    def fit(self, features, labels):
        labels = np.asarray(labels)
        features = np.asarray(features)
        self.negative_label = labels[features[:, 0] < 0][0]
        self.positive_label = labels[features[:, 0] >= 0][0]
        return self

    def predict(self, features):
        features = np.asarray(features)
        return np.where(features[:, 0] < 0, self.negative_label, self.positive_label)


def _fit_sign_classifier(features, labels):
    return _SignClassifier().fit(features, labels)


@pytest.mark.parametrize(
    "components_pca",
    [np.asarray(True), np.array([True]), np.asarray(1), np.array([1])],
)
def test_fit_window_model_rejects_array_scalar_pca_components(components_pca):
    with pytest.raises(ValueError, match="components_pca must be"):
        fit_window_model(
            np.array([[-1.0], [1.0]]),
            np.array([0, 1]),
            fit_model=_fit_sign_classifier,
            components_pca=components_pca,
        )


@pytest.mark.parametrize(
    "n_permutations",
    [np.asarray(True), np.array([True]), np.asarray(1), np.array([1])],
)
def test_permutation_score_curves_rejects_array_scalar_counts(n_permutations):
    with pytest.raises(ValueError, match="n_permutations must be a non-negative integer"):
        permutation_score_curves(
            np.array([[-1.0], [1.0]]),
            train_labels=np.array([0, 1]),
            validation_features=np.array([[-1.0], [1.0]]),
            validation_labels=np.array([0, 1]),
            fit_model=_fit_sign_classifier,
            n_permutations=n_permutations,
        )
