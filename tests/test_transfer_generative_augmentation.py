import numpy as np

from neureptrace.decoding.transfer import cross_validate_feature_decoding, evaluate_feature_transfer


class _RecordingConstantClassifier:
    def __init__(self, features, labels, label):
        self.n_rows = features.shape[0]
        self.labels = np.asarray(labels)
        self.label = label

    def predict(self, features):
        return np.full(features.shape[0], self.label)


def test_cross_validate_feature_decoding_augments_each_training_fold_only():
    seen_rows = []

    def fit_model(features, labels):
        seen_rows.append(features.shape[0])
        return _RecordingConstantClassifier(features, labels, label=1)

    result = cross_validate_feature_decoding(
        np.array([[-2.0], [2.0], [-1.0], [1.0]]),
        np.array([1, 2, 1, 2]),
        n_folds=2,
        components_pca=float("inf"),
        fit_model=fit_model,
        generative_augmentation={"method": "source_gaussian", "synthetic_per_class": 1, "random_state": 1},
    )

    assert seen_rows == [4, 4]
    assert result.predictions.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_evaluate_feature_transfer_accepts_explicit_unlabeled_target_style_augmentation():
    def fit_model(features, labels):
        return _RecordingConstantClassifier(features, labels, label=0)

    result = evaluate_feature_transfer(
        train_features=np.array([[-2.0], [-1.0], [1.0], [2.0]]),
        train_labels=np.array([0, 0, 1, 1]),
        validation_features=np.array([[-1.5], [1.5]]),
        validation_labels=np.array([0, 0]),
        classifier="multiclass-svm",
        classifier_param=1.0,
        components_pca=float("inf"),
        fit_model=fit_model,
        generative_augmentation={"method": "target_style_gaussian", "synthetic_per_class": 1, "random_state": 2},
        generative_target_features=np.array([[10.0], [11.0], [12.0]]),
    )

    assert result.model_bundle.model.n_rows == 6
    assert result.accuracy == 1.0
