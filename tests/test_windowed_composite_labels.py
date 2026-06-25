import numpy as np

from neureptrace.decoding.windowed import score_windowed_decoding


class _FixedTupleClassifier:
    def predict(self, features):
        predictions = np.empty(np.asarray(features).shape[0], dtype=object)
        predictions[0] = ("a", "x")
        predictions[1] = ("b", "x")
        return predictions

    def decision_function(self, features):
        return np.zeros(np.asarray(features).shape[0])


def _fit_fixed_tuple_classifier(_features, _labels):
    return _FixedTupleClassifier()


def test_score_windowed_decoding_accepts_tuple_labels():
    result = score_windowed_decoding(
        train_features=np.array([[0.0], [1.0]]),
        train_labels=[("a", "x"), ("b", "x")],
        validation_features=np.array([[0.0], [1.0]]),
        validation_labels=[("a", "x"), ("b", "x")],
        fit_model=_fit_fixed_tuple_classifier,
        components_pca=float("inf"),
        n_permutations=1,
        permutation_rng=np.random.default_rng(13),
    )

    assert result.accuracy == 1.0
    assert result.balanced_accuracy == 1.0
    assert result.predictions.dtype == object
    assert result.predictions.tolist() == [("a", "x"), ("b", "x")]
    assert result.permutation_accuracy.shape == (1,)
