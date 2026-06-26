import numpy as np

from neureptrace.decoding.vrex import fit_vrex_predict_proba


def test_vrex_preserves_tuple_labels_with_balanced_class_weight():
    source_labels = [(0, "left"), (1, "right")] * 6
    source_domains = ["sub-01"] * 6 + ["sub-02"] * 6
    class_signal = np.asarray([-2.0 if label[0] == 0 else 2.0 for label in source_labels])
    domain_signal = np.asarray([-0.2 if domain == "sub-01" else 0.2 for domain in source_domains])
    source_features = np.column_stack([class_signal + 0.05 * domain_signal, domain_signal])
    test_features = np.asarray([[-2.0, 0.0], [2.0, 0.0]])

    result = fit_vrex_predict_proba(
        source_features=source_features,
        source_labels=source_labels,
        source_domains=source_domains,
        test_features=test_features,
        penalty_weight=0.1,
        max_iter=100,
        tol=1e-7,
    )

    assert result.probabilities.shape == (2, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(np.isfinite(result.model.class_weight_vector_))
    assert np.all(result.model.class_weight_vector_ > 0.0)
    assert result.model.classes_.shape == (2,)
    assert result.model.classes_.tolist() == [(0, "left"), (1, "right")]
    predictions = result.model.predict(test_features)
    assert predictions.shape == (2,)
    assert all(prediction in result.model.classes_.tolist() for prediction in predictions.tolist())
