import numpy as np

from neureptrace import _decoding_regularization_patch
from neureptrace.decoding import make_decoder


def _logistic_step(model):
    return model.named_steps["logisticregression"]


def _expected_penalty(legacy_penalty):
    if _decoding_regularization_patch._uses_l1_ratio_logistic_api():
        return "deprecated"
    return legacy_penalty


def test_sparse_logistic_uses_actual_l1_penalty():
    model = make_decoder("sparse-logreg", max_iter=10)
    classifier = _logistic_step(model)

    assert classifier.penalty == _expected_penalty("l1")
    assert classifier.l1_ratio == 1.0
    assert classifier.solver == "saga"
    assert classifier.class_weight == "balanced"


def test_elastic_net_logistic_uses_actual_elasticnet_penalty():
    model = make_decoder("logistic-elastic-net", max_iter=10)
    classifier = _logistic_step(model)

    assert classifier.penalty == _expected_penalty("elasticnet")
    assert classifier.solver == "saga"
    assert classifier.l1_ratio == 0.5
    assert classifier.class_weight == "balanced"


def test_tuned_regularized_logistic_uses_correct_penalty_inside_grid_search():
    features = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [-2.0, 1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    sparse = make_decoder("sparse_logistic", max_iter=2000, tune_hyperparameters=True, tuning_cv=2, tuning_c_grid=(0.1, 1.0))
    sparse.fit(features, labels)
    sparse_classifier = sparse.best_estimator_.named_steps["logisticregression"]
    assert sparse_classifier.penalty == _expected_penalty("l1")
    assert sparse_classifier.l1_ratio == 1.0

    elastic = make_decoder("elastic_net_logistic", max_iter=2000, tune_hyperparameters=True, tuning_cv=2, tuning_c_grid=(0.1, 1.0))
    elastic.fit(features, labels)
    elastic_classifier = elastic.best_estimator_.named_steps["logisticregression"]
    assert elastic_classifier.penalty == _expected_penalty("elasticnet")
    assert "logisticregression__l1_ratio" in elastic.best_params_
