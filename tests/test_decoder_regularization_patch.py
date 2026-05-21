import numpy as np

from neureptrace.decoding import make_decoder


def _logistic_step(model):
    return model.named_steps["logisticregression"]


def test_sparse_logistic_uses_actual_l1_penalty():
    model = make_decoder("sparse-logreg", max_iter=10)
    classifier = _logistic_step(model)

    assert classifier.penalty == "l1"
    assert classifier.solver == "saga"
    assert classifier.class_weight == "balanced"


def test_elastic_net_logistic_uses_actual_elasticnet_penalty():
    model = make_decoder("logistic-elastic-net", max_iter=10)
    classifier = _logistic_step(model)

    assert classifier.penalty == "elasticnet"
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
    assert sparse.best_estimator_.named_steps["logisticregression"].penalty == "l1"

    elastic = make_decoder("elastic_net_logistic", max_iter=2000, tune_hyperparameters=True, tuning_cv=2, tuning_c_grid=(0.1, 1.0))
    elastic.fit(features, labels)
    assert elastic.best_estimator_.named_steps["logisticregression"].penalty == "elasticnet"
    assert "logisticregression__l1_ratio" in elastic.best_params_
