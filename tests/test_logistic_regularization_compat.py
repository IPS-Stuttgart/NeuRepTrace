import numpy as np

from neureptrace import _decoding_regularization_patch
from neureptrace.decoding import make_decoder


def test_regularized_logistic_kwargs_match_installed_sklearn_api():
    kwargs = _decoding_regularization_patch._logistic_regularization_kwargs(1.0)

    assert kwargs["l1_ratio"] == 1.0
    if _decoding_regularization_patch._uses_l1_ratio_logistic_api():
        assert "penalty" not in kwargs
    else:
        assert kwargs["penalty"] == "l1"


def test_sparse_logistic_decoder_preserves_l1_semantics():
    rng = np.random.default_rng(13)
    features = rng.normal(size=(30, 6))
    labels = np.array([0, 1] * 15)

    model = make_decoder("sparse-logreg", max_iter=2000)
    model.fit(features, labels)
    classifier = model.named_steps["logisticregression"]

    assert classifier.l1_ratio == 1.0
    assert model.predict_proba(features[:3]).shape == (3, 2)


def test_elastic_net_logistic_tuning_keeps_l1_ratio_grid():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(24, 6))
    labels = np.array([0, 1] * 12)

    model = make_decoder(
        "elasticnet-logistic",
        max_iter=2000,
        tune_hyperparameters=True,
        tuning_cv=2,
        tuning_c_grid=(0.1, 1.0),
    )
    model.fit(features, labels)

    assert "logisticregression__l1_ratio" in model.best_params_
    assert model.best_params_["logisticregression__l1_ratio"] in {0.15, 0.5, 0.85}
