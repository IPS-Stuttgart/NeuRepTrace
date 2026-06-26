import numpy as np
import pytest

from neureptrace.decoding.vrex import LinearVRExClassifier


def _source_table():
    return (
        np.asarray(
            [
                [10.0, 2.0],
                [11.0, 2.5],
                [12.0, 3.0],
                [13.0, 3.5],
            ]
        ),
        np.asarray(["left", "right", "left", "right"], dtype=object),
        np.asarray(["s1", "s1", "s2", "s2"], dtype=object),
    )


def test_vrex_normalizes_string_boolean_hyperparameters():
    features, labels, domains = _source_table()

    model = LinearVRExClassifier(fit_intercept="false", standardize="false", max_iter=3, tol=1e-4)
    model.fit(features, labels, source_domains=domains)

    assert model.fit_intercept_ is False
    assert model.standardize_ is False
    assert np.allclose(model.intercept_, 0.0)
    assert np.allclose(model.feature_mean_, np.zeros(features.shape[1]))
    assert np.allclose(model.feature_scale_, np.ones(features.shape[1]))
    assert model.metadata()["vrex_fit_intercept"] is False
    assert model.metadata()["vrex_standardize"] is False


def test_vrex_accepts_numeric_boolean_hyperparameters():
    features, labels, domains = _source_table()

    model = LinearVRExClassifier(fit_intercept=1.0, standardize=0.0, max_iter=3, tol=1e-4)
    model.fit(features, labels, source_domains=domains)

    assert model.fit_intercept_ is True
    assert model.standardize_ is False
    assert model.intercept_.shape == (1,)
    assert np.allclose(model.feature_scale_, np.ones(features.shape[1]))


@pytest.mark.parametrize("param", ["maybe", 2, np.nan])
def test_vrex_rejects_invalid_boolean_hyperparameters(param):
    features, labels, domains = _source_table()

    model = LinearVRExClassifier(fit_intercept=param)

    with pytest.raises(ValueError, match="fit_intercept must be a boolean"):
        model.fit(features, labels, source_domains=domains)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("max_iter", True, "max_iter must be a positive integer"),
        ("max_iter", np.bool_(True), "max_iter must be a positive integer"),
        ("tol", True, "tol must be positive and finite"),
        ("penalty_weight", True, "penalty_weight must be finite and non-negative"),
        ("l2", False, "l2 must be finite and non-negative"),
    ],
)
def test_vrex_rejects_boolean_numeric_hyperparameters(parameter, value, message):
    features, labels, domains = _source_table()

    model = LinearVRExClassifier(**{parameter: value})

    with pytest.raises(ValueError, match=message):
        model.fit(features, labels, source_domains=domains)
