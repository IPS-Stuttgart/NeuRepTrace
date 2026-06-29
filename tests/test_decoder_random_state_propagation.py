from __future__ import annotations

from neureptrace.decoding import make_decoder, make_tuned_decoder


def _random_state_values(params: dict[str, object], suffix: str) -> set[object]:
    return {value for name, value in params.items() if name.endswith(suffix)}


def test_linear_svm_decoder_threads_random_state_to_estimator():
    model = make_decoder("linear_svm", emission_mode="uncalibrated", random_state=41)

    assert model.get_params(deep=True)["linearsvc__random_state"] == 41


def test_calibrated_linear_svm_decoder_threads_random_state_to_estimator():
    model = make_decoder("linear_svm", random_state=43)

    values = _random_state_values(model.get_params(deep=True), "linearsvc__random_state")
    assert values == {43}


def test_tuned_linear_svm_decoder_threads_random_state_before_grid_search_cloning():
    model = make_tuned_decoder(
        "linear_svm",
        emission_mode="uncalibrated",
        random_state=53,
        cv=2,
        c_grid=(0.1, 1.0),
    )

    assert model.get_params(deep=True)["estimator__linearsvc__random_state"] == 53
