import numpy as np
import pytest

from neureptrace.decoding import make_decoder


@pytest.mark.parametrize(
    "decoder",
    [
        "logistic",
        "sparse_logistic",
        "elastic_net_logistic",
        "linear_svm",
        "ovo_logistic",
        "ovo_linear_svm",
        "ecoc_linear_svm",
        "hierarchical_logistic",
        "torch_mlp",
    ],
)
@pytest.mark.parametrize("classifier_param", [True, False, np.bool_(True), np.bool_(False), np.asarray(True), np.array([True])])
def test_make_decoder_rejects_boolean_classifier_params(decoder, classifier_param):
    with pytest.raises(ValueError, match="numeric, not boolean"):
        make_decoder(decoder, classifier_param=classifier_param)


@pytest.mark.parametrize("classifier_param", [0, -1, np.nan, np.inf, "not-a-number"])
def test_make_decoder_rejects_malformed_positive_classifier_params(classifier_param):
    with pytest.raises(ValueError, match="positive finite"):
        make_decoder("logistic", classifier_param=classifier_param)


def test_make_decoder_allows_zero_torch_mlp_weight_decay():
    decoder = make_decoder("torch_mlp", classifier_param=0.0)

    assert decoder.named_steps["torchmlpclassifier"].weight_decay == 0.0


@pytest.mark.parametrize("classifier_param", [-1, np.nan, np.inf, "not-a-number"])
def test_make_decoder_rejects_malformed_torch_mlp_weight_decay(classifier_param):
    with pytest.raises(ValueError, match="non-negative finite"):
        make_decoder("torch_mlp", classifier_param=classifier_param)
