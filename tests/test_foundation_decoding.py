import numpy as np
import pytest

from neureptrace.decoding.foundation import (
    FrozenTorchEncoderTransformer,
    fit_foundation_linear_probe,
    register_foundation_linear_probe,
)


torch = pytest.importorskip("torch")


class TinyEncoder(torch.nn.Module):
    def forward(self, x):
        if x.ndim == 2:
            return torch.stack((x[:, :3].mean(dim=1), x[:, 3:].mean(dim=1)), dim=1)
        return torch.stack((x[:, 0, :].mean(dim=1), x[:, 1, :].mean(dim=1)), dim=1)


def test_frozen_torch_encoder_transformer_shapes_and_freezes_parameters():
    encoder = torch.nn.Linear(6, 2, bias=False)
    features = np.arange(30, dtype=np.float32).reshape(5, 6)

    transformer = FrozenTorchEncoderTransformer(encoder=encoder, batch_size=2)
    latent = transformer.fit_transform(features)

    assert latent.shape == (5, 2)
    assert transformer.n_features_in_ == 6
    assert not any(parameter.requires_grad for parameter in transformer.encoder_.parameters())


def test_foundation_linear_probe_fits_source_labels():
    rng = np.random.default_rng(7)
    labels = np.array([0, 1] * 20)
    features = rng.normal(size=(40, 6)).astype(np.float32)
    features[labels == 0, :3] += 2.0
    features[labels == 1, 3:] += 2.0

    model = fit_foundation_linear_probe(
        features,
        labels,
        {
            "encoder": TinyEncoder(),
            "input_shape": (2, 3),
            "probe": "logistic",
            "C": 1.0,
            "max_iter": 200,
        },
        random_state=11,
    )

    probabilities = model.predict_proba(features[:4])
    assert probabilities.shape == (4, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 4


def test_register_foundation_linear_probe_exposes_optional_decoder():
    import neureptrace.decoding as decoding

    register_foundation_linear_probe()

    assert "foundation-linear-probe" in decoding.DECODER_CHOICES
    assert "foundation-linear-probe" in decoding.DECODER_CLI_CHOICES
    assert decoding.normalize_decoder_name("foundation_linear_probe") == "foundation-linear-probe"


def test_registered_foundation_linear_probe_works_with_make_decoder():
    import neureptrace.decoding as decoding

    register_foundation_linear_probe()
    rng = np.random.default_rng(13)
    labels = np.array([0, 1] * 18)
    features = rng.normal(size=(36, 6)).astype(np.float32)
    features[labels == 0, :3] += 1.5
    features[labels == 1, 3:] += 1.5

    model = decoding.make_decoder(
        "foundation-linear-probe",
        emission_mode="uncalibrated",
        classifier_param={
            "encoder": TinyEncoder(),
            "input_shape": (2, 3),
            "probe": "logistic",
            "max_iter": 200,
        },
    )
    model.fit(features, labels)

    probabilities = decoding.predict_emission_probabilities(model, features[:5], emission_mode="uncalibrated")
    assert probabilities.shape == (5, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 5


def test_foundation_linear_probe_requires_encoder_or_model_path():
    features = np.ones((6, 4), dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="requires either an encoder object or a model_path"):
        fit_foundation_linear_probe(features, labels, {"probe": "logistic"})
