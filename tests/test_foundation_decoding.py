import numpy as np
import pytest

from neureptrace.decoding.foundation import (
    FrozenTorchEncoderTransformer,
    fit_foundation_linear_probe,
    get_foundation_model_spec,
    list_foundation_model_families,
    make_bendr_linear_probe,
    normalize_foundation_linear_probe_params,
    register_foundation_linear_probe,
)


torch = pytest.importorskip("torch")


class TinyEncoder(torch.nn.Module):
    def forward(self, x):
        if x.ndim == 2:
            return torch.stack((x[:, :3].mean(dim=1), x[:, 3:].mean(dim=1)), dim=1)
        return torch.stack((x[:, 0, :].mean(dim=1), x[:, 1, :].mean(dim=1)), dim=1)


class TokenEncoder(torch.nn.Module):
    def forward(self, x):
        base = x.mean(dim=-1)
        return torch.stack((base, base + 1.0, base + 2.0), dim=1)


class LinearStateEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 2, bias=False)

    def forward(self, x):
        return self.linear(x)


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


def test_model_family_specs_normalize_aliases_and_defaults():
    assert "bendr" in list_foundation_model_families()
    assert get_foundation_model_spec("EEG-PT").name == "eegpt"

    labram_params = normalize_foundation_linear_probe_params(
        {
            "foundation_model": "LaBraM",
            "encoder": TokenEncoder(),
            "input_shape": "2x3",
        }
    )

    assert labram_params["model_family"] == "labram"
    assert labram_params["pooling"] == "cls"
    assert labram_params["input_shape"] == (2, 3)


def test_family_default_pooling_works_for_token_encoders():
    features = np.arange(24, dtype=np.float32).reshape(4, 6)

    transformer = FrozenTorchEncoderTransformer(model_family="labram", encoder=TokenEncoder(), input_shape=(2, 3))
    latent = transformer.fit_transform(features)

    assert latent.shape == (4, 2)
    assert np.allclose(latent[0], [1.0, 4.0])


def test_state_dict_load_mode_uses_model_factory_and_checkpoint_key(tmp_path):
    encoder = LinearStateEncoder()
    with torch.no_grad():
        encoder.linear.weight[:] = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    checkpoint_path = tmp_path / "encoder.pt"
    torch.save({"encoder_state_dict": encoder.state_dict()}, checkpoint_path)
    features = np.arange(18, dtype=np.float32).reshape(3, 6)

    transformer = FrozenTorchEncoderTransformer(
        model_family="bendr",
        model_path=checkpoint_path,
        model_factory=LinearStateEncoder,
        load_mode="state_dict",
        state_dict_key="encoder_state_dict",
    )
    latent = transformer.fit_transform(features)

    assert latent.shape == (3, 2)
    assert np.allclose(latent, [[0, 5], [6, 11], [12, 17]])
    assert not any(parameter.requires_grad for parameter in transformer.encoder_.parameters())


def test_bendr_convenience_wrapper_uses_bendr_family_defaults():
    model = make_bendr_linear_probe(
        {
            "encoder": TinyEncoder(),
            "input_shape": (2, 3),
            "probe": "ridge",
        }
    )

    transformer = model.named_steps["frozentorchencodertransformer"]
    assert transformer.model_family == "bendr"
    assert transformer.pooling == "mean_time"


def test_register_foundation_linear_probe_exposes_optional_decoder():
    import neureptrace.decoding as decoding

    register_foundation_linear_probe()

    assert "foundation-linear-probe" in decoding.DECODER_CHOICES
    assert "foundation-linear-probe" in decoding.DECODER_CLI_CHOICES
    assert "bendr-linear-probe" in decoding.DECODER_CHOICES
    assert "labram-linear-probe" in decoding.DECODER_CLI_CHOICES
    assert decoding.normalize_decoder_name("foundation_linear_probe") == "foundation-linear-probe"
    assert decoding.normalize_decoder_name("bendr_linear_probe") == "bendr-linear-probe"


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


def test_registered_family_linear_probe_works_with_make_decoder():
    import neureptrace.decoding as decoding

    register_foundation_linear_probe()
    rng = np.random.default_rng(17)
    labels = np.array([0, 1] * 16)
    features = rng.normal(size=(32, 6)).astype(np.float32)
    features[labels == 0, :3] += 1.5
    features[labels == 1, 3:] += 1.5

    model = decoding.make_decoder(
        "bendr-linear-probe",
        emission_mode="uncalibrated",
        classifier_param={
            "encoder": TinyEncoder(),
            "input_shape": (2, 3),
            "probe": "ridge",
        },
    )
    model.fit(features, labels)

    probabilities = decoding.predict_emission_probabilities(model, features[:5], emission_mode="uncalibrated")
    assert probabilities.shape == (5, 2)
    assert probabilities.sum(axis=1).round(6).tolist() == [1.0] * 5


def test_foundation_linear_probe_requires_encoder_model_path_or_factory():
    features = np.ones((6, 4), dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match="requires an encoder object, model_path, or model_factory"):
        fit_foundation_linear_probe(features, labels, {"probe": "logistic"})
