"""Frozen foundation-model feature extraction for M/EEG decoders.

The classes in this module provide integration points for BENDR-, LaBraM-,
EEGPT-, CBraMod-, or project-local encoders without making any of those model
packages mandatory NeuRepTrace dependencies. Encoders are used as frozen feature
extractors; downstream probes are trained with ordinary NeuRepTrace
source-label workflows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

POOLING_CHOICES = ("flatten", "identity", "mean", "mean_time", "mean_tokens", "mean_channels", "cls", "last")
PROBE_CHOICES = ("logistic", "linear_svm", "ridge", "lda")
INPUT_LAYOUT_CHOICES = ("auto", "none", "channels_first", "time_first", "add_channel_dim", "add_feature_dim")
PREPROCESSING_CHOICES = ("none", "trial_zscore", "channel_zscore")
_STATE_DICT_AUTO_KEYS = ("state_dict", "model_state_dict", "encoder_state_dict", "module_state_dict", "network_state_dict")
_CHECKPOINT_MODULE_AUTO_KEYS = ("encoder", "model", "module", "network")


@dataclass(frozen=True)
class FoundationModelSpec:
    """Metadata and conservative defaults for an external foundation encoder family."""

    name: str
    aliases: tuple[str, ...]
    default_pooling: str = "flatten"
    default_input_layout: str = "channels_first"
    default_load_mode: str = "torchscript"
    default_preprocessing: str = "none"
    default_output_key: str | None = None
    default_output_attribute: str | None = None
    package_hint: str = ""
    notes: str = ""

    def defaults(self) -> dict[str, Any]:
        """Return classifier-param defaults for this family."""

        return {
            "model_family": self.name,
            "pooling": self.default_pooling,
            "input_layout": self.default_input_layout,
            "load_mode": self.default_load_mode,
            "preprocessing": self.default_preprocessing,
            "output_key": self.default_output_key,
            "output_attribute": self.default_output_attribute,
        }


FOUNDATION_MODEL_SPECS: dict[str, FoundationModelSpec] = {
    "generic": FoundationModelSpec(
        name="generic",
        aliases=("external", "custom", "torchscript", "torch", "module"),
        default_pooling="flatten",
        default_input_layout="auto",
        package_hint="A TorchScript module, torch.nn.Module, or importable factory supplied by the caller.",
        notes="Use this for project-local encoders or exported checkpoints when no model-family defaults are desired.",
    ),
    "bendr": FoundationModelSpec(
        name="bendr",
        aliases=("bendr", "bendr_encoder", "bendr-contextualizer", "brain_encode_decode"),
        default_pooling="mean_time",
        default_input_layout="channels_first",
        package_hint="Export a BENDR encoder/contextualizer to TorchScript, or pass an importable factory plus a state_dict checkpoint.",
        notes="BENDR-style encoders usually consume trial tensors shaped batch x channels x time.",
    ),
    "labram": FoundationModelSpec(
        name="labram",
        aliases=("labram", "large-brain-model", "large_brain_model"),
        default_pooling="cls",
        default_input_layout="channels_first",
        package_hint="Export the LaBraM backbone/encoder to TorchScript, or pass a factory from the LaBraM project plus checkpoint weights.",
        notes="Transformer-style outputs commonly expose a class token; override pooling/output selectors if your export differs.",
    ),
    "eegpt": FoundationModelSpec(
        name="eegpt",
        aliases=("eegpt", "eeg-pt", "eeg_pt"),
        default_pooling="cls",
        default_input_layout="channels_first",
        package_hint="Export the EEGPT backbone/encoder to TorchScript, or pass a factory from the EEGPT project plus checkpoint weights.",
        notes="Transformer-style outputs commonly expose a class token or last hidden state; override selectors for project-specific exports.",
    ),
    "cbramod": FoundationModelSpec(
        name="cbramod",
        aliases=("cbramod", "cbra-mod", "cbra_mod", "cbra"),
        default_pooling="mean_time",
        default_input_layout="channels_first",
        package_hint="Export the CBraMod encoder/backbone to TorchScript, or pass an importable factory plus checkpoint weights.",
        notes="CBraMod-like encoders are treated as frozen time-series backbones; override pooling/layout for your concrete export.",
    ),
}

_FOUNDATION_MODEL_ALIASES: dict[str, str] = {}
for _family_name, _spec in FOUNDATION_MODEL_SPECS.items():
    for _alias in (_family_name, *_spec.aliases):
        _FOUNDATION_MODEL_ALIASES[_alias.strip().lower().replace("_", "-")] = _family_name

DEFAULT_FOUNDATION_LINEAR_PROBE_PARAMS: dict[str, Any] = {
    "model_family": "generic",
    "probe": "logistic",
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "batch_size": 128,
    "device": "cpu",
    "pooling": None,
    "input_layout": None,
    "preprocessing": None,
    "load_mode": None,
    "dtype": "float32",
    "pre_encoder_standardize": False,
    "model_kwargs": None,
    "forward_kwargs": None,
    "checkpoint_key": None,
    "state_dict_key": None,
    "strict_load": True,
    "strip_state_dict_prefix": None,
    "encoder_attr": None,
    "output_key": None,
    "output_attribute": None,
    "output_index": None,
}

FAMILY_LINEAR_PROBE_DECODER_NAMES: dict[str, str] = {
    "bendr": "bendr-linear-probe",
    "labram": "labram-linear-probe",
    "eegpt": "eegpt-linear-probe",
    "cbramod": "cbramod-linear-probe",
}


class FrozenTorchEncoderTransformer(TransformerMixin, BaseEstimator):
    """Transform feature rows with a frozen PyTorch foundation encoder."""

    def __init__(
        self,
        *,
        model_family: str | None = None,
        encoder: Any | None = None,
        model_path: str | Path | None = None,
        model_factory: str | Callable[..., Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        input_shape: Sequence[int] | str | None = None,
        input_layout: str | None = None,
        preprocessing: str | None = None,
        batch_size: int = 128,
        device: str = "cpu",
        output_key: str | None = None,
        output_attribute: str | None = None,
        output_index: int | None = None,
        pooling: str | None = None,
        load_mode: str | None = None,
        dtype: str = "float32",
        checkpoint_key: str | Sequence[str] | None = None,
        state_dict_key: str | Sequence[str] | None = None,
        strict_load: bool = True,
        strip_state_dict_prefix: str | Sequence[str] | bool | None = None,
        encoder_attr: str | None = None,
        forward_kwargs: Mapping[str, Any] | None = None,
    ):
        self.model_family = model_family
        self.encoder = encoder
        self.model_path = model_path
        self.model_factory = model_factory
        self.model_kwargs = model_kwargs
        self.input_shape = input_shape
        self.input_layout = input_layout
        self.preprocessing = preprocessing
        self.batch_size = batch_size
        self.device = device
        self.output_key = output_key
        self.output_attribute = output_attribute
        self.output_index = output_index
        self.pooling = pooling
        self.load_mode = load_mode
        self.dtype = dtype
        self.checkpoint_key = checkpoint_key
        self.state_dict_key = state_dict_key
        self.strict_load = strict_load
        self.strip_state_dict_prefix = strip_state_dict_prefix
        self.encoder_attr = encoder_attr
        self.forward_kwargs = forward_kwargs

    def fit(self, features: Sequence[Sequence[float]] | np.ndarray, labels: Sequence | np.ndarray | None = None):
        del labels
        x = _validate_feature_matrix(features, name="features")
        self.n_features_in_ = int(x.shape[1])
        self.model_spec_ = get_foundation_model_spec(self.model_family)
        self.input_shape_ = parse_input_shape(self.input_shape)
        if self.input_shape_ is not None and int(np.prod(self.input_shape_)) != self.n_features_in_:
            raise ValueError(
                "input_shape product must equal the number of feature columns; "
                f"got product {int(np.prod(self.input_shape_))} for {self.n_features_in_} columns."
            )
        self.pooling_ = normalize_pooling(self.pooling if self.pooling is not None else self.model_spec_.default_pooling)
        self.input_layout_ = normalize_input_layout(self.input_layout if self.input_layout is not None else self.model_spec_.default_input_layout)
        self.preprocessing_ = normalize_preprocessing(self.preprocessing if self.preprocessing is not None else self.model_spec_.default_preprocessing)
        self.load_mode_ = normalize_load_mode(self.load_mode if self.load_mode is not None else self.model_spec_.default_load_mode)
        self.batch_size_ = _positive_int(self.batch_size, name="batch_size")
        self.forward_kwargs_ = dict(self.forward_kwargs or {})
        self.encoder_ = self._load_encoder()
        self._freeze_encoder(self.encoder_)
        return self

    def transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "encoder_"):
            raise RuntimeError("FrozenTorchEncoderTransformer must be fitted before transform.")
        x = _validate_feature_matrix(features, name="features")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} feature columns, got {x.shape[1]}.")

        torch = _import_torch()
        tensor_dtype = _torch_dtype(torch, self.dtype)
        encoded_batches: list[np.ndarray] = []
        for start in range(0, x.shape[0], self.batch_size_):
            batch = torch.as_tensor(x[start : start + self.batch_size_], dtype=tensor_dtype, device=self.device)
            batch = self._prepare_batch(torch, batch)
            with torch.no_grad():
                output = self.encoder_(batch, **self.forward_kwargs_)
            output = self._select_output(output)
            output = self._pool_output(torch, output)
            batch_features = output.detach().cpu().numpy()
            batch_features = _ensure_2d_output(batch_features)
            encoded_batches.append(batch_features.astype(np.float32, copy=False))

        transformed = np.concatenate(encoded_batches, axis=0) if encoded_batches else np.empty((0, 0), dtype=np.float32)
        if transformed.ndim != 2:
            raise ValueError("Foundation encoder output must be convertible to a two-dimensional feature matrix.")
        self.n_features_out_ = int(transformed.shape[1])
        return transformed

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        del input_features
        n_features = getattr(self, "n_features_out_", None)
        if n_features is None:
            raise RuntimeError("Feature names are available after transform has determined the encoder output width.")
        family = getattr(getattr(self, "model_spec_", None), "name", "foundation")
        return np.asarray([f"{family}_foundation_{index}" for index in range(n_features)], dtype=object)

    def _load_encoder(self):
        if self.encoder is not None:
            encoder = self.encoder
        elif self.load_mode_ == "factory":
            encoder = self._instantiate_model()
        elif self.model_path is None:
            hint = f" {self.model_spec_.package_hint}" if self.model_spec_.package_hint else ""
            raise ValueError("FrozenTorchEncoderTransformer requires an encoder object, model_path, or model_factory." + hint)
        elif self.load_mode_ in {"torchscript", "jit", "torch_jit"}:
            torch = _import_torch()
            encoder = torch.jit.load(str(self.model_path), map_location=self.device)
        elif self.load_mode_ in {"module", "torch", "pickle", "stateful_module"}:
            torch = _import_torch()
            checkpoint = _torch_load(torch, str(self.model_path), self.device, weights_only=False)
            encoder = _extract_path_or_auto(checkpoint, self.checkpoint_key, _CHECKPOINT_MODULE_AUTO_KEYS)
        elif self.load_mode_ in {"state_dict", "weights", "checkpoint"}:
            encoder = self._load_state_dict_encoder()
        else:  # pragma: no cover - normalize_load_mode should prevent this branch
            raise ValueError(f"Unsupported load_mode: {self.load_mode!r}")

        if self.encoder_attr:
            encoder = _resolve_object_path(encoder, self.encoder_attr)
        return encoder

    def _instantiate_model(self):
        if self.model_factory is None:
            raise ValueError("load_mode='state_dict' or 'factory' requires model_factory or an encoder object.")
        factory = _resolve_import_path(self.model_factory) if isinstance(self.model_factory, str) else self.model_factory
        if not callable(factory):
            raise TypeError("model_factory must be callable or an import path to a callable.")
        return factory(**dict(self.model_kwargs or {}))

    def _load_state_dict_encoder(self):
        torch = _import_torch()
        encoder = self._instantiate_model()
        checkpoint = _torch_load(torch, str(self.model_path), self.device, weights_only=True)
        state_dict = _extract_state_dict(checkpoint, self.state_dict_key)
        state_dict = _strip_state_dict_prefixes(state_dict, self.strip_state_dict_prefix)
        missing_or_unexpected = encoder.load_state_dict(state_dict, strict=bool(self.strict_load))
        self.load_state_dict_result_ = missing_or_unexpected
        return encoder

    def _freeze_encoder(self, encoder: Any) -> None:
        if hasattr(encoder, "to"):
            encoder.to(self.device)
        if hasattr(encoder, "eval"):
            encoder.eval()
        if hasattr(encoder, "parameters"):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

    def _prepare_batch(self, torch: Any, batch: Any) -> Any:
        if self.input_shape_ is not None:
            batch = batch.reshape((batch.shape[0], *self.input_shape_))
        batch = self._apply_input_layout(batch)
        batch = self._apply_preprocessing(torch, batch)
        return batch

    def _apply_input_layout(self, batch: Any) -> Any:
        layout = self.input_layout_
        if layout in {"auto", "none", "channels_first"}:
            return batch
        if layout == "time_first":
            if batch.ndim != 3:
                raise ValueError("input_layout='time_first' expects a 3-D batch after input_shape reshaping.")
            return batch.permute(0, 2, 1)
        if layout == "add_channel_dim":
            if batch.ndim < 2:
                raise ValueError("input_layout='add_channel_dim' expects at least a 2-D batch.")
            return batch.unsqueeze(1)
        if layout == "add_feature_dim":
            if batch.ndim != 3:
                raise ValueError("input_layout='add_feature_dim' expects a 3-D batch after input_shape reshaping.")
            return batch.unsqueeze(2)
        raise ValueError(f"Unknown input_layout: {self.input_layout!r}")

    def _apply_preprocessing(self, torch: Any, batch: Any) -> Any:
        preprocessing = self.preprocessing_
        if preprocessing == "none":
            return batch
        eps = torch.finfo(batch.dtype).eps if getattr(batch, "is_floating_point", lambda: False)() else 1e-6
        if preprocessing == "trial_zscore":
            dims = tuple(range(1, batch.ndim))
            mean = batch.mean(dim=dims, keepdim=True)
            std = batch.std(dim=dims, keepdim=True, unbiased=False).clamp_min(eps)
            return (batch - mean) / std
        if preprocessing == "channel_zscore":
            if batch.ndim < 2:
                raise ValueError("channel_zscore preprocessing expects at least a 2-D batch.")
            axis = -1 if batch.ndim >= 3 else 1
            mean = batch.mean(dim=axis, keepdim=True)
            std = batch.std(dim=axis, keepdim=True, unbiased=False).clamp_min(eps)
            return (batch - mean) / std
        raise ValueError(f"Unknown preprocessing mode: {self.preprocessing!r}")

    def _select_output(self, output: Any) -> Any:
        if self.output_attribute is not None:
            output = _resolve_object_path(output, self.output_attribute)
        elif self.model_spec_.default_output_attribute is not None and hasattr(output, self.model_spec_.default_output_attribute):
            output = getattr(output, self.model_spec_.default_output_attribute)

        if self.output_key is not None:
            if not isinstance(output, Mapping):
                raise ValueError("output_key can only be used when the encoder returns a mapping.")
            output = _resolve_mapping_path(output, self.output_key)
        elif self.model_spec_.default_output_key is not None and isinstance(output, Mapping) and self.model_spec_.default_output_key in output:
            output = output[self.model_spec_.default_output_key]
        elif isinstance(output, Mapping):
            output = _select_mapping_output(output)

        if self.output_index is not None:
            output = output[int(self.output_index)]
        elif isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def _pool_output(self, torch: Any, output: Any) -> Any:
        if not hasattr(output, "ndim"):
            output = torch.as_tensor(output, device=self.device)
        pooling = self.pooling_
        if pooling in {"flatten", "identity"}:
            return output
        if pooling in {"mean", "mean_time"}:
            if output.ndim <= 2:
                return output
            return output.mean(dim=-1)
        if pooling == "mean_tokens":
            if output.ndim <= 2:
                return output
            return output.mean(dim=1)
        if pooling == "mean_channels":
            if output.ndim <= 2:
                return output
            return output.mean(dim=1)
        if pooling == "cls":
            if output.ndim < 3:
                raise ValueError("pooling='cls' expects a sequence-shaped encoder output with at least three dimensions.")
            return output[:, 0, ...]
        if pooling == "last":
            if output.ndim < 3:
                raise ValueError("pooling='last' expects a sequence-shaped encoder output with at least three dimensions.")
            return output[:, -1, ...]
        raise ValueError(f"Unknown pooling mode: {self.pooling!r}")


def list_foundation_model_families() -> tuple[str, ...]:
    """Return canonical foundation-model family names supported by NeuRepTrace wrappers."""

    return tuple(FOUNDATION_MODEL_SPECS)


def normalize_foundation_model_family(model_family: str | None) -> str:
    """Normalize aliases for foundation-model families."""

    if model_family is None:
        return "generic"
    normalized = str(model_family).strip().lower().replace("_", "-")
    if normalized in {"", "none", "default"}:
        return "generic"
    try:
        return _FOUNDATION_MODEL_ALIASES[normalized]
    except KeyError as exc:
        families = ", ".join(list_foundation_model_families())
        raise ValueError(f"Unknown foundation model family '{model_family}'. Available families: {families}.") from exc


def get_foundation_model_spec(model_family: str | None) -> FoundationModelSpec:
    """Return the spec for a foundation-model family or alias."""

    return FOUNDATION_MODEL_SPECS[normalize_foundation_model_family(model_family)]


def foundation_model_defaults(model_family: str | None) -> dict[str, Any]:
    """Return default classifier parameters for a model family."""

    return get_foundation_model_spec(model_family).defaults()


def normalize_pooling(pooling: str | None) -> str:
    """Normalize foundation-encoder pooling names."""

    normalized = "flatten" if pooling is None else str(pooling).strip().lower().replace("-", "_")
    aliases = {
        "flat": "flatten",
        "ravel": "flatten",
        "none": "identity",
        "token_mean": "mean_tokens",
        "tokens_mean": "mean_tokens",
        "sequence_mean": "mean_tokens",
        "mean_sequence": "mean_tokens",
        "time_mean": "mean_time",
        "mean_tokens": "mean_tokens",
        "class_token": "cls",
        "cls_token": "cls",
        "final": "last",
        "last_token": "last",
        "channel_mean": "mean_channels",
        "channels_mean": "mean_channels",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in POOLING_CHOICES:
        raise ValueError(f"Unknown pooling mode '{pooling}'. Available modes: {', '.join(POOLING_CHOICES)}.")
    return normalized


def normalize_input_layout(input_layout: str | None) -> str:
    """Normalize input-layout names for foundation encoders."""

    normalized = "auto" if input_layout is None else str(input_layout).strip().lower().replace("-", "_")
    aliases = {
        "flat": "auto",
        "identity": "auto",
        "batch_channel_time": "channels_first",
        "batch_channels_time": "channels_first",
        "bct": "channels_first",
        "channels_first": "channels_first",
        "channel_first": "channels_first",
        "batch_time_channel": "time_first",
        "batch_time_channels": "time_first",
        "btc": "time_first",
        "time_first": "time_first",
        "time_major": "time_first",
        "4d": "add_channel_dim",
        "add_channel": "add_channel_dim",
        "add_channel_axis": "add_channel_dim",
        "unsqueeze_channel": "add_channel_dim",
        "b1ct": "add_channel_dim",
        "add_feature": "add_feature_dim",
        "add_feature_axis": "add_feature_dim",
        "unsqueeze_feature": "add_feature_dim",
        "bc1t": "add_feature_dim",
        "bendr": "channels_first",
        "labram": "channels_first",
        "eegpt": "channels_first",
        "cbramod": "channels_first",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in INPUT_LAYOUT_CHOICES:
        raise ValueError(f"Unknown input_layout '{input_layout}'. Available layouts: {', '.join(INPUT_LAYOUT_CHOICES)}.")
    return normalized


def normalize_preprocessing(preprocessing: str | None) -> str:
    """Normalize stateless foundation-input preprocessing modes."""

    normalized = "none" if preprocessing is None else str(preprocessing).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "no": "none",
        "zscore": "trial_zscore",
        "z_score": "trial_zscore",
        "per_trial_zscore": "trial_zscore",
        "trial_z_score": "trial_zscore",
        "channel_z_score": "channel_zscore",
        "per_channel_zscore": "channel_zscore",
        "per_channel_z_score": "channel_zscore",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PREPROCESSING_CHOICES:
        raise ValueError(f"Unknown preprocessing mode '{preprocessing}'. Available modes: {', '.join(PREPROCESSING_CHOICES)}.")
    return normalized


def normalize_load_mode(load_mode: str | None) -> str:
    """Normalize foundation-model loading mode names."""

    normalized = "torchscript" if load_mode is None else str(load_mode).strip().lower().replace("-", "_")
    aliases = {
        "jit": "torchscript",
        "torch_jit": "torchscript",
        "torchscript_module": "torchscript",
        "torch_module": "module",
        "full_module": "module",
        "pickle": "module",
        "stateful_module": "module",
        "weights": "state_dict",
        "checkpoint": "state_dict",
        "ckpt": "state_dict",
        "factory_only": "factory",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"torchscript", "module", "state_dict", "factory"}:
        raise ValueError("load_mode must be one of 'torchscript', 'module', 'state_dict', or 'factory'.")
    return normalized


def parse_input_shape(input_shape: Sequence[int] | str | None) -> tuple[int, ...] | None:
    """Parse an encoder input-shape specification."""

    if input_shape is None:
        return None
    if isinstance(input_shape, str):
        stripped = input_shape.strip()
        if not stripped or stripped.lower() in {"none", "flat", "flatten", "auto"}:
            return None
        for separator in ("x", "X", ";", " "):
            stripped = stripped.replace(separator, ",")
        values = [value for value in stripped.split(",") if value]
    else:
        values = list(input_shape)
    shape = tuple(_positive_int(value, name="input_shape") for value in values)
    if not shape:
        return None
    return shape


def normalize_foundation_linear_probe_params(classifier_param: Any) -> dict[str, Any]:
    """Normalize a frozen-foundation-encoder linear-probe configuration."""

    if classifier_param is None:
        params: dict[str, Any] = {}
    elif isinstance(classifier_param, Mapping):
        params = dict(classifier_param)
    elif isinstance(classifier_param, (str, Path)):
        params = {"model_path": str(classifier_param)}
    else:
        params = {"C": classifier_param}

    aliases = {
        "path": "model_path",
        "checkpoint": "model_path",
        "checkpoint_path": "model_path",
        "weights_path": "model_path",
        "encoder_path": "model_path",
        "shape": "input_shape",
        "layout": "input_layout",
        "pool": "pooling",
        "standardize_before_encoder": "pre_encoder_standardize",
        "pre_standardize": "pre_encoder_standardize",
        "preprocess": "preprocessing",
        "head": "probe",
        "family": "model_family",
        "foundation_family": "model_family",
        "foundation_model": "model_family",
        "model_name": "model_family",
        "factory": "model_factory",
        "model_class": "model_factory",
        "factory_kwargs": "model_kwargs",
        "kwargs": "model_kwargs",
        "module_attr": "encoder_attr",
        "feature_extractor_attr": "encoder_attr",
        "output_attr": "output_attribute",
        "state_key": "state_dict_key",
        "state_dict": "state_dict_key",
        "trusted_load": "load_mode",
    }
    for alias, canonical in aliases.items():
        if alias in params and canonical not in params:
            params[canonical] = params.pop(alias)

    model_family = normalize_foundation_model_family(params.get("model_family"))
    family_defaults = foundation_model_defaults(model_family)
    normalized = {**DEFAULT_FOUNDATION_LINEAR_PROBE_PARAMS, **family_defaults, **params}
    normalized["model_family"] = model_family
    normalized["pooling"] = normalize_pooling(normalized.get("pooling"))
    normalized["input_layout"] = normalize_input_layout(normalized.get("input_layout"))
    normalized["preprocessing"] = normalize_preprocessing(normalized.get("preprocessing"))
    normalized["load_mode"] = normalize_load_mode(normalized.get("load_mode"))
    normalized["input_shape"] = parse_input_shape(normalized.get("input_shape"))
    normalized["batch_size"] = _positive_int(normalized.get("batch_size"), name="batch_size")
    normalized["max_iter"] = _positive_int(normalized.get("max_iter"), name="max_iter")
    normalized["C"] = _positive_float(normalized.get("C"), name="C")
    normalized["pre_encoder_standardize"] = _bool_param(normalized.get("pre_encoder_standardize"), name="pre_encoder_standardize")
    normalized["strict_load"] = _bool_param(normalized.get("strict_load"), name="strict_load")
    if normalized.get("model_kwargs") is None:
        normalized["model_kwargs"] = None
    elif not isinstance(normalized.get("model_kwargs"), Mapping):
        raise ValueError("model_kwargs must be a mapping when provided.")
    else:
        normalized["model_kwargs"] = dict(normalized["model_kwargs"])
    if normalized.get("forward_kwargs") is None:
        normalized["forward_kwargs"] = None
    elif not isinstance(normalized.get("forward_kwargs"), Mapping):
        raise ValueError("forward_kwargs must be a mapping when provided.")
    else:
        normalized["forward_kwargs"] = dict(normalized["forward_kwargs"])
    probe = str(normalized.get("probe", "logistic")).strip().lower().replace("-", "_")
    if probe in {"svm", "linear_svc", "linear_support_vector_machine"}:
        probe = "linear_svm"
    if probe not in PROBE_CHOICES:
        raise ValueError(f"Unknown foundation probe '{normalized.get('probe')}'. Available probes: {', '.join(PROBE_CHOICES)}.")
    normalized["probe"] = probe
    return normalized


def make_foundation_linear_probe(
    classifier_param: Mapping[str, Any] | str | Path | None,
    *,
    max_iter: int = 1000,
    random_state: int | None = 13,
):
    """Create a frozen-foundation-encoder probe pipeline."""

    params = normalize_foundation_linear_probe_params(classifier_param)
    probe_max_iter = int(params.get("max_iter") or max_iter)
    steps: list[tuple[str, Any]] = []
    if params["pre_encoder_standardize"]:
        steps.append(("pre_encoder_standardscaler", StandardScaler()))
    steps.extend(
        [
            (
                "frozen_torch_encoder",
                FrozenTorchEncoderTransformer(
                    model_family=params.get("model_family"),
                    encoder=params.get("encoder"),
                    model_path=params.get("model_path"),
                    model_factory=params.get("model_factory"),
                    model_kwargs=params.get("model_kwargs"),
                    input_shape=params.get("input_shape"),
                    input_layout=params.get("input_layout"),
                    preprocessing=params.get("preprocessing"),
                    batch_size=params.get("batch_size"),
                    device=params.get("device", "cpu"),
                    output_key=params.get("output_key"),
                    output_attribute=params.get("output_attribute"),
                    output_index=params.get("output_index"),
                    pooling=params.get("pooling", "flatten"),
                    load_mode=params.get("load_mode", "torchscript"),
                    dtype=params.get("dtype", "float32"),
                    checkpoint_key=params.get("checkpoint_key"),
                    state_dict_key=params.get("state_dict_key"),
                    strict_load=params.get("strict_load", True),
                    strip_state_dict_prefix=params.get("strip_state_dict_prefix"),
                    encoder_attr=params.get("encoder_attr"),
                    forward_kwargs=params.get("forward_kwargs"),
                ),
            ),
            ("foundation_standardscaler", StandardScaler()),
            ("foundation_probe", _make_probe(params, max_iter=probe_max_iter, random_state=random_state)),
        ]
    )
    return make_pipeline(*(step for _, step in steps))


def make_bendr_linear_probe(classifier_param: Mapping[str, Any] | str | Path | None = None, **overrides: Any):
    """Create a frozen BENDR-family linear-probe pipeline."""

    return make_foundation_linear_probe(_with_model_family("bendr", classifier_param, overrides))


def make_labram_linear_probe(classifier_param: Mapping[str, Any] | str | Path | None = None, **overrides: Any):
    """Create a frozen LaBraM-family linear-probe pipeline."""

    return make_foundation_linear_probe(_with_model_family("labram", classifier_param, overrides))


def make_eegpt_linear_probe(classifier_param: Mapping[str, Any] | str | Path | None = None, **overrides: Any):
    """Create a frozen EEGPT-family linear-probe pipeline."""

    return make_foundation_linear_probe(_with_model_family("eegpt", classifier_param, overrides))


def make_cbramod_linear_probe(classifier_param: Mapping[str, Any] | str | Path | None = None, **overrides: Any):
    """Create a frozen CBraMod-family linear-probe pipeline."""

    return make_foundation_linear_probe(_with_model_family("cbramod", classifier_param, overrides))


def fit_foundation_linear_probe(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    classifier_param: Mapping[str, Any] | str | Path | None,
    random_state: int | None = None,
):
    """Fit a frozen-foundation-encoder probe on labeled source features."""

    params = normalize_foundation_linear_probe_params(classifier_param)
    model = make_foundation_linear_probe(params, max_iter=int(params["max_iter"]), random_state=random_state)
    model.fit(features, labels)
    return model


def register_foundation_linear_probe() -> None:
    """Register foundation-model linear probes as optional decoder names."""

    from neureptrace.decoding.classifiers import CLASSIFIER_REGISTRY, DEFAULT_CLASSIFIER_PARAMS, ClassifierSpec

    DEFAULT_CLASSIFIER_PARAMS.setdefault("foundation-linear-probe", normalize_foundation_linear_probe_params(None))
    CLASSIFIER_REGISTRY["foundation-linear-probe"] = ClassifierSpec(_build_foundation_linear_probe_classifier, fits_in_builder=True)
    for family, decoder_name in FAMILY_LINEAR_PROBE_DECODER_NAMES.items():
        DEFAULT_CLASSIFIER_PARAMS.setdefault(decoder_name, normalize_foundation_linear_probe_params({"model_family": family}))
        CLASSIFIER_REGISTRY[decoder_name] = ClassifierSpec(_family_builder(family), fits_in_builder=True)
    _refresh_decoding_choices()


def _build_foundation_linear_probe_classifier(features: np.ndarray, labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return fit_foundation_linear_probe(features, labels, classifier_param, random_state=random_state)


def _family_builder(family: str):
    def _build(features: np.ndarray, labels: np.ndarray, classifier_param: Any, random_state: int | None):
        return fit_foundation_linear_probe(features, labels, _with_model_family(family, classifier_param, {}), random_state=random_state)

    return _build


def _with_model_family(family: str, classifier_param: Mapping[str, Any] | str | Path | None, overrides: Mapping[str, Any]) -> dict[str, Any]:
    if classifier_param is None:
        params: dict[str, Any] = {}
    elif isinstance(classifier_param, Mapping):
        params = dict(classifier_param)
    elif isinstance(classifier_param, (str, Path)):
        params = {"model_path": str(classifier_param)}
    else:
        params = {"C": classifier_param}
    params.update(dict(overrides))
    params.setdefault("model_family", family)
    return params


def _refresh_decoding_choices() -> None:
    try:
        import neureptrace.decoding as decoding
    except ImportError:
        return
    if hasattr(decoding, "_registry_decoder_lookup"):
        decoding._REGISTRY_DECODER_LOOKUP = decoding._registry_decoder_lookup()
    if all(hasattr(decoding, name) for name in ("BUILTIN_DECODER_CHOICES", "CLASSIFIER_REGISTRY", "DECODER_ALIASES")):
        fold_aware = getattr(decoding, "FOLD_AWARE_DECODER_CHOICES", ())
        fold_aliases = getattr(decoding, "FOLD_AWARE_DECODER_ALIASES", ())
        decoding.DECODER_CHOICES = tuple(
            dict.fromkeys(
                (
                    *decoding.BUILTIN_DECODER_CHOICES,
                    *fold_aware,
                    *decoding.CLASSIFIER_REGISTRY.keys(),
                    *decoding.DECODER_ALIASES,
                    *fold_aliases,
                )
            )
        )
        decoding.DECODER_CLI_CHOICES = tuple(
            dict.fromkeys(
                (
                    *decoding.BUILTIN_DECODER_CHOICES,
                    *decoding.CLASSIFIER_REGISTRY.keys(),
                    *decoding.DECODER_ALIASES,
                )
            )
        )


def _make_probe(params: Mapping[str, Any], *, max_iter: int, random_state: int | None):
    probe = params["probe"]
    class_weight = params.get("class_weight", "balanced")
    if str(class_weight).strip().lower() in {"none", "false", "off", ""}:
        class_weight = None
    if probe == "logistic":
        return LogisticRegression(C=float(params["C"]), class_weight=class_weight, max_iter=max_iter, random_state=random_state, solver="lbfgs")
    if probe == "linear_svm":
        return LinearSVC(C=float(params["C"]), class_weight=class_weight, max_iter=max_iter, random_state=random_state)
    if probe == "ridge":
        return RidgeClassifier(alpha=1.0 / float(params["C"]), class_weight=class_weight, max_iter=max_iter)
    if probe == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    raise ValueError(f"Unknown foundation probe: {probe!r}")


def _validate_feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values.")
    return x


def _ensure_2d_output(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        return values.reshape(1, 1)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim == 2:
        return values
    return values.reshape(values.shape[0], -1)


def _resolve_import_path(path_or_callable: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(path_or_callable):
        return path_or_callable
    path = str(path_or_callable).strip()
    if not path:
        raise ValueError("model_factory import path must be non-empty.")
    if ":" in path:
        module_name, attr_path = path.split(":", 1)
    else:
        module_name, _, attr_path = path.rpartition(".")
    if not module_name or not attr_path:
        raise ValueError("model_factory must use 'package.module:Factory' or 'package.module.Factory' notation.")
    obj: Any = import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if not callable(obj):
        raise TypeError(f"Imported model_factory '{path}' is not callable.")
    return obj


def _extract_state_dict(checkpoint: Any, state_dict_key: str | Sequence[str] | None) -> Mapping[str, Any]:
    candidate = _extract_path_or_auto(checkpoint, state_dict_key, _STATE_DICT_AUTO_KEYS)
    if isinstance(candidate, Mapping) and candidate and all(hasattr(value, "shape") for value in candidate.values()):
        return candidate
    if isinstance(candidate, Mapping):
        tensor_like = {key: value for key, value in candidate.items() if hasattr(value, "shape")}
        if tensor_like and len(tensor_like) == len(candidate):
            return tensor_like
    raise ValueError("Could not extract a tensor state_dict from checkpoint. Set state_dict_key to the correct checkpoint entry.")


def _extract_path_or_auto(obj: Any, key_or_path: str | Sequence[str] | None, auto_keys: Sequence[str]) -> Any:
    if key_or_path is not None:
        return _resolve_object_path(obj, key_or_path)
    if isinstance(obj, Mapping):
        for key in auto_keys:
            if key in obj:
                return obj[key]
    return obj


def _strip_state_dict_prefixes(state_dict: Mapping[str, Any], prefixes: str | Sequence[str] | bool | None) -> dict[str, Any]:
    cleaned = dict(state_dict)
    if prefixes is None or prefixes is False:
        return cleaned
    if prefixes is True:
        prefix_values: tuple[str, ...] = ("module.", "model.", "encoder.")
    elif isinstance(prefixes, str):
        prefix_values = (prefixes,)
    else:
        prefix_values = tuple(str(prefix) for prefix in prefixes)
    for prefix in prefix_values:
        if prefix and cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
    return cleaned


def _resolve_object_path(obj: Any, path: str | Sequence[str]) -> Any:
    current = obj
    for part in _path_parts(path):
        if isinstance(current, Mapping):
            current = current[part]
        elif isinstance(current, (tuple, list)) and str(part).lstrip("-").isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, str(part))
    return current


def _resolve_mapping_path(mapping: Mapping[str, Any], path: str | Sequence[str]) -> Any:
    current: Any = mapping
    for part in _path_parts(path):
        if not isinstance(current, Mapping):
            raise ValueError(f"Cannot resolve mapping path {path!r}; {part!r} is below a non-mapping object.")
        current = current[part]
    return current


def _path_parts(path: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(path, str):
        delimiter = "/" if "/" in path else "."
        return tuple(part for part in path.split(delimiter) if part)
    return tuple(str(part) for part in path)


def _select_mapping_output(output: Mapping[str, Any]) -> Any:
    for key in ("features", "feature", "embeddings", "embedding", "latent", "last_hidden_state", "pooler_output", "x"):
        if key in output:
            return output[key]
    keys = ", ".join(str(key) for key in output)
    raise ValueError("Encoder returned a mapping; set output_key to one of its entries. Available keys: " + keys)


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without optional torch extra
        raise ImportError("Foundation-model feature extraction requires the optional torch extra, e.g. `pip install neureptrace[torch]`.") from exc
    return torch


def _torch_load(torch: Any, path: str, device: str, *, weights_only: bool | None) -> Any:
    kwargs: dict[str, Any] = {"map_location": device}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        if "weights_only" in kwargs:
            kwargs.pop("weights_only")
            return torch.load(path, **kwargs)
        raise


def _torch_dtype(torch: Any, dtype: str | Any):
    if not isinstance(dtype, str):
        return dtype
    normalized = dtype.strip().lower()
    if not hasattr(torch, normalized):
        raise ValueError(f"Unknown torch dtype '{dtype}'.")
    return getattr(torch, normalized)


def _bool_param(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "none", ""}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(number) or number % 1.0 != 0.0 or number < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(number)


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return number
