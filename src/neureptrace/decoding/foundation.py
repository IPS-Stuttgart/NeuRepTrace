"""Frozen foundation-model feature extraction for M/EEG decoders.

The classes in this module provide a light-weight integration point for
BENDR-, LaBraM-, EEGPT-, CBraMod-, or project-local encoders without making any
of those packages mandatory NeuRepTrace dependencies.  Encoders are used as
frozen feature extractors; downstream probes are trained with ordinary
NeuRepTrace source-label workflows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

POOLING_CHOICES = ("flatten", "identity", "mean", "mean_time", "cls", "last")
PROBE_CHOICES = ("logistic", "linear_svm", "ridge", "lda")

DEFAULT_FOUNDATION_LINEAR_PROBE_PARAMS: dict[str, Any] = {
    "probe": "logistic",
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "batch_size": 128,
    "device": "cpu",
    "pooling": "flatten",
    "load_mode": "torchscript",
    "dtype": "float32",
    "pre_encoder_standardize": False,
}


class FrozenTorchEncoderTransformer(TransformerMixin, BaseEstimator):
    """Transform feature rows with a frozen PyTorch encoder.

    Parameters
    ----------
    encoder:
        Optional already-constructed ``torch.nn.Module`` or callable. This is
        mainly intended for Python API usage and tests.
    model_path:
        Optional path to a serialized encoder. The default ``load_mode`` expects
        a TorchScript module loadable with ``torch.jit.load``.
    input_shape:
        Optional per-trial tensor shape used to reshape each flattened feature
        row before calling the encoder. For example, ``(n_channels, n_times)``.
        The product must equal the number of input columns.
    batch_size:
        Number of rows encoded per forward pass.
    device:
        Torch device passed to the encoder and input batches.
    output_key, output_index:
        Optional selectors for encoders returning dictionaries, tuples, or
        lists. ``output_key`` is applied before ``output_index``.
    pooling:
        How to convert encoder outputs to a two-dimensional feature matrix.
        ``flatten`` and ``identity`` preserve all non-batch dimensions by
        flattening them. ``mean``/``mean_time`` average the last output axis,
        while ``cls`` and ``last`` select token 0 or the final token from
        sequence-shaped outputs.
    load_mode:
        ``torchscript`` loads ``model_path`` with ``torch.jit.load``. ``module``
        loads it with ``torch.load`` and is therefore appropriate only for
        trusted checkpoints.
    dtype:
        Torch dtype name used for input batches, normally ``float32``.
    """

    def __init__(
        self,
        *,
        encoder: Any | None = None,
        model_path: str | Path | None = None,
        input_shape: Sequence[int] | str | None = None,
        batch_size: int = 128,
        device: str = "cpu",
        output_key: str | None = None,
        output_index: int | None = None,
        pooling: str = "flatten",
        load_mode: str = "torchscript",
        dtype: str = "float32",
    ):
        self.encoder = encoder
        self.model_path = model_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.device = device
        self.output_key = output_key
        self.output_index = output_index
        self.pooling = pooling
        self.load_mode = load_mode
        self.dtype = dtype

    def fit(self, features: Sequence[Sequence[float]] | np.ndarray, labels: Sequence | np.ndarray | None = None):
        del labels
        x = _validate_feature_matrix(features, name="features")
        self.n_features_in_ = int(x.shape[1])
        self.input_shape_ = parse_input_shape(self.input_shape)
        if self.input_shape_ is not None and int(np.prod(self.input_shape_)) != self.n_features_in_:
            raise ValueError(
                "input_shape product must equal the number of feature columns; "
                f"got product {int(np.prod(self.input_shape_))} for {self.n_features_in_} columns."
            )
        self.pooling_ = normalize_pooling(self.pooling)
        self.batch_size_ = _positive_int(self.batch_size, name="batch_size")
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
            if self.input_shape_ is not None:
                batch = batch.reshape((batch.shape[0], *self.input_shape_))
            with torch.no_grad():
                output = self.encoder_(batch)
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
        return np.asarray([f"foundation_{index}" for index in range(n_features)], dtype=object)

    def _load_encoder(self):
        if self.encoder is not None:
            return self.encoder
        if self.model_path is None:
            raise ValueError("FrozenTorchEncoderTransformer requires either an encoder object or a model_path.")
        torch = _import_torch()
        model_path = str(self.model_path)
        load_mode = str(self.load_mode).strip().lower().replace("-", "_")
        if load_mode in {"torchscript", "jit", "torch_jit"}:
            return torch.jit.load(model_path, map_location=self.device)
        if load_mode in {"module", "torch", "pickle", "stateful_module"}:
            return torch.load(model_path, map_location=self.device)
        raise ValueError("load_mode must be 'torchscript' or 'module'.")

    def _freeze_encoder(self, encoder: Any) -> None:
        if hasattr(encoder, "to"):
            encoder.to(self.device)
        if hasattr(encoder, "eval"):
            encoder.eval()
        if hasattr(encoder, "parameters"):
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

    def _select_output(self, output: Any) -> Any:
        if self.output_key is not None:
            if not isinstance(output, Mapping):
                raise ValueError("output_key can only be used when the encoder returns a mapping.")
            output = output[self.output_key]
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
        if pooling == "cls":
            if output.ndim < 3:
                raise ValueError("pooling='cls' expects a sequence-shaped encoder output with at least three dimensions.")
            return output[:, 0, ...]
        if pooling == "last":
            if output.ndim < 3:
                raise ValueError("pooling='last' expects a sequence-shaped encoder output with at least three dimensions.")
            return output[:, -1, ...]
        raise ValueError(f"Unknown pooling mode: {self.pooling!r}")


def normalize_pooling(pooling: str | None) -> str:
    """Normalize foundation-encoder pooling names."""

    normalized = "flatten" if pooling is None else str(pooling).strip().lower().replace("-", "_")
    aliases = {
        "flat": "flatten",
        "ravel": "flatten",
        "none": "identity",
        "token_mean": "mean_time",
        "time_mean": "mean_time",
        "mean_tokens": "mean_time",
        "class_token": "cls",
        "cls_token": "cls",
        "final": "last",
        "last_token": "last",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in POOLING_CHOICES:
        raise ValueError(f"Unknown pooling mode '{pooling}'. Available modes: {', '.join(POOLING_CHOICES)}.")
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
        "encoder_path": "model_path",
        "shape": "input_shape",
        "pool": "pooling",
        "standardize_before_encoder": "pre_encoder_standardize",
        "pre_standardize": "pre_encoder_standardize",
        "head": "probe",
    }
    for alias, canonical in aliases.items():
        if alias in params and canonical not in params:
            params[canonical] = params.pop(alias)

    normalized = {**DEFAULT_FOUNDATION_LINEAR_PROBE_PARAMS, **params}
    normalized["pooling"] = normalize_pooling(normalized.get("pooling"))
    normalized["input_shape"] = parse_input_shape(normalized.get("input_shape"))
    normalized["batch_size"] = _positive_int(normalized.get("batch_size"), name="batch_size")
    normalized["max_iter"] = _positive_int(normalized.get("max_iter"), name="max_iter")
    normalized["C"] = _positive_float(normalized.get("C"), name="C")
    normalized["pre_encoder_standardize"] = bool(normalized.get("pre_encoder_standardize"))
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
    """Create a frozen-foundation-encoder probe pipeline.

    The returned estimator can be used directly or through the optional
    ``foundation-linear-probe`` registry hook installed by
    :func:`register_foundation_linear_probe`.
    """

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
                    encoder=params.get("encoder"),
                    model_path=params.get("model_path"),
                    input_shape=params.get("input_shape"),
                    batch_size=params.get("batch_size"),
                    device=params.get("device", "cpu"),
                    output_key=params.get("output_key"),
                    output_index=params.get("output_index"),
                    pooling=params.get("pooling", "flatten"),
                    load_mode=params.get("load_mode", "torchscript"),
                    dtype=params.get("dtype", "float32"),
                ),
            ),
            ("foundation_standardscaler", StandardScaler()),
            ("foundation_probe", _make_probe(params, max_iter=probe_max_iter, random_state=random_state)),
        ]
    )
    return make_pipeline(*(step for _, step in steps))


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
    """Register ``foundation-linear-probe`` as an optional decoder name.

    Registration is explicit so normal NeuRepTrace imports remain free of any
    foundation-model or PyTorch dependency.  The registered decoder uses the
    same classifier-registry path as other legacy decoders.
    """

    from neureptrace.decoding.classifiers import CLASSIFIER_REGISTRY, DEFAULT_CLASSIFIER_PARAMS, ClassifierSpec

    DEFAULT_CLASSIFIER_PARAMS.setdefault("foundation-linear-probe", DEFAULT_FOUNDATION_LINEAR_PROBE_PARAMS.copy())
    CLASSIFIER_REGISTRY["foundation-linear-probe"] = ClassifierSpec(_build_foundation_linear_probe_classifier, fits_in_builder=True)
    _refresh_decoding_choices()


def _build_foundation_linear_probe_classifier(features: np.ndarray, labels: np.ndarray, classifier_param: Any, random_state: int | None):
    return fit_foundation_linear_probe(features, labels, classifier_param, random_state=random_state)


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
        return LogisticRegression(
            C=float(params["C"]),
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
    if probe == "linear_svm":
        return LinearSVC(
            C=float(params["C"]),
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
    if probe == "ridge":
        return RidgeClassifier(
            alpha=1.0 / float(params["C"]),
            class_weight=class_weight,
            max_iter=max_iter,
        )
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


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without optional torch extra
        raise ImportError(
            "Foundation-model feature extraction requires the optional torch extra, e.g. `pip install neureptrace[torch]`."
        ) from exc
    return torch


def _torch_dtype(torch: Any, dtype: str | Any):
    if not isinstance(dtype, str):
        return dtype
    normalized = dtype.strip().lower()
    if not hasattr(torch, normalized):
        raise ValueError(f"Unknown torch dtype '{dtype}'.")
    return getattr(torch, normalized)


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
