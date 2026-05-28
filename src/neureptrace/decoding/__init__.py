from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from neureptrace.decoding.classifiers import (
    CLASSIFIER_REGISTRY,
    get_default_classifier_param,
    train_multiclass_classifier,
)
from neureptrace.decoding.sampling import (
    CLASS_LIMIT_SELECTION_MODES as CLASS_LIMIT_SELECTION_MODES,
    DEFAULT_CLASS_LIMIT_SEED as DEFAULT_CLASS_LIMIT_SEED,
    DEFAULT_CLASS_LIMIT_SELECTION as DEFAULT_CLASS_LIMIT_SELECTION,
    normalize_class_limit_seed as normalize_class_limit_seed,
    normalize_class_limit_selection as normalize_class_limit_selection,
    select_class_limited_indices as select_class_limited_indices,
)
from neureptrace.metrics import brier_score_multiclass, expected_calibration_error

BUILTIN_DECODER_CHOICES = (
    "logistic",
    "sparse_logistic",
    "elastic_net_logistic",
    "ridge",
    "gaussian_nb",
    "lda",
    "shrinkage_lda",
    "linear_svm",
    "ovo_linear_svm",
    "ecoc_linear_svm",
    "torch_mlp",
)
DECODER_ALIASES = (
    "l1-logistic",
    "logistic-l1",
    "sparse-logreg",
    "elasticnet-logistic",
    "logistic-elastic-net",
    "elastic-net-logreg",
    "nb",
    "naive-bayes",
    "gaussian-naive-bayes",
    "svm",
    "linear-svm",
    "lda-shrinkage",
    "shrinkage-lda",
    "one-vs-one-linear-svm",
    "onevsone-linear-svm",
    "ovo-linear-svm",
    "ovo-svm",
    "ecoc-svm",
    "output-code-linear-svm",
    "outputcode-linear-svm",
    "deep-mlp",
    "shallow-torch-mlp",
)
DECODER_CHOICES = tuple(
    dict.fromkeys(
        (
            *BUILTIN_DECODER_CHOICES,
            *CLASSIFIER_REGISTRY.keys(),
            *DECODER_ALIASES,
        )
    )
)
DECODER_CLI_CHOICES = DECODER_CHOICES
EMISSION_MODE_CHOICES = ("calibrated", "uncalibrated")
FEATURE_PREPROCESSOR_CHOICES = ("none", "pca", "pca_whiten", "anova_select", "pls_da")
TUNING_SCORING_CHOICES = ("accuracy", "balanced_accuracy", "neg_log_loss", "neg_brier", "neg_ece")
DEFAULT_TUNING_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_ANOVA_SELECT_PERCENTILE = 20
ANOVA_SELECT_PERCENTILE_GRID = (10, 20, 40, 60)
DEFAULT_PLS_COMPONENTS = 16
PLS_COMPONENT_GRID = (8, 16, 32, 48)
DEFAULT_ELASTIC_NET_L1_RATIO = 0.5
ELASTIC_NET_L1_RATIO_GRID = (0.15, 0.5, 0.85)
DEFAULT_TUNING_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_TUNING_VAR_SMOOTHING_GRID = (1e-12, 1e-10, 1e-9, 1e-8, 1e-6)


class PCA(SklearnPCA):
    """PCA that caps explicit component counts to the current training fold.

    Cross-subject MEG folds can become smaller than a requested PCA dimension,
    especially inside nested calibration or small OpenNeuro smoke runs. Sklearn's
    PCA raises in that case; this subclass keeps the public ``n_components``
    parameter unchanged for provenance and grid-search names, but uses the
    largest feasible integer component count during each fit.
    """

    def _fit(self, X):
        requested_n_components = self.n_components
        effective_n_components = requested_n_components
        if isinstance(requested_n_components, (int, np.integer)) and not isinstance(requested_n_components, bool):
            n_samples, n_features = X.shape
            effective_n_components = min(int(requested_n_components), max(1, min(int(n_samples), int(n_features))))

        self.n_components = effective_n_components
        try:
            result = super()._fit(X)
        except Exception:
            self.n_components = requested_n_components
            raise
        self.n_components = requested_n_components
        self.requested_n_components_ = requested_n_components
        self.effective_n_components_ = getattr(self, "n_components_", effective_n_components)
        return result


def _positive_float_classifier_param(
    classifier_param: Any,
    *,
    default: float,
    name: str,
) -> float:
    value = default if classifier_param is None else float(classifier_param)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _registry_decoder_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for registry_name in CLASSIFIER_REGISTRY:
        for alias in {
            registry_name,
            registry_name.lower(),
            registry_name.replace("-", "_"),
            registry_name.lower().replace("-", "_"),
        }:
            lookup[alias] = registry_name
    return lookup


_REGISTRY_DECODER_LOOKUP = _registry_decoder_lookup()


def _normalize_registry_decoder_name_or_none(name: str) -> str | None:
    raw = str(name).strip()
    candidates = (
        raw,
        raw.lower(),
        raw.replace("_", "-"),
        raw.lower().replace("_", "-"),
        raw.replace("-", "_"),
        raw.lower().replace("-", "_"),
    )
    for candidate in candidates:
        if candidate in _REGISTRY_DECODER_LOOKUP:
            return _REGISTRY_DECODER_LOOKUP[candidate]
    return None


def normalize_registry_decoder_name(name: str) -> str:
    """Normalize aliases for classifier-registry decoders."""

    normalized = _normalize_registry_decoder_name_or_none(name)
    if normalized is None:
        supported = ", ".join(sorted(CLASSIFIER_REGISTRY))
        raise ValueError(f"Unknown registry decoder '{name}'. Available registry decoders: {supported}.")
    return normalized


class PLSDiscriminantTransformer(TransformerMixin, BaseEstimator):
    """Supervised PLS-DA feature projection for high-dimensional M/EEG windows.

    The transformer maps class labels to one-hot targets and fits a
    ``PLSRegression`` model on the training fold only.  Its output is the PLS
    X-score matrix, which can then be consumed by the existing sklearn
    classifiers.  This gives the BUSH-MEG pipelines a supervised dimensionality
    reduction option without changing outer LOSO semantics.
    """

    def __init__(self, n_components: int | str | None = DEFAULT_PLS_COMPONENTS):
        self.n_components = n_components

    def fit(self, features: Sequence[Sequence[float]] | np.ndarray, labels: Sequence | np.ndarray):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError("PLSDiscriminantTransformer expects a two-dimensional feature matrix.")
        if x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError("PLSDiscriminantTransformer needs at least two samples and one feature.")
        y_raw = np.asarray(labels)
        if y_raw.shape[0] != x.shape[0]:
            raise ValueError("features and labels must contain the same number of rows.")
        self.classes_, encoded = np.unique(y_raw, return_inverse=True)
        if self.classes_.shape[0] < 2:
            raise ValueError("PLSDiscriminantTransformer needs at least two classes.")

        requested = normalize_pls_components(self.n_components)
        max_components = max(1, min(int(x.shape[1]), int(x.shape[0]) - 1))
        n_components = min(int(requested), max_components)

        y = np.zeros((x.shape[0], self.classes_.shape[0]), dtype=float)
        y[np.arange(x.shape[0]), encoded] = 1.0
        self.model_ = PLSRegression(n_components=n_components, scale=False)
        self.model_.fit(x, y)
        self.n_components_ = n_components
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("PLSDiscriminantTransformer must be fitted before transform.")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError("PLSDiscriminantTransformer expects a two-dimensional feature matrix.")
        transformed = self.model_.transform(x)
        if isinstance(transformed, tuple):
            transformed = transformed[0]
        return np.asarray(transformed, dtype=float)


class RegistryDecoder(ClassifierMixin, BaseEstimator):
    """Scikit-learn estimator adapter for ``decoding.classifiers`` entries.

    The time-resolved MNE decoder path expects estimators that can be placed in
    a sklearn pipeline and, optionally, wrapped in ``CalibratedClassifierCV``.
    Most legacy registry classifiers are factory functions rather than sklearn
    estimators themselves; this adapter exposes them through the standard
    ``fit``/``predict``/``decision_function``/``predict_proba`` API.
    """

    def __init__(self, classifier: str, classifier_param: Any = None, random_state: int | None = 13):
        self.classifier = classifier
        self.classifier_param = classifier_param
        self.random_state = random_state

    def fit(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence | np.ndarray,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ):
        classifier = normalize_registry_decoder_name(self.classifier)
        classifier_param = get_default_classifier_param(classifier) if self.classifier_param is None else self.classifier_param
        self.model_ = train_multiclass_classifier(
            features,
            labels,
            classifier,
            classifier_param,
            random_state=self.random_state,
            sample_weight=sample_weight,
        )
        self.classes_ = np.asarray(getattr(self.model_, "classes_", np.unique(labels)))
        self.classifier_ = classifier
        self.classifier_param_ = classifier_param
        return self

    def _raw_model(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("RegistryDecoder must be fitted before prediction.")
        return getattr(self.model_, "model", self.model_)

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("RegistryDecoder must be fitted before prediction.")
        return np.asarray(self.model_.predict(features))

    def decision_function(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        raw_model = self._raw_model()
        if hasattr(raw_model, "decision_function"):
            scores = np.asarray(raw_model.decision_function(features), dtype=float)
            if scores.ndim == 2 and getattr(self, "classes_", np.array([])).shape[0] == 2:
                return scores[:, 1] - scores[:, 0]
            return scores
        if hasattr(raw_model, "predict_proba"):
            probabilities = np.asarray(raw_model.predict_proba(features), dtype=float)
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                return np.log(np.clip(probabilities[:, 1], 1e-12, 1.0)) - np.log(np.clip(probabilities[:, 0], 1e-12, 1.0))
            return np.log(np.clip(probabilities, 1e-12, 1.0))
        return np.asarray(self.model_.decision_function(features), dtype=float)

    def predict_proba(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("RegistryDecoder must be fitted before prediction.")
        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError(f"{self.classifier!r} does not provide predict_proba")
        return np.asarray(self.model_.predict_proba(features), dtype=float)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Small CPU-friendly PyTorch MLP exposed as a sklearn classifier.

    The estimator intentionally imports torch only inside ``fit`` and
    ``predict`` so the optional torch extra is not required for normal sklearn
    decoder use or for constructing config grids that do not select this model.
    It is designed for held-out-subject MEG smoke runs: a single hidden layer,
    class-balanced cross entropy, modest early stopping, and no background GPU
    assumptions.
    """

    def __init__(
        self,
        hidden_units: int = 64,
        max_iter: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.1,
        patience: int = 8,
        dropout: float = 0.1,
        random_state: int | None = 13,
        class_weight: str | None = "balanced",
    ):
        self.hidden_units = hidden_units
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.dropout = dropout
        self.random_state = random_state
        self.class_weight = class_weight

    def _torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
            raise ImportError("The 'torch_mlp' decoder requires the optional torch extra, e.g. `pip install neureptrace[torch]`.") from exc
        return torch

    def fit(self, features: Sequence[Sequence[float]] | np.ndarray, labels: Sequence | np.ndarray):
        torch = self._torch()
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))

        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("TorchMLPClassifier expects a two-dimensional feature matrix.")
        y_raw = np.asarray(labels)
        self.classes_, y = np.unique(y_raw, return_inverse=True)
        y = y.astype(np.int64, copy=False)
        n_classes = int(self.classes_.shape[0])
        if n_classes < 2:
            raise ValueError("TorchMLPClassifier needs at least two classes.")

        hidden_units = int(self.hidden_units)
        max_iter = int(self.max_iter)
        batch_size = int(self.batch_size)
        if hidden_units < 1 or max_iter < 1 or batch_size < 1:
            raise ValueError("hidden_units, max_iter, and batch_size must be positive integers.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite.")

        indices = np.arange(y.shape[0])
        class_counts = np.bincount(y, minlength=n_classes)
        can_validate = (
            0.0 < float(self.validation_fraction) < 1.0
            and y.shape[0] >= 2 * n_classes
            and np.min(class_counts) >= 2
        )
        if can_validate:
            train_idx, validation_idx = train_test_split(
                indices,
                test_size=float(self.validation_fraction),
                stratify=y,
                random_state=self.random_state,
            )
        else:
            train_idx = indices
            validation_idx = np.array([], dtype=int)

        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], hidden_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(float(self.dropout)),
            torch.nn.Linear(hidden_units, n_classes),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))

        class_weights = None
        if self.class_weight == "balanced":
            weights = y.shape[0] / (n_classes * np.maximum(class_counts, 1))
            class_weights = torch.tensor(weights, dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        best_state = None
        best_loss = float("inf")
        stale_epochs = 0
        rng = np.random.default_rng(self.random_state)
        for _epoch in range(max_iter):
            model.train()
            shuffled = np.array(train_idx, copy=True)
            rng.shuffle(shuffled)
            for start in range(0, shuffled.shape[0], batch_size):
                batch_idx = shuffled[start : start + batch_size]
                optimizer.zero_grad()
                logits = model(x_tensor[batch_idx])
                loss = criterion(logits, y_tensor[batch_idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                eval_idx = validation_idx if validation_idx.size else train_idx
                val_loss = float(criterion(model(x_tensor[eval_idx]), y_tensor[eval_idx]).item())
            if val_loss + 1e-7 < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if validation_idx.size and stale_epochs >= int(self.patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.n_features_in_ = x.shape[1]
        return self

    def _predict_logits(self, features: Sequence[Sequence[float]] | np.ndarray):
        torch = self._torch()
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMLPClassifier must be fitted before prediction.")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("TorchMLPClassifier expects a two-dimensional feature matrix.")
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.tensor(x, dtype=torch.float32))

    def predict_proba(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        torch = self._torch()
        probabilities = torch.softmax(self._predict_logits(features), dim=1)
        return probabilities.detach().cpu().numpy()

    def decision_function(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        logits = self._predict_logits(features)
        return logits.detach().cpu().numpy()

    def predict(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return self.classes_[np.argmax(probabilities, axis=1)]


class ECOCLinearSVC(OutputCodeClassifier):
    """Deterministic output-code linear SVM with probability-like emissions.

    ``OutputCodeClassifier`` is useful for many-class MEG decoding, but its
    internal random codebook is otherwise a hidden source of run-to-run variation.
    This subclass also exposes a score matrix aligned with ``classes_`` so
    NeuRepTrace's generic emission conversion can be used for downstream state
    inference.
    """

    def __init__(self, C: float = 1.0, code_size: float = 1.5, max_iter: int = 1000, random_state: int | None = 13):
        self.C = C
        self.code_size = code_size
        self.max_iter = max_iter
        self.random_state = random_state
        super().__init__(
            estimator=LinearSVC(
                class_weight="balanced",
                C=C,
                max_iter=max_iter,
                random_state=random_state,
            ),
            code_size=code_size,
            random_state=random_state,
        )

    def get_params(self, deep: bool = True):
        return {
            "C": self.C,
            "code_size": self.code_size,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.estimator = LinearSVC(
            class_weight="balanced",
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.estimator_ = self.estimator
        return self

    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        self.classes_ = np.asarray(self.classes_)
        return self

    def _class_score_matrix(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        decisions = np.asarray(super().decision_function(features), dtype=float)
        if decisions.ndim == 1:
            return np.column_stack([-decisions, decisions])
        return decisions



def _make_registry_decoder_pipeline(
    name: str,
    *,
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    classifier_param: Any = None,
    random_state: int | None = 13,
):
    return make_pipeline(
        StandardScaler(),
        *_feature_preprocessor_steps(feature_preprocessor, pca_components),
        RegistryDecoder(
            normalize_registry_decoder_name(name),
            classifier_param=classifier_param,
            random_state=random_state,
        ),
    )


def _registry_tuning_param_grid(name: str, c_grid: Sequence[float]) -> dict[str, Sequence[Any]]:
    registry_name = normalize_registry_decoder_name(name)
    if registry_name in {"multiclass-svm", "multiclass-svm-weighted", "multinomial-logistic"}:
        return {"registrydecoder__classifier_param": c_grid}
    if registry_name == "knn":
        return {"registrydecoder__classifier_param": (3, 5, 7, 11)}
    if registry_name in {"random-forest", "gradient-boosting", "xgboost"}:
        return {"registrydecoder__classifier_param": (50, 100, 200)}
    return {}


def make_logistic_decoder(
    max_iter: int = 1000,
    *,
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
):
    """Create the default calibrated-probability baseline decoder."""
    return make_decoder(
        "logistic",
        max_iter=max_iter,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
    )


def make_decoder(
    name: str = "logistic",
    *,
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    classifier_param: Any = None,
    random_state: int | None = 13,
):
    """Create a standard probability-producing decoder by name.

    Optional feature preprocessing is inserted after fold-local standardization
    and before the classifier. This keeps low-rank transforms such as PCA inside
    each cross-validation fold and prevents train/test leakage.

    When ``tune_hyperparameters`` is enabled, the returned estimator is a
    ``GridSearchCV`` wrapper around the same decoder family. The caller can pass
    an integer CV count or precomputed inner-CV splits via ``tuning_cv``.
    """
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)

    if tune_hyperparameters:
        return make_tuned_decoder(
            normalized,
            max_iter=max_iter,
            emission_mode=emission_mode,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            cv=tuning_cv,
            scoring=tuning_scoring,
            c_grid=tuning_c_grid,
            classifier_param=classifier_param,
            random_state=random_state,
        )

    if normalized == "logistic":
        c_value = _positive_float_classifier_param(classifier_param, default=1.0, name="LogisticRegression C")
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                C=c_value,
                max_iter=max_iter,
                random_state=random_state,
                solver="lbfgs",
            ),
        )
    if normalized == "sparse_logistic":
        c_value = _positive_float_classifier_param(classifier_param, default=1.0, name="LogisticRegression C")
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                penalty="l1",
                C=c_value,
                max_iter=max_iter,
                random_state=random_state,
                solver="saga",
            ),
        )
    if normalized == "elastic_net_logistic":
        c_value = _positive_float_classifier_param(classifier_param, default=1.0, name="LogisticRegression C")
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                penalty="elasticnet",
                C=c_value,
                l1_ratio=DEFAULT_ELASTIC_NET_L1_RATIO,
                max_iter=max_iter,
                random_state=random_state,
                solver="saga",
            ),
        )
    if normalized == "gaussian_nb":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            GaussianNB(),
        )
    if normalized == "lda":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(solver="svd"),
        )
    if normalized == "shrinkage_lda":
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
    if normalized == "ridge":
        ridge = make_pipeline(
            StandardScaler(),
            *feature_steps,
            RidgeClassifier(
                class_weight="balanced",
                max_iter=max_iter,
                random_state=random_state,
            ),
        )
        if emission_mode == "uncalibrated":
            return ridge
        return _make_calibrated_classifier(
            ridge,
            method="sigmoid",
            cv=3,
        )

    if normalized == "linear_svm":
        c_value = _positive_float_classifier_param(classifier_param, default=1.0, name="LinearSVC C")
        linear_svm = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearSVC(
                class_weight="balanced",
                C=c_value,
                max_iter=max_iter,
                random_state=random_state,
            ),
        )
        if emission_mode == "uncalibrated":
            return linear_svm
        return _make_calibrated_classifier(
            linear_svm,
            method="sigmoid",
            cv=3,
        )

    if normalized in {"ovo_linear_svm", "ecoc_linear_svm"}:
        c_value = _positive_float_classifier_param(classifier_param, default=1.0, name="LinearSVC C")
        multiclass_svm = (
            OneVsOneClassifier(
                LinearSVC(
                    class_weight="balanced",
                    C=c_value,
                    max_iter=max_iter,
                    random_state=random_state,
                )
            )
            if normalized == "ovo_linear_svm"
            else ECOCLinearSVC(
                C=c_value,
                max_iter=max_iter,
                random_state=random_state,
            )
        )
        model = make_pipeline(
            StandardScaler(),
            *feature_steps,
            multiclass_svm,
        )
        if emission_mode == "uncalibrated":
            return model
        return _make_calibrated_classifier(
            model,
            method="sigmoid",
            cv=3,
        )

    if normalized == "torch_mlp":
        weight_decay = _positive_float_classifier_param(
            classifier_param,
            default=1e-4,
            name="TorchMLP weight_decay",
        )
        return make_pipeline(
            StandardScaler(),
            *feature_steps,
            TorchMLPClassifier(
                max_iter=max_iter,
                weight_decay=weight_decay,
                random_state=random_state,
            ),
        )

    registry_decoder = _make_registry_decoder_pipeline(
        normalized,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        classifier_param=classifier_param,
        random_state=random_state,
    )
    if emission_mode == "uncalibrated":
        return registry_decoder
    return _make_calibrated_classifier(
        registry_decoder,
        method="sigmoid",
        cv=3,
    )


def make_tuned_decoder(
    name: str = "logistic",
    *,
    max_iter: int = 1000,
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    cv: int | Sequence[tuple[np.ndarray, np.ndarray]] = 3,
    scoring: str = "accuracy",
    c_grid: Sequence[float] | str | None = None,
    classifier_param: Any = None,
    random_state: int | None = 13,
):
    """Create a decoder with inner-CV hyperparameter selection.

    Logistic regression, sparse logistic regression, and linear SVM tune the
    regularization strength ``C``. Elastic-net logistic regression tunes both
    ``C`` and the L1/L2 mixing ratio. Ridge tunes the L2 penalty strength
    ``alpha``. Gaussian NB tunes variance smoothing. LDA compares the default
    SVD solver with shrinkage LDA
    (``solver='lsqr', shrinkage='auto'``), which is often better conditioned for
    high-dimensional M/EEG windows.
    """
    normalized = normalize_decoder_name(name)
    emission_mode = normalize_emission_mode(emission_mode)
    scoring = normalize_tuning_scoring(scoring)
    c_grid = parse_c_grid(c_grid)
    feature_steps = _feature_preprocessor_steps(feature_preprocessor, pca_components)

    if normalized == "logistic":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                random_state=random_state,
                solver="lbfgs",
            ),
        )
        param_grid = {"logisticregression__C": c_grid}
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized == "sparse_logistic":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                penalty="l1",
                max_iter=max_iter,
                random_state=random_state,
                solver="saga",
            ),
        )
        param_grid = {"logisticregression__C": c_grid}
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized == "elastic_net_logistic":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LogisticRegression(
                class_weight="balanced",
                penalty="elasticnet",
                l1_ratio=DEFAULT_ELASTIC_NET_L1_RATIO,
                max_iter=max_iter,
                random_state=random_state,
                solver="saga",
            ),
        )
        param_grid = {
            "logisticregression__C": c_grid,
            "logisticregression__l1_ratio": ELASTIC_NET_L1_RATIO_GRID,
        }
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized == "gaussian_nb":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            GaussianNB(),
        )
        param_grid = {"gaussiannb__var_smoothing": DEFAULT_TUNING_VAR_SMOOTHING_GRID}
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized == "lda":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(),
        )
        param_grid = [
            {
                "lineardiscriminantanalysis__solver": ["svd"],
                "lineardiscriminantanalysis__shrinkage": [None],
            },
            {
                "lineardiscriminantanalysis__solver": ["lsqr"],
                "lineardiscriminantanalysis__shrinkage": ["auto"],
            },
        ]
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized == "shrinkage_lda":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearDiscriminantAnalysis(solver="lsqr"),
        )
        param_grid = {"lineardiscriminantanalysis__shrinkage": ["auto", 0.1, 0.3, 0.5, 0.7, 0.9]}
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized == "ridge":
        if emission_mode == "uncalibrated" and scoring == "neg_log_loss":
            raise ValueError("neg_log_loss tuning requires probability estimates; use calibrated emissions for ridge.")
        ridge = make_pipeline(
            StandardScaler(),
            *feature_steps,
            RidgeClassifier(
                class_weight="balanced",
                max_iter=max_iter,
                random_state=random_state,
            ),
        )
        if emission_mode == "uncalibrated":
            estimator = ridge
            param_grid = {"ridgeclassifier__alpha": DEFAULT_TUNING_ALPHA_GRID}
        else:
            estimator = _make_calibrated_classifier(ridge, method="sigmoid", cv=3)
            param_grid = {_calibrated_estimator_param(estimator, "ridgeclassifier__alpha"): DEFAULT_TUNING_ALPHA_GRID}
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized == "linear_svm":
        linear_svm = make_pipeline(
            StandardScaler(),
            *feature_steps,
            LinearSVC(
                class_weight="balanced",
                max_iter=max_iter,
                random_state=random_state,
            ),
        )
        if emission_mode == "uncalibrated":
            estimator = linear_svm
            param_grid = {"linearsvc__C": c_grid}
        else:
            estimator = _make_calibrated_classifier(linear_svm, method="sigmoid", cv=3)
            param_grid = {_calibrated_estimator_param(estimator, "linearsvc__C"): c_grid}
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized in {"ovo_linear_svm", "ecoc_linear_svm"}:
        multiclass_svm = (
            OneVsOneClassifier(
                LinearSVC(
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=random_state,
                )
            )
            if normalized == "ovo_linear_svm"
            else ECOCLinearSVC(
                max_iter=max_iter,
                random_state=random_state,
            )
        )
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            multiclass_svm,
        )
        svm_c_param = "onevsoneclassifier__estimator__C" if normalized == "ovo_linear_svm" else "ecoclinearsvc__C"
        if emission_mode == "uncalibrated":
            param_grid = {svm_c_param: c_grid}
        else:
            estimator = _make_calibrated_classifier(estimator, method="sigmoid", cv=3)
            param_grid = {_calibrated_estimator_param(estimator, svm_c_param): c_grid}
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    elif normalized == "torch_mlp":
        estimator = make_pipeline(
            StandardScaler(),
            *feature_steps,
            TorchMLPClassifier(
                max_iter=max_iter,
                random_state=random_state,
            ),
        )
        # Interpret the shared C grid as inverse regularization strength for this
        # decoder so CLI tuning semantics remain consistent with linear models.
        param_grid = {"torchmlpclassifier__weight_decay": tuple(1.0 / value for value in c_grid)}
        param_grid = _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
    else:
        registry_name = normalize_registry_decoder_name(normalized)
        registry_decoder = _make_registry_decoder_pipeline(
            registry_name,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            classifier_param=classifier_param,
            random_state=random_state,
        )
        param_grid = _registry_tuning_param_grid(registry_name, c_grid)
        if emission_mode == "uncalibrated":
            estimator = registry_decoder
        else:
            estimator = _make_calibrated_classifier(registry_decoder, method="sigmoid", cv=3)
            param_grid = _calibrated_param_grid(estimator, param_grid)

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=make_tuning_scorer(scoring, emission_mode=emission_mode),
        cv=cv,
        refit=True,
    )


def _make_calibrated_classifier(estimator, *, method: str, cv: int):
    """Construct CalibratedClassifierCV across sklearn estimator/base_estimator APIs."""
    kwargs = {"method": method, "cv": cv}
    if "estimator" in inspect.signature(CalibratedClassifierCV).parameters:
        kwargs["estimator"] = estimator
    else:
        kwargs["base_estimator"] = estimator
    return CalibratedClassifierCV(**kwargs)


def _calibrated_estimator_param(estimator, nested_parameter: str) -> str:
    params = estimator.get_params()
    for prefix in ("estimator", "base_estimator"):
        candidate = f"{prefix}__{nested_parameter}"
        if candidate in params:
            return candidate
    raise ValueError(f"Could not find calibrated-estimator parameter for '{nested_parameter}'.")


def _feature_preprocessor_param(estimator, feature_preprocessor: str | None) -> str | None:
    normalized = normalize_feature_preprocessor(feature_preprocessor)
    if normalized == "anova_select":
        direct = "selectpercentile__percentile"
    elif normalized == "pls_da":
        direct = "plsdiscriminanttransformer__n_components"
    else:
        return None
    if direct in estimator.get_params():
        return direct
    return _calibrated_estimator_param(estimator, direct)


def _with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor: str | None):
    normalized = normalize_feature_preprocessor(feature_preprocessor)
    feature_param = _feature_preprocessor_param(estimator, feature_preprocessor)
    if feature_param is None:
        return param_grid
    feature_values = ANOVA_SELECT_PERCENTILE_GRID if normalized == "anova_select" else PLS_COMPONENT_GRID
    if isinstance(param_grid, list):
        return [{**grid, feature_param: feature_values} for grid in param_grid]
    return {**param_grid, feature_param: feature_values}


def parse_c_grid(values: Sequence[float] | str | None) -> tuple[float, ...]:
    """Normalize a regularization-strength grid for CLI and API callers."""
    if values is None:
        return DEFAULT_TUNING_C_GRID
    if isinstance(values, str):
        values = [value.strip() for value in values.split(",") if value.strip()]
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError("At least one C value is required for hyperparameter tuning.")
    if any(value <= 0 for value in grid):
        raise ValueError("All C values must be positive.")
    return grid


def normalize_tuning_scoring(scoring: str) -> str:
    """Normalize inner-CV scoring names."""
    normalized = scoring.lower().replace("-", "_")
    if normalized not in TUNING_SCORING_CHOICES:
        raise ValueError(f"Unknown tuning scoring '{scoring}'. Available values: {', '.join(TUNING_SCORING_CHOICES)}.")
    return normalized


def make_tuning_scorer(scoring: str, *, emission_mode: str = "calibrated") -> str | Callable:
    """Return a GridSearchCV scorer for decoder hyperparameter tuning.

    Accuracy-oriented objectives are forwarded to scikit-learn by name. Probability
    objectives are implemented here so they use the same calibrated or
    score-derived emissions that NeuRepTrace writes to the held-out observation
    tables. This keeps model selection aligned with downstream temporal-state
    inference, where probability quality matters more than the hard class label.
    """
    normalized = normalize_tuning_scoring(scoring)
    emission_mode = normalize_emission_mode(emission_mode)
    if normalized in {"accuracy", "balanced_accuracy"}:
        return normalized
    return _make_probability_tuning_scorer(normalized, emission_mode=emission_mode)


def _make_probability_tuning_scorer(scoring: str, *, emission_mode: str) -> Callable:
    def scorer(estimator, features: np.ndarray, labels: np.ndarray) -> float:
        probabilities = predict_emission_probabilities(estimator, features, emission_mode=emission_mode)
        label_indices = _labels_to_probability_columns(labels, estimator=estimator, n_classes=probabilities.shape[1])
        if scoring == "neg_log_loss":
            return -float(log_loss(label_indices, probabilities, labels=np.arange(probabilities.shape[1])))
        if scoring == "neg_brier":
            return -brier_score_multiclass(probabilities, label_indices)
        if scoring == "neg_ece":
            return -expected_calibration_error(probabilities, label_indices)
        raise ValueError(f"Unknown probability tuning scoring '{scoring}'.")

    return scorer


def _labels_to_probability_columns(
    labels: np.ndarray,
    *,
    estimator,
    n_classes: int,
) -> np.ndarray:
    """Map estimator labels to probability-column indices for multiclass metrics."""
    labels = np.asarray(labels)
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        classes = np.asarray(classes)
        if len(classes) != n_classes:
            raise ValueError(f"Estimator reports {len(classes)} classes but predicted {n_classes} probability columns.")
        class_to_index = {class_label: class_index for class_index, class_label in enumerate(classes.tolist())}
        try:
            return np.asarray([class_to_index[label] for label in labels.tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Validation label {exc.args[0]!r} was not seen by the fitted estimator.") from exc

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Probability tuning metrics require fitted estimator classes for non-integer labels.")
    label_indices = labels.astype(int, copy=False)
    if np.any((label_indices < 0) | (label_indices >= n_classes)):
        raise ValueError("Integer labels must be valid probability-column indices.")
    return label_indices


def normalize_decoder_name(name: str) -> str:
    """Normalize decoder aliases to the names used in result tables."""
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"nb", "naive_bayes", "gaussian_naive_bayes"}:
        return "gaussian_nb"
    if normalized == "svm":
        return "linear_svm"
    if normalized in {"l1_logistic", "logistic_l1", "sparse_logreg"}:
        return "sparse_logistic"
    if normalized in {"elasticnet_logistic", "logistic_elastic_net", "elastic_net_logreg"}:
        return "elastic_net_logistic"
    if normalized in {"ridge_classifier", "ridge_classification"}:
        return "ridge"
    if normalized in {"lda_shrinkage", "shrinkage_lda", "shrinkagelda"}:
        return "shrinkage_lda"
    if normalized in {"one_vs_one_linear_svm", "onevsone_linear_svm", "ovo_svm", "ovo_linear_svm"}:
        return "ovo_linear_svm"
    if normalized in {"ecoc_svm", "output_code_linear_svm", "outputcode_linear_svm", "ecoc_linear_svm"}:
        return "ecoc_linear_svm"
    if normalized in {"deep_mlp", "mlp", "torch_deep_mlp", "shallow_torch_mlp"}:
        return "torch_mlp"
    if normalized in BUILTIN_DECODER_CHOICES:
        return normalized
    registry_name = _normalize_registry_decoder_name_or_none(name)
    if registry_name is not None:
        return registry_name
    raise ValueError(f"Unknown decoder '{name}'. Available decoders: {', '.join(DECODER_CHOICES)}.")


def normalize_emission_mode(mode: str) -> str:
    """Normalize calibrated/uncalibrated emission mode names."""
    normalized = mode.lower().replace("-", "_")
    if normalized not in EMISSION_MODE_CHOICES:
        raise ValueError(f"Unknown emission mode '{mode}'. Available modes: {', '.join(EMISSION_MODE_CHOICES)}.")
    return normalized


def normalize_feature_preprocessor(name: str | None) -> str:
    """Normalize feature-preprocessor aliases to canonical result-table names."""
    normalized = "none" if name is None else name.lower().replace("-", "_")
    if normalized in {"identity", "standard", "standardize", "scaler", "standard_scaler"}:
        return "none"
    if normalized in {"pca_whitened", "whitened_pca", "whiten_pca"}:
        return "pca_whiten"
    if normalized in {"anova", "anova_percentile", "select_percentile", "select_k_best", "kbest"}:
        return "anova_select"
    if normalized in {"pls", "plsd", "pls_da", "pls_discriminant", "pls_regression", "pls_discriminant_analysis", "supervised_pca"}:
        return "pls_da"
    if normalized not in FEATURE_PREPROCESSOR_CHOICES:
        raise ValueError(
            f"Unknown feature preprocessor '{name}'. Available preprocessors: {', '.join(FEATURE_PREPROCESSOR_CHOICES)}."
        )
    return normalized


def normalize_pca_components(n_components: int | float | str | None) -> int | float | None:
    """Normalize PCA component specifications for sklearn.

    Integers select an explicit component count. Floats in ``(0, 1)`` select an
    explained-variance fraction. ``None``, ``auto``, or an empty string keep
    sklearn's default ``PCA(n_components=None)`` behavior.
    """
    if n_components is None:
        return None
    if isinstance(n_components, str):
        stripped = n_components.strip()
        if stripped == "" or stripped.lower() in {"none", "auto", "default"}:
            return None
        try:
            parsed: int | float = float(stripped) if any(marker in stripped for marker in (".", "e", "E")) else int(stripped)
        except ValueError as exc:
            raise ValueError("pca_components must be an integer count, a variance fraction in (0, 1), or None.") from exc
        return normalize_pca_components(parsed)
    if isinstance(n_components, (np.integer,)):
        n_components = int(n_components)
    if isinstance(n_components, (np.floating,)):
        n_components = float(n_components)
    if isinstance(n_components, bool):
        raise ValueError("pca_components must be numeric, not boolean.")
    if isinstance(n_components, int):
        if n_components < 1:
            raise ValueError("Integer pca_components must be at least 1.")
        return n_components
    if isinstance(n_components, float):
        if not np.isfinite(n_components) or n_components <= 0.0:
            raise ValueError("Float pca_components must be finite and positive.")
        if n_components < 1.0:
            return float(n_components)
        if n_components.is_integer():
            return int(n_components)
    raise ValueError("pca_components must be an integer count, a variance fraction in (0, 1), or None.")


def normalize_anova_select_percentile(percentile: int | float | str | None) -> int:
    """Normalize ANOVA feature-selection percentile specifications."""
    if percentile is None:
        return DEFAULT_ANOVA_SELECT_PERCENTILE
    if isinstance(percentile, str):
        stripped = percentile.strip()
        if stripped == "" or stripped.lower() in {"auto", "default"}:
            return DEFAULT_ANOVA_SELECT_PERCENTILE
        try:
            parsed: int | float = float(stripped) if any(marker in stripped for marker in (".", "e", "E")) else int(stripped)
        except ValueError as exc:
            raise ValueError("anova_select percentile must be a number in (0, 100].") from exc
        return normalize_anova_select_percentile(parsed)
    if isinstance(percentile, (np.integer,)):
        percentile = int(percentile)
    if isinstance(percentile, (np.floating,)):
        percentile = float(percentile)
    if isinstance(percentile, bool):
        raise ValueError("anova_select percentile must be numeric, not boolean.")
    if not isinstance(percentile, (int, float)) or not np.isfinite(percentile) or percentile <= 0 or percentile > 100:
        raise ValueError("anova_select percentile must be finite and in (0, 100].")
    if not float(percentile).is_integer():
        raise ValueError("anova_select percentile must be an integer percentage.")
    return int(percentile)


def normalize_pls_components(n_components: int | str | None) -> int:
    """Normalize supervised PLS-DA component counts.

    PLS component counts are integer-only.  Fractional explained-variance values
    are intentionally rejected because PLS-DA is supervised and does not have the
    same variance-retention semantics as PCA.
    """

    if n_components is None:
        return DEFAULT_PLS_COMPONENTS
    if isinstance(n_components, str) and n_components.strip().lower() in {"", "none", "auto", "default"}:
        return DEFAULT_PLS_COMPONENTS
    normalized = normalize_pca_components(n_components)
    if isinstance(normalized, float):
        raise ValueError("PLS-DA components must be an integer count or auto/default, not a variance fraction.")
    if normalized is None:
        return DEFAULT_PLS_COMPONENTS
    return int(normalized)


def _feature_preprocessor_steps(
    feature_preprocessor: str | None,
    pca_components: int | float | str | None,
) -> list[PCA | SelectPercentile | PLSDiscriminantTransformer]:
    normalized = normalize_feature_preprocessor(feature_preprocessor)
    if normalized == "none":
        if pca_components is not None:
            raise ValueError("pca_components can only be set when feature_preprocessor is 'pca', 'pca_whiten', or 'pls_da'.")
        return []
    if normalized == "pca":
        return [PCA(n_components=normalize_pca_components(pca_components), whiten=False)]
    if normalized == "pca_whiten":
        return [PCA(n_components=normalize_pca_components(pca_components), whiten=True)]
    if normalized == "anova_select":
        return [SelectPercentile(f_classif, percentile=normalize_anova_select_percentile(pca_components))]
    if normalized == "pls_da":
        return [PLSDiscriminantTransformer(n_components=normalize_pls_components(pca_components))]
    raise ValueError(f"Unknown feature preprocessor '{feature_preprocessor}'.")


def predict_emission_probabilities(
    estimator,
    features: np.ndarray,
    *,
    emission_mode: str = "calibrated",
) -> np.ndarray:
    """Return class probability emissions from a fitted estimator.

    In calibrated mode, the estimator must expose ``predict_proba``. In
    uncalibrated mode, margins are converted with a numerically stable softmax so
    deterministic models such as LinearSVC and RidgeClassifier can be compared in
    the same downstream state-inference code path without Platt scaling.
    """
    emission_mode = normalize_emission_mode(emission_mode)
    if emission_mode == "calibrated" and hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
    elif hasattr(estimator, "decision_function"):
        probabilities = score_to_probabilities(estimator.decision_function(features))
    elif hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
    else:
        raise AttributeError("Estimator must expose predict_proba or decision_function.")
    return _normalize_probability_rows(probabilities)


def score_to_probabilities(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return _normalize_probability_rows(probabilities)


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probability predictions must be a two-dimensional array.")
    if probabilities.shape[1] < 2:
        raise ValueError("Probability predictions must contain at least two class columns.")
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Probability rows must have positive total mass.")
    return probabilities / row_sums


def time_windows(times: Sequence[float], window_ms: float, step_ms: float) -> list[tuple[int, int, float]]:
    """Generate sliding time windows over sample times."""
    times_array = np.asarray(times, dtype=float)
    if times_array.ndim != 1 or times_array.size == 0:
        raise ValueError("times must be a non-empty one-dimensional sequence.")
    window_s = float(window_ms) / 1000.0
    step_s = float(step_ms) / 1000.0
    if window_s <= 0 or step_s <= 0:
        raise ValueError("window_ms and step_ms must be positive.")
    start_time = float(times_array[0])
    end_time = float(times_array[-1])
    windows: list[tuple[int, int, float]] = []
    current = start_time
    tolerance = 1e-12
    while current + window_s <= end_time + tolerance:
        start = int(np.searchsorted(times_array, current, side="left"))
        stop = int(np.searchsorted(times_array, current + window_s, side="left"))
        if stop > start:
            windows.append((start, stop, float((times_array[start] + times_array[stop - 1]) / 2.0)))
        current += step_s
    return windows


def make_cross_validator(labels: Sequence, groups: Sequence | None = None, n_splits: int = 5):
    labels = np.asarray(labels)
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_splits = min(n_splits, len(unique_groups))
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=13).split(np.zeros_like(labels), labels, groups)
    class_counts = np.bincount(labels.astype(int)) if np.issubdtype(labels.dtype, np.integer) else np.array([np.sum(labels == value) for value in np.unique(labels)])
    n_splits = min(n_splits, int(class_counts.min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13).split(np.zeros_like(labels), labels)


def make_tuning_cross_validator(labels: Sequence, groups: Sequence | None = None, n_splits: int = 3):
    labels = np.asarray(labels)
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        feasible_splits = int(min(n_splits, len(unique_groups)))
        if feasible_splits < 2:
            raise ValueError("Grouped tuning CV requires at least two groups.")
        return list(StratifiedGroupKFold(n_splits=feasible_splits, shuffle=True, random_state=13).split(np.zeros_like(labels), labels, groups))

    class_counts = np.bincount(labels.astype(int)) if np.issubdtype(labels.dtype, np.integer) else np.array([np.sum(labels == value) for value in np.unique(labels)])
    feasible_splits = int(min(n_splits, class_counts.min()))
    if feasible_splits < 2:
        raise ValueError("Tuning CV requires at least two samples per class.")
    return list(StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=13).split(np.zeros_like(labels), labels))
