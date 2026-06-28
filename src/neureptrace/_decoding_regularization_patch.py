"""Runtime compatibility patch for logistic decoder regularization settings.

This module keeps the public ``neureptrace.decoding`` API stable while fixing
legacy sparse-logistic and elastic-net-logistic decoder construction.  It can be
removed once the overridden functions are folded directly into
``neureptrace.decoding``.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


_PATCH_MARKER = "_neureptrace_regularization_patch_installed"


def _uses_l1_ratio_logistic_api() -> bool:
    """Return whether scikit-learn expects logistic regularization via ``l1_ratio``."""

    penalty = inspect.signature(LogisticRegression).parameters.get("penalty")
    return penalty is not None and penalty.default == "deprecated"


def _logistic_regularization_kwargs(l1_ratio: float) -> dict[str, float | str]:
    """Build version-compatible ``LogisticRegression`` regularization kwargs."""

    l1_ratio = float(l1_ratio)
    if _uses_l1_ratio_logistic_api():
        return {"l1_ratio": l1_ratio}
    if l1_ratio >= 1.0:
        return {"penalty": "l1", "l1_ratio": 1.0}
    if l1_ratio <= 0.0:
        return {"penalty": "l2", "l1_ratio": 0.0}
    return {"penalty": "elasticnet", "l1_ratio": l1_ratio}


def _install_decoding_option_type_validation() -> None:
    """Install decoder-option type validation alongside the decoder API patch."""

    patch = importlib.import_module("neureptrace._decoding_option_type_validation_patch")
    patch.install()


def install() -> None:
    """Install corrected sparse/elastic-net logistic decoder builders."""

    from neureptrace import decoding

    _install_decoding_option_type_validation()

    if getattr(decoding, _PATCH_MARKER, False):
        return

    original_make_decoder = decoding.make_decoder
    original_make_tuned_decoder = decoding.make_tuned_decoder

    def _regularized_logistic_estimator(
        *,
        normalized: str,
        max_iter: int,
        feature_preprocessor: str,
        pca_components: int | float | str | None,
        random_state: int | None,
        classifier_param: Any = None,
    ):
        feature_steps = decoding._feature_preprocessor_steps(feature_preprocessor, pca_components)
        c_value = decoding._positive_float_classifier_param(
            classifier_param,
            default=1.0,
            name="LogisticRegression C",
        )
        if normalized == "sparse_logistic":
            return make_pipeline(
                StandardScaler(),
                *feature_steps,
                LogisticRegression(
                    class_weight="balanced",
                    C=c_value,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="saga",
                    **_logistic_regularization_kwargs(1.0),
                ),
            )
        if normalized == "elastic_net_logistic":
            return make_pipeline(
                StandardScaler(),
                *feature_steps,
                LogisticRegression(
                    class_weight="balanced",
                    C=c_value,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="saga",
                    **_logistic_regularization_kwargs(decoding.DEFAULT_ELASTIC_NET_L1_RATIO),
                ),
            )
        raise ValueError(f"Unsupported regularized logistic decoder: {normalized}")

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
        normalized = decoding.normalize_decoder_name(name)
        if normalized not in {"sparse_logistic", "elastic_net_logistic"}:
            return original_make_decoder(
                name,
                max_iter=max_iter,
                emission_mode=emission_mode,
                feature_preprocessor=feature_preprocessor,
                pca_components=pca_components,
                tune_hyperparameters=tune_hyperparameters,
                tuning_cv=tuning_cv,
                tuning_scoring=tuning_scoring,
                tuning_c_grid=tuning_c_grid,
                classifier_param=classifier_param,
                random_state=random_state,
            )
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
        return _regularized_logistic_estimator(
            normalized=normalized,
            max_iter=max_iter,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            random_state=random_state,
            classifier_param=classifier_param,
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
        normalized = decoding.normalize_decoder_name(name)
        if normalized not in {"sparse_logistic", "elastic_net_logistic"}:
            return original_make_tuned_decoder(
                name,
                max_iter=max_iter,
                emission_mode=emission_mode,
                feature_preprocessor=feature_preprocessor,
                pca_components=pca_components,
                cv=cv,
                scoring=scoring,
                c_grid=c_grid,
                classifier_param=classifier_param,
                random_state=random_state,
            )

        decoding.normalize_emission_mode(emission_mode)
        c_grid_values = decoding.parse_c_grid(c_grid)
        scoring_name = decoding.normalize_tuning_scoring(scoring)
        estimator = _regularized_logistic_estimator(
            normalized=normalized,
            max_iter=max_iter,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            random_state=random_state,
        )
        param_grid: dict[str, Sequence[Any]] = {"logisticregression__C": c_grid_values}
        if normalized == "elastic_net_logistic":
            param_grid["logisticregression__l1_ratio"] = decoding.ELASTIC_NET_L1_RATIO_GRID
        param_grid = decoding._with_feature_preprocessor_tuning(estimator, param_grid, feature_preprocessor)
        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=decoding.make_tuning_scorer(scoring_name, emission_mode=emission_mode),
            cv=cv,
            refit=True,
        )

    make_decoder.__doc__ = original_make_decoder.__doc__
    make_tuned_decoder.__doc__ = original_make_tuned_decoder.__doc__
    decoding.make_decoder = make_decoder
    decoding.make_tuned_decoder = make_tuned_decoder
    setattr(decoding, _PATCH_MARKER, True)
