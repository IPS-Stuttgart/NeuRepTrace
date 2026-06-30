"""Route random-subspace sample weights through sklearn Pipeline final estimators."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

_PATCH_MARKER = "_neureptrace_random_subspace_pipeline_weight_patch_installed"


class _PipelineSampleWeightAdapter(BaseEstimator):
    """Adapter that accepts estimator-level sample weights for sklearn Pipelines."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def fit(self, features, labels, sample_weight=None):
        self.pipeline_ = clone(self.pipeline)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            final_step_name = self.pipeline_.steps[-1][0]
            fit_kwargs[f"{final_step_name}__sample_weight"] = sample_weight
        self.pipeline_.fit(features, labels, **fit_kwargs)
        classes = getattr(self.pipeline_, "classes_", None)
        if classes is None:
            classes = getattr(self.pipeline_.steps[-1][1], "classes_", None)
        if classes is not None:
            self.classes_ = np.asarray(classes)
        return self

    def __getattr__(self, name: str):
        pipeline = self.__dict__.get("pipeline_")
        if pipeline is not None and hasattr(pipeline, name):
            return getattr(pipeline, name)
        raise AttributeError(name)


def _wrap_pipeline(estimator):
    if isinstance(estimator, Pipeline):
        return _PipelineSampleWeightAdapter(estimator)
    return estimator


def install() -> None:
    """Install the random-subspace Pipeline sample-weight adapter."""

    from neureptrace.decoding import random_subspace

    original_fit = random_subspace.fit_random_subspace_ensemble
    if getattr(original_fit, _PATCH_MARKER, False):
        return

    @wraps(original_fit)
    def fit_random_subspace_ensemble(*args, **kwargs):
        if kwargs.get("sample_weight") is not None:
            kwargs = dict(kwargs)
            estimator = kwargs.get("estimator")
            if estimator is None:
                config = kwargs.get("config")
                cfg = random_subspace.random_subspace_ensemble_config() if config is None else random_subspace._coerce_config(config)
                estimator = random_subspace._default_estimator(cfg)
            kwargs["estimator"] = _wrap_pipeline(estimator)
        return original_fit(*args, **kwargs)

    setattr(fit_random_subspace_ensemble, _PATCH_MARKER, True)
    random_subspace.fit_random_subspace_ensemble = fit_random_subspace_ensemble


__all__ = ["install"]
