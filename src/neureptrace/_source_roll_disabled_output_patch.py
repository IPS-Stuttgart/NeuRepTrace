"""Honor ``preserve_original=False`` when source feature-roll is disabled."""

from __future__ import annotations

import importlib
from dataclasses import replace
from functools import wraps
from typing import Any

_PATCH_MARKER = "_source_roll_disabled_output_patch"


def install() -> None:
    """Install the disabled-output guard after the source-roll compatibility patch."""

    source_roll = importlib.import_module("neureptrace.decoding.source_roll")
    original = source_roll.augment_source_with_feature_roll
    # ``functools.wraps`` copies function attributes. A compatibility wrapper can
    # therefore inherit our marker without actually being the outer guard. Store
    # the wrapper itself as the marker value and only skip when it is still final.
    if getattr(original, _PATCH_MARKER, None) is original:
        return

    @wraps(original)
    def augment_source_with_feature_roll(
        source_features: Any,
        source_labels: Any,
        *,
        source_domains: Any = None,
        config: Any = None,
    ):
        cfg = source_roll.source_feature_roll_config() if config is None else source_roll._coerce_config(config)
        result = original(
            source_features,
            source_labels,
            source_domains=source_domains,
            config=cfg,
        )
        if cfg.enabled or cfg.preserve_original:
            return result
        return replace(
            result,
            features=result.features[:0].copy(),
            labels=result.labels[:0].copy(),
            synthetic_mask=result.synthetic_mask[:0].copy(),
        )

    setattr(augment_source_with_feature_roll, _PATCH_MARKER, augment_source_with_feature_roll)
    source_roll.augment_source_with_feature_roll = augment_source_with_feature_roll


__all__ = ["install"]
