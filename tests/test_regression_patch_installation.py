from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import neureptrace  # noqa: F401
from neureptrace import bushmeg_all_protocols_report as report
from neureptrace.decoding.source_jitter import source_feature_jitter_config
from neureptrace.decoding.source_masking import source_feature_masking_config


def test_fractional_protocol_report_patch_installed_on_package_import() -> None:
    summary = pd.DataFrame(
        {
            "protocol_category": [1.5],
            "method": ["fractional_protocol_method"],
            "method_family": ["smoke"],
            "outer_test_subject": ["sub-01"],
            "balanced_accuracy": [0.75],
            "accuracy": [0.80],
            "log_loss": [0.20],
            "brier": [0.10],
            "ece": [0.05],
        }
    )
    leaderboard = pd.DataFrame(
        {
            "protocol_category": [1.5],
            "method": ["fractional_protocol_method"],
            "n_rows": [1],
            "n_skipped": [0],
        }
    )

    protocol_summary = report.build_protocol_summary(summary, leaderboard)

    assert protocol_summary.loc[0, "protocol_category"] == 1.5
    assert protocol_summary.loc[0, "n_rows"] == 1
    assert report._format_protocol_label(1.5) == "1.5"


def test_source_augmentation_config_patch_installed_on_package_import() -> None:
    jitter_cfg = source_feature_jitter_config(
        preserve_original=np.asarray(1),
        random_state=np.asarray("none"),
    )
    assert jitter_cfg.preserve_original is True
    assert jitter_cfg.random_state is None

    masking_cfg = source_feature_masking_config(
        block_size=np.asarray("none"),
        random_state=" NULL ",
    )
    assert masking_cfg.block_size is None
    assert masking_cfg.random_state is None

    with pytest.raises(ValueError, match="random_state"):
        source_feature_jitter_config(random_state=[1])

    with pytest.raises(ValueError, match="block_size"):
        source_feature_masking_config(block_size=[1])
