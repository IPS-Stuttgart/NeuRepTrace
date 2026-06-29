from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_alignment import source_alignment_config
from neureptrace.decoding.unlabeled_calibration_alignment import unlabeled_calibration_alignment_config

_DISABLED_COMPONENT_STRINGS = ("", " ", "\t", " none ", "None", "NONE", " null ", "NULL")



def test_optional_mcca_subject_pca_components_parse_disabled_strings():
    for value in _DISABLED_COMPONENT_STRINGS:
        assert source_alignment_config(method="mcca", mcca_subject_pca_components=value).mcca_subject_pca_components is None
        assert (
            unlabeled_calibration_alignment_config(method="mcca", mcca_subject_pca_components=value).mcca_subject_pca_components
            is None
        )



def test_optional_mcca_subject_pca_components_preserve_numeric_values():
    assert source_alignment_config(method="mcca", mcca_subject_pca_components="8").mcca_subject_pca_components == 8
    assert unlabeled_calibration_alignment_config(method="mcca", mcca_subject_pca_components="8").mcca_subject_pca_components == 8


@pytest.mark.parametrize("value", [False, True])
def test_optional_mcca_subject_pca_components_reject_booleans(value):
    with pytest.raises(ValueError, match="alignment_components"):
        source_alignment_config(method="mcca", mcca_subject_pca_components=value)
    with pytest.raises(ValueError, match="alignment_components"):
        unlabeled_calibration_alignment_config(method="mcca", mcca_subject_pca_components=value)


@pytest.mark.parametrize(
    "value",
    [
        [8],
        (8,),
        {"components": 8},
        np.asarray([8]),
        np.asarray(8),
    ],
)
def test_optional_mcca_subject_pca_components_reject_array_like_values(value):
    with pytest.raises(ValueError, match="mcca_subject_pca_components.*scalar"):
        source_alignment_config(method="mcca", mcca_subject_pca_components=value)
    with pytest.raises(ValueError, match="mcca_subject_pca_components.*scalar"):
        unlabeled_calibration_alignment_config(method="mcca", mcca_subject_pca_components=value)
