import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics import confusion_category_enrichment


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_label": [1, 1, 2, 2],
            "predicted_label": [2, 1, 1, 2],
        }
    )


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [1, 2],
            "semantic_category": ["face", "object"],
        }
    )


@pytest.mark.parametrize("seed", [True, False, np.bool_(True), np.asarray(True), np.array([1]), 1.5, np.nan, -1])
def test_confusion_category_enrichment_rejects_malformed_permutation_seed(seed):
    with pytest.raises(ValueError, match="seed must be a non-negative integer or None"):
        confusion_category_enrichment(
            _predictions(),
            metadata_frame=_metadata(),
            category_columns=("semantic_category",),
            n_permutations=1,
            seed=seed,
        )


@pytest.mark.parametrize("seed", [None, 0, np.int64(3), np.asarray(4), 4.0])
def test_confusion_category_enrichment_accepts_integer_like_permutation_seed(seed):
    result = confusion_category_enrichment(
        _predictions(),
        metadata_frame=_metadata(),
        category_columns=("semantic_category",),
        n_permutations=1,
        seed=seed,
    )

    assert len(result) == 1
    assert np.isfinite(result.loc[0, "same_category_permutation_p_value"])


@pytest.mark.parametrize("n_permutations", [True, False, np.bool_(True), np.asarray(True), np.array([1]), 1.5, np.nan, -1])
def test_confusion_category_enrichment_rejects_malformed_permutation_count(n_permutations):
    with pytest.raises(ValueError, match="n_permutations must be a non-negative integer"):
        confusion_category_enrichment(
            _predictions(),
            metadata_frame=_metadata(),
            category_columns=("semantic_category",),
            n_permutations=n_permutations,
            seed=0,
        )


@pytest.mark.parametrize("n_permutations", [None, 0, np.int64(1), np.asarray(1), 2.0])
def test_confusion_category_enrichment_accepts_integer_like_permutation_count(n_permutations):
    result = confusion_category_enrichment(
        _predictions(),
        metadata_frame=_metadata(),
        category_columns=("semantic_category",),
        n_permutations=n_permutations,
        seed=0,
    )

    assert len(result) == 1
    if n_permutations is None or int(n_permutations) == 0:
        assert np.isnan(result.loc[0, "same_category_permutation_p_value"])
    else:
        assert np.isfinite(result.loc[0, "same_category_permutation_p_value"])
