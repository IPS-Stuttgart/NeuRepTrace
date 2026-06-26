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


@pytest.mark.parametrize("seed", [True, False, np.bool_(True), 1.5, np.nan, -1])
def test_confusion_category_enrichment_rejects_malformed_permutation_seed(seed):
    with pytest.raises(ValueError, match="seed must be a non-negative integer or None"):
        confusion_category_enrichment(
            _predictions(),
            metadata_frame=_metadata(),
            category_columns=("semantic_category",),
            n_permutations=1,
            seed=seed,
        )


@pytest.mark.parametrize("seed", [None, 0, np.int64(3), 4.0])
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
