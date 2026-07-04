from pathlib import Path

import pytest

import neureptrace.bushmeg_loso_decode as bushmeg_loso


def test_bushmeg_loso_rejects_negative_max_folds_before_loading_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_folds must be a non-negative integer"):
        bushmeg_loso.run_bushmeg_loso_decode(
            data_dir=tmp_path,
            out_path=tmp_path / "bushmeg_loso.csv",
            participants="1,2",
            normalization="none",
            max_folds=-1,
        )


def test_bushmeg_loso_accepts_none_max_folds_validator() -> None:
    assert bushmeg_loso._non_negative_max_folds(None) is None
    assert bushmeg_loso._non_negative_max_folds("2") == 2
