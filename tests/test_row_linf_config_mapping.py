from __future__ import annotations

import pytest

from neureptrace.decoding.row_linf import normalize_train_test_rows_linf


def test_row_linf_mapping_config_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="Unknown row L-infinity config option"):
        normalize_train_test_rows_linf(
            train_features=[[1.0]],
            test_features=[[1.0]],
            config={"epsilon": 1e-6, "unexpected_option": 1.0},
        )
