from __future__ import annotations

import numpy as np
import pytest

import neureptrace.fieldtrip_mat as fieldtrip_mat


def test_fieldtrip_parse_path_tokens_rejects_boolean_indices() -> None:
    bad_values = (
        (True,),
        [np.bool_(False)],
        np.array([True, False], dtype=bool),
        (token for token in ["data", True]),
    )

    for value in bad_values:
        with pytest.raises(ValueError, match="not boolean"):
            fieldtrip_mat.parse_path_tokens(value, fieldtrip_mat.DEFAULT_ROOT_PATH)


def test_fieldtrip_parse_path_tokens_still_accepts_string_fields_and_integer_indices() -> None:
    assert fieldtrip_mat.parse_path_tokens(("outer", 0, "data"), fieldtrip_mat.DEFAULT_ROOT_PATH) == ("outer", 0, "data")
    assert fieldtrip_mat.parse_path_tokens("outer,0,data", fieldtrip_mat.DEFAULT_ROOT_PATH) == ("outer", 0, "data")
