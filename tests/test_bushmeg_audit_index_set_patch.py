from __future__ import annotations

import numpy as np

from neureptrace.bushmeg_all_protocols_audit import _parse_index_set


def test_parse_index_set_accepts_list_like_values() -> None:
    assert _parse_index_set([1, "2", 3.0, "bad", None]) == {1, 2, 3}
    assert _parse_index_set((4, "5.0")) == {4, 5}
    assert _parse_index_set(np.array([6, 7.0, np.nan, "ignored"], dtype=object)) == {6, 7}


def test_parse_index_set_keeps_existing_delimited_string_support() -> None:
    assert _parse_index_set("[1|2; 3,4]") == {1, 2, 3, 4}
    assert _parse_index_set("") == set()
