from __future__ import annotations

import pytest

from neureptrace.decode_from_config import _section


def test_decode_from_config_section_allows_missing_and_null_sections() -> None:
    assert _section({}, "outputs") == {}
    assert _section({"outputs": None}, "outputs") == {}
    assert _section({"outputs": {"provenance": False}}, "outputs") == {"provenance": False}


@pytest.mark.parametrize("bad_value", ["", [], False, 0])
def test_decode_from_config_section_rejects_falsey_non_mappings(bad_value) -> None:
    with pytest.raises(ValueError, match="Config section 'outputs' must be a mapping"):
        _section({"outputs": bad_value}, "outputs")
