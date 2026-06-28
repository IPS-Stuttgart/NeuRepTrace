from __future__ import annotations

import pytest

from neureptrace.decoding.source_balancing import source_class_balancing_config


def test_preserve_order_string_false_is_parsed_as_false() -> None:
    cfg = source_class_balancing_config(preserve_order="false")

    assert cfg.preserve_order is False


def test_preserve_order_accepts_common_true_string() -> None:
    cfg = source_class_balancing_config(preserve_order="yes")

    assert cfg.preserve_order is True


def test_preserve_order_rejects_ambiguous_strings() -> None:
    with pytest.raises(ValueError, match="preserve_order must be a boolean"):
        source_class_balancing_config(preserve_order="sometimes")  # type: ignore[arg-type]
