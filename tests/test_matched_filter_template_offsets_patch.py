from __future__ import annotations

import numpy as np
import pytest

from neureptrace.matched_filter_detection import _template_offsets


def test_template_offsets_do_not_step_past_window_stop():
    offsets = _template_offsets((0.0, 1.0), 0.6)

    assert offsets == pytest.approx([0.0, 0.6])
    assert np.all(offsets <= np.nextafter(1.0, np.inf))


def test_template_offsets_keep_endpoint_lost_to_roundoff():
    offsets = _template_offsets((0.0, 0.3), 0.1)

    assert offsets == pytest.approx([0.0, 0.1, 0.2, 0.3])
